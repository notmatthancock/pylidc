import os, sys, re
import numpy as np
from xml.etree import ElementTree
import dicom

assert not os.path.exists(os.path.join(os.path.pardir, 'pylidc.sqlite')), "`pylidc.sqlite` already exists. Aborting."


# Change these. The dicom path should end with `LIDC-IDRI`, and the xml path should end with `tcia-lidc-xml`.
dicom_root_path = '/media/matt/fatty/Data/LIDC-IDRI'
# The 161-resubmitted-... file should replace the 161.xml file in this directory. I replace it by overwriting 161.xml while taking the name 161.xml. I'm not sure if this makes a difference.
xml_root_path = '/home/matt/Downloads/tcia-lidc-xml'
os.listdir(xml_root_path)




#################################################################
# Populate the database
#################################################################

# Import the module so we can use the models defined therein.
import pylidc as pl

# This line actually creates the database, `pylidc.db`.
pl._Base.Base.metadata.create_all(pl._engine)

# Add the dicom path as a configuration, so that the path to
# dicom files persists across session. This is done so we can write
# functions for visualizing annotations on top of CT data.
pl._session.add(pl._Configuration(key='path_to_dicom_files', value=dicom_root_path))

characteristic_names =\
['subtlety',
'internalStructure',
'calcification',
'sphericity',
'margin',
'lobulation',
'spiculation',
'texture',
'malignancy']
  
xml_paths = np.load(os.path.join(os.path.curdir,'metadata','xml_paths.pkl'))
init_ids  = np.load(os.path.join(os.path.curdir,'metadata','initial_release_ids.pkl'))
tcia_ids  = np.load(os.path.join(os.path.curdir,'metadata','tcia_ids.pkl'))

def load_xml(path):
    with open(path,'r') as f:
        # Parse the xml file into an element tree so we can find elements easy.
        # ElementTree doesn't like the xmlns=... attribute for some reason, so
        # we blank it out before creating the element tree from the file string.
        xml = ElementTree.fromstring( re.sub(' xmlns="[^"]+"', '', f.read()) )
    return xml

assert os.path.exists(xml_root_path), "`xml_root_path` provided doesn't exist."

def load_dcm(path):
    with open(path,'r') as f:
        img = dicom.read_file(f)
    return img

assert os.path.exists(dicom_root_path), "`dicom_root_path` provided doesn't exist."

for count,xml_base_path in enumerate(xml_paths):
    sys.stdout.write('Populating: %04d / 1018\r' % (count+1))
    sys.stdout.flush()

    # Create the xml tree.
    xml_tree = load_xml(os.path.join(xml_root_path, xml_base_path+'.xml'))
    study_instance_uid  = xml_tree.find('ResponseHeader').find('StudyInstanceUID').text
    series_instance_uid = xml_tree.find('ResponseHeader').find('SeriesInstanceUid').text

    # Load the dicom images into memory.
    dcm_path = os.path.join(dicom_root_path, tcia_ids[xml_base_path], study_instance_uid, series_instance_uid) 
    dcm_file_paths = os.listdir(dcm_path)
    dcm_imgs = [load_dcm(os.path.join(dcm_path,dcm_file_path)) for dcm_file_path in dcm_file_paths if dcm_file_path.endswith('.dcm')]

    # ##############################################
    # Clean multiple z scans.
    #
    # Some scans contain multiple slices with the same `z` coordinate from the `ImagePositionPatient` tag.
    # I'm making the arbitrary choice to take the slice with lesser `InstanceNumber` tag.
    # This takes some work to accomplish...
    zs    = [float(img.ImagePositionPatient[-1]) for img in dcm_imgs]
    inums = [float(img.InstanceNumber) for img in dcm_imgs]
    inds = range(len(zs))
    while np.unique(zs).shape[0] != len(inds):
        for i in inds:
            for j in inds:
                if i!=j and zs[i] == zs[j]:
                    k = i if inums[i] > inums[j] else j
                    inds.pop(inds.index(k))

    # Prune the duplicates found in the loops above.
    zs             = [zs[i] for i in range(len(zs)) if i in inds]
    dcm_file_paths = [dcm_file_paths[i] for i in range(len(dcm_file_paths)) if i in inds]
    dcm_imgs       = [dcm_imgs[i] for i in range(len(dcm_imgs)) if i in inds]

    # Now sort everything by (now unique) ImagePositionPatient z coordinate.
    sort_inds      = np.argsort(zs)
    dcm_imgs       = [dcm_imgs[s] for s in sort_inds]
    dcm_file_paths = [dcm_file_paths[s] for s in sort_inds]
    zs             = [zs[s] for s in sort_inds]
    # End multiple z clean.
    # ##############################################

    # This will be a handy field for the `Contour` object.
    z_to_dcm_path = {float(dcm_imgs[i].ImagePositionPatient[-1]): dcm_file_paths[i] for i in range(len(dcm_imgs))}

    # Create the scan object.
    scan = pl.Scan(
        study_instance_uid      = study_instance_uid,
        series_instance_uid     = series_instance_uid,
        patient_id              = tcia_ids[xml_base_path],
        slice_thickness         = float(img.SliceThickness),
        pixel_spacing           = float(img.PixelSpacing[0]),
        is_from_initial         = (tcia_ids[xml_base_path] in init_ids),
        contrast_used           = any(['Contrast' in d for d in dir(img)]),
        sorted_dicom_file_names = ",".join(dcm_file_paths)
    )

    scan.annotations = []
    # Now we add the annotations for nodules >3mm to the scan.
    for session in xml_tree.findall('readingSession'):
        for ann in session.findall('unblindedReadNodule'):
            # If the nodule annotation doesn't contain a characteristics
            # field, we continue, since we only care about the large nodule class.
            chars = ann.find('characteristics')
            if chars is None or len(chars)==0:
                continue
            # There are a few cases where a small nodule actually *does* have a characteristics field, but all the characteristics are 0 or blank.
            bad_annotation = False
            for c in chars:
                if c.text == '0' or c.text == '' or c.text == None:
                    bad_annotation = True
                    break
            if bad_annotation:
                continue
            
            # Ok, now we know that this a large class nodule annotation.
            annotation = pl.Annotation(_nodule_id=ann.find('noduleID').text)
            for char_name in characteristic_names:
                setattr(annotation, char_name, int(chars.find(char_name).text))

            # Now we need to go one last level and add all the contours for each annotation.
            annotation.contours = []
            rois = ann.findall('roi')
            for roi in rois:
                # If the ROI only has a single edgemap, 
                # this means the "contour" is just a single dot,
                # which we consider a stray mark and therefore, ignore.
                if len(roi.findall('edgeMap')) <= 1:
                    continue
                # The coords line looks cryptic, but all we're doing is taking all the edgmaps and putting
                # them into a single string with x,y points separated by a newline.
                z_pos = float(roi.find('imageZposition').text) 
                contour = pl.Contour(
                    image_z_position = z_pos,
                    dicom_file_name  = z_to_dcm_path[ z_pos ],
                    inclusion        = roi.find('inclusion').text == 'TRUE',
                    coords           = "\n".join([(em.find('xCoord').text+','+em.find('yCoord').text) for em in roi.findall('edgeMap')])
                )
                annotation.contours.append(contour)
            scan.annotations.append(annotation)
        pl._session.add(scan)
print("")

# Finally, we add the configuration option to set where the dicom files are stored.
# This path is initialized to an empty string, and it's updated when later on when
# the annotation viewing function is called.
pl._session.commit()
