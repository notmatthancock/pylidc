import sqlalchemy as sq
from sqlalchemy.orm import relationship
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from ._Base import Base

import os, warnings
import dicom
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons
from scipy.spatial.distance import squareform
from scipy.cluster import hierarchy


# Load the configuration file and get the dicom file path.

try:
    import configparser
except ImportError:
    import ConfigParser
    configparser = ConfigParser

cfgpath   = os.path.join(os.path.expanduser('~'), '.pylidcrc')
dicompath = None
warndpath = True

if os.path.exists(cfgpath):
    cp = configparser.SafeConfigParser()
    cp.read(cfgpath)
    if cp.has_option('dicom', 'path'):
        dicompath = cp.get('dicom', 'path')
    if cp.has_option('dicom', 'warn'):
        warndpath = cp.get('dicom', 'warn') == 'True'

if dicompath is None:
    dpath_msg = \
    '\n\n`.pylidcrc` configuration file does not exist ' +  \
    'or path is not set. CT images will not be viewable.\n' + \
    ('The file, `.pylidcrc`, should exist in %s. '%os.path.expanduser('~')) + \
    'This file should have format:\n\n' + \
    '[dicom]\n' + \
    'path = /path/to/dicom/data/LIDC-IDRI\n' + \
    'warn = True\n\n' + \
    'Set `warn` to `False` to suppress this message.\n'
    if warndpath:
        warnings.warn(dpath_msg)

_off_limits = ['id','study_instance_uid','series_instance_uid',
               'patient_id','slice_thickness','pixel_spacing',
               'contrast_used','is_from_initial','sorted_dicom_file_names']

class Scan(Base):
    """
    The Scan model class refers to the top-level XML file from the LIDC.
    A scan has many annotations, each of which is the `unblindedReadNodule`
    for the scan.

    Attributes:
        study_instance_uid: string
        series_instance_uid: string 

        patient_id: string
            Identifier of the from `LIDC-IDRI-#`

        slice_thickness: float
            Dicom attribute, `(0018,0050)`. Note that this may not be 
            equal to `spacing_betwee_slices`.

        pixel_spacing: float
            Dicom attribute, `(0028,0030)`. This is normally two 
            values. All scans in the LIDC have equal resolutions 
            in the transverse plane, so only one value is taken here.

        contrast_used: bool
            If the dicom file for the scan had any Contrast tag, 
            this is marked as `True`.

        is_from_initial: bool 
            Indicates whether or not this PatientID was tagged as 
            part of the initial 399 release.

        sorted_dicom_file_names: string
            This is a string containing a comma-separated list 
            like `[number].dcm`. It is a list of the dicom file 
            names for this scan in order of increasing z-coordinate 
            of dicom attribute, `(0020,0032)`. In rare cases where 
            a scan includes multiple files with the same z-coordinate, 
            the one with the lesser `InstanceNumber` is used.

    Example:
        >>> import pylidc as pl
        >>> qu = pl.query(pl.Scan).filter(pl.Scan.slice_thickness <= 1)
        >>> print(qu.count())
        >>> # => 97
        >>> scan = qu.first()
        >>> print(scan.patient_id, scan.pixel_spacing, scan.slice_thickness)
        >>> # => LIDC-IDRI-0066, 0.63671875, 0.6
        >>> print(len(scan.annotations))
        >>> # => 11
    """
    __tablename__           = 'scans'
    id                      = sq.Column('id', sq.Integer, primary_key=True)
    study_instance_uid      = sq.Column('study_instance_uid', sq.String)
    series_instance_uid     = sq.Column('series_instance_uid', sq.String)
    patient_id              = sq.Column('patient_id', sq.String)
    slice_thickness         = sq.Column('slice_thickness', sq.Float)
    pixel_spacing           = sq.Column('pixel_spacing', sq.Float)
    contrast_used           = sq.Column('contrast_used', sq.Boolean)
    is_from_initial         = sq.Column('is_from_initial', sq.Boolean)
    sorted_dicom_file_names = sq.Column('sorted_dicom_file_names', sq.String)

    def __repr__(self):
        return "Scan(id=%d,patient_id=%s)" % (self.id,self.patient_id)

    def __setattr__(self, name, value):
        if name in _off_limits:
            msg = "Trying to assign read-only Scan object attribute \
                   `%s` a value of `%s`." % (name,value)
            raise ValueError(msg)
        else:
            super(Scan,self).__setattr__(name,value)
    
    def get_path_to_dicom_files(self, checkpath=True):
        """
        Get the path to where the dicom files are stored for this scan, 
        relative to the root path set in the your configuration file.

        Example:
            >>> scan = pl.query(pl.Scan).first()
            >>> print(scan.get_path_to_dicom_files())
            >>> # => /data/storage/path/LIDC-IDRI/LIDC-IDRI-0078/1.3.6.1.4.1.14519.5.2.1.6279.6001.339170810277323131167631068432/1.3.6.1.4.1.14519.5.2.1.6279.6001.303494235102183795724852353824
        """
        if dicompath is None:
            raise EnvironmentError(dpath_msg)
        path = os.path.join(dicompath,
                            self.patient_id,
                            self.study_instance_uid,
                            self.series_instance_uid)
        errstr = \
        "The path:\n\n %s \n\n doesn't exist.\n\
        Does the folder exists there? Have you set set the root path to \n\
        your dicom folder, using `pylidc.set_path_to_dicom_files()`?" % path 

        if checkpath:
            assert os.path.exists(path), errstr
        return path

    def annotations_with_matching_overlap(self, tol=0.5,
                                          return_overlap_scores=False):
        """
        TODO: This function should ideally match match nodule annotations 
        not just on overlap, but on centroid distance, too.

        Find which annotations refer to the same nodule. This method 
        clusters nodule annotations for this scan by comparing the 
        overlap between contour annotations. Specifically, a pairwise 
        distance between annotation[i] with annotation[j] is formed 
        by computing:
        
        1 - ( size_of_intersection( annotation[i], annotation[j] ) / size_of_union( annotation[i], annotation[j] ) ).

        tol: float, default 0.5
            A value between 0 and 1. The tolerance whereby annotations 
            are grouped. A high tolerance means annotations with less 
            overlap are grouped. A low tolerance requires that annotations 
            must have high overlap to be grouped.

        returns: clusters, a list of nodule groups.
            `clusters[j]` is a list of nodule annotations grouped by 
            above. It should be the case that `len(clusters[j])` is 
            always be between 1 and 4 because a max of 4 radiologists
            could have annotated a single nodule. `len(clusters)` is 
            the number of unique nodules present in this scan as 
            determined by this overlap method and the tolerance used.
        """
        assert 0 <= tol <= 1., "tolerance should be between 0 and 1"

        # Some special cases.
        if len(self.annotations)==0:
            return []
        elif len(self.annotations)==1:
            return [[self.annotations[0]]]

        sets = [ann._as_set() for ann in self.annotations]
        N = len(self.annotations)
        A = np.zeros((N,N)) # A is the affinity matrix
        for i in range(N):
            for j in range(i,N):
                if i==j:
                    A[i,j] = 1.
                else:
                    A[i,j] = len(sets[i].intersection(sets[j])) 
                    A[i,j] = A[i,j] / float(len(sets[i].union(sets[j])))
                    A[j,i] = A[i,j]
        D = 1-A # convert the affinity matrix to a distance matrix.

        # Cluster using the overlap distance.
        cluster_labels = hierarchy.fcluster(hierarchy.linkage(D),
                                            tol, 'distance')
        clusters = []
        # indexing for hierachy.fcluster starts at 1.
        for label in range(1,cluster_labels.max()+1):
            clusters.append([])
            for i,l in enumerate(cluster_labels):
                if l==label:
                    clusters[-1].append( self.annotations[i] )
        if return_overlap_scores:
            return A, clusters
        else:
            return clusters

    def load_all_dicom_images(self, verbose=True):
        """
        Load all the DICOM images assocated with this scan and return as list.

        The listed is sorted according to the Scan object's
        `sorted_dicom_file_names` attribute, which *should* load the images
        in order of increasing z index of the `ImagePositionPatient` attribute
        of the DICOM files.

        Example:
            >>> scan = pl.query(pl.Scan).first()
            >>> images = scan.load_all_dicom_images()
            >>> zs = [float(img.ImagePositionPatient[2]) for img in images]
            >>> print(zs[1] - zs[0], img.SliceThickness, scan.slice_thickness)
            >>>
            >>> import matplotlib.pyplot as plt
            >>> plt.imshow( images[0].pixel_array, cmap=plt.cm.gray )
            >>> plt.show()
        """
        path = self.get_path_to_dicom_files()

        if verbose: print("Loading dicom files ... This may take a moment.")

        fnames = [fname for fname in os.listdir(path)]
        fnames = [fname for fname in fnames if fname.endswith('.dcm')]
        sorted_fnames = self.sorted_dicom_file_names.split(',')
        
        # Some sets have the dicom files padded with 
        # zeros in front, some don't apparently.
        if all([(len(fname)==len(fnames[0])) for fname in fnames]):
            L = len(fnames[0]) - len('.dcm') # Just being explicit here.
            str_format = ("%0"+str(L)+"d.dcm")
            for i,fname in enumerate(sorted_fnames):
                sorted_fnames[i] = str_format % int(fname.split('.')[0])

        images = []
        for dicom_file_name in sorted_fnames:
            with open(os.path.join(path, dicom_file_name), 'rb') as f:
                images.append( dicom.read_file(f) )
        return images

    def visualize(self):
        """
        Visualize the scan without any annotations.
        """
        images = self.load_all_dicom_images()

        fig = plt.figure(figsize=(16,8))
        current_slice = int( len(images) / 2 )

        ax_image = fig.add_axes([0.5,0.0,0.5,1.0])
        img = ax_image.imshow(images[current_slice].pixel_array,
                              cmap=plt.cm.gray)

        ax_image.set_xlim(-0.5,511.5); ax_image.set_ylim(511.5,-0.5)
        ax_image.axis('off')
        
        ax_scan_info = fig.add_axes([0.1, 0.8, 0.3, 0.1])
        ax_scan_info.set_facecolor('w')
        scan_info_table = ax_scan_info.table(
            cellText=[
                ['Patient ID:', self.patient_id],
                ['Slice thickness:', '%.3f mm' % self.slice_thickness],
                ['Pixel spacing:', '%.3f mm' % self.pixel_spacing]
            ],
            loc='center', cellLoc='left'
        )
        # Remove the table cell borders
        for cell in scan_info_table.properties()['child_artists']:
            cell.set_color('w')
        
        ax_scan_info.set_title('Scan Info')
        ax_scan_info.set_xticks([])
        ax_scan_info.set_yticks([])

        # Add the widgets.
        ax_slice = fig.add_axes([0.1, 0.1, 0.3, 0.05])
        ax_slice.set_facecolor('w')
        sslice = Slider(ax_slice,
            'Z: %.3f'%float(images[current_slice].ImagePositionPatient[-1]),
             0, len(images)-1, valinit=current_slice, valfmt=u'Slice: %d')

        def update(_):
            # Update image itself.
            current_slice = int(sslice.val)
            img.set_data(images[current_slice].pixel_array)
            sslice.label.set_text(\
            'Z: %.3f'%float(images[current_slice].ImagePositionPatient[-1]))
            fig.canvas.draw_idle()

        sslice.on_changed(update)
        update(None)
        plt.show()

    def to_volume(self):
        """
        Return the scan as a 3D numpy array volume.
        """
        path = self.get_path_to_dicom_files()

        images = []
        for dicom_file_name in self.sorted_dicom_file_names.split(','):
            with open( os.path.join(path, dicom_file_name) ) as f:
                images.append( dicom.read_file(f) )

        volume = np.zeros((512,512,len(images)))
        for i in range(len(images)):
           volume[:,:,i] = images[i].pixel_array
        return volume
