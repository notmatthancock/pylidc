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

from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import connected_components
from .annotation_distance_metrics import metrics


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

    def cluster_annotations(self, metric='min', tol=None, factor=0.9,
                            return_distance_matrix=False, verbose=True):
        """
        Estimate which annotations refer to the same physical 
        nodule. This method clusters all nodule Annotations for a 
        Scan by computing a distance measure between the annotations.

        metric: string or callable, default 'min'
            If string, see 
                `from pylidc.annotation_distance_metrics import metrics`
                `metrics.keys()`
            for available metrics. If callable, the provided function,
            should take two Annotation objects and return a float, i.e.,
            metric(ann1, ann2).

        tol: float, default None
            A distance in millimeters. Annotations are grouped when 
            the minimum distance between their boundary contour points
            is less than `tol`. The default, None, sets
            `tol = scan.pixel_spacing`. More detail on this is found below.

        factor: float, default=0.9
            If `tol` resulted in any group of nodules with more than
            4 Annotations, then `tol` is multiplied by `factor` and the
            grouping is performed again.

        return_distance_matrix: bool, default False
            Optionally return the distance matrix that was used
            to produce the clusters.

        verbose: bool, default True
            If `tol` is reduced below 1e-1, then we conclude that 
            the nodule groups cannot be automatically reduced to have
            groups with number of Annotations <= 4, and a warning message
            is printed. If verbose=False, this message is not printed.

        returns: clusters, a list of lists.
            `clusters[j]` is a list of Annotation objects deemed
            to refer to the same physical nodule in the Scan.

            `len(clusters)` attempts to estimate (via the specified `tol`)
            the number of unique physical nodules present in this Scan as 
            determined by this overlap method and the tolerance used.

        ---

        More on the `tol` parameter and distance measures:

        The "distance" matrix, `d[i,j]`, between all Annotations for 
        the Scan is first computed using the provided `metric` parameter.

        Annotations are "grouped" or estimated to refer to the same physical
        nodule when `d <= tol`. The groupings are formed by determining 
        the adjacency matrix for the Annotations. Annotations are said to be
        adjacent when `d[i,j] <= tol`. Groups are determined by finding 
        the connected components of the graph associated with 
        this adjacency matrix.
        """
        assert 0 < factor < 1, "`factor` must be in the interval (0,1)."

        if isinstance(metric, str) and metric not in metrics.keys():
            msg = 'Invalid `metric` string. See \n\n'
            msg += '`from pylidc.annotation_distance_metrics import metrics`\n'
            msg += '`print metrics.keys()`\n\n'
            msg += 'for valid `metric` strings.'
            raise ValueError(msg)
        elif not callable(metric):
            metric = metrics[metric]

        N = len(self.annotations)

        tol = self.slice_thickness if tol is None else tol
        assert tol >= 0, "`tol` should be >= 0."

        # Some special cases.
        if   N == 0:
            return []
        elif N == 1:
            return [[self.annotations[0]]]

        D = np.zeros((N,N)) # The distance matrix.

        for i in range(N):
            for j in range(i+1,N):
                D[i,j] = D[j,i] = metric(self.annotations[i],
                                         self.annotations[j])

        adjacency = D <= tol
        nnods, cids = connected_components(adjacency, directed=False)
        ucids = np.unique(cids)
        counts = [(cids==cid).sum() for cid in ucids]

        # Group again with smaller tolerance until there are 
        # no nodules with more than 4 annotations.
        while any([c > 4 for c in counts]):
            tol *= factor
            if tol < 1e-1:
                msg = "Failed to reduce all groups to <= 4 Annotations.\n"
                msg+= "Some nodules may be close and must be grouped manually."
                if verbose: print(msg)
                break
            adjacency = D <= tol
            nnods, cids = connected_components(adjacency, directed=False)
            ucids = np.unique(cids)
            counts = [(cids==cid).sum() for cid in ucids]

        clusters = [[] for _ in range(nnods)]
        for i,cid in enumerate(cids):
            clusters[cid].append(self.annotations[i])

        # Sort the clusters by increasing average z value of centroids.
        # This is really a convienience thing for the `scan.visualize` method.
        clusters = sorted(clusters, 
                          key=lambda cluster: np.mean([ann.centroid()[2]
                                                       for ann in cluster]))

        if return_distance_matrix:
            return clusters, D
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

    def visualize(self, annotation_groups=None):
        """
        Visualize the scan.

        annotation_groups: list of lists of Annotation objects
            This argument should be supplied by the returned object from
            the `cluster_annotations` method.
        """
        images = self.load_all_dicom_images()

        fig = plt.figure(figsize=(16,8))
        current_slice = int( len(images) / 2 )

        ax_image = fig.add_axes([0.5,0.0,0.5,1.0])
        img = ax_image.imshow(images[current_slice].pixel_array,
                              cmap=plt.cm.gray)

        ax_image.set_xlim(-0.5,511.5); ax_image.set_ylim(511.5,-0.5)
        ax_image.axis('off')

        # Add annotation indicators if necessary.
        if annotation_groups is not None:
            nnods = len(annotation_groups)
            centroids = [np.array([a.centroid() for a in group]).mean(0)
                                          for group in annotation_groups]
            radii = [np.mean([a.estimate_diameter()/2 for a in group])
                                        for group in annotation_groups]

            arrows = []
            for i in range(nnods):
                r = radii[i]
                c = centroids[i]
                s = '%d Annotations'%len(annotation_groups[i])
                a = ax_image.annotate(s,
                                      xy=(c[0]-r, c[1]-r),
                                      xytext=(c[0]-50, c[1]-50),
                                      bbox=dict(fc='w', ec='r'),
                                      arrowprops=dict(arrowstyle='->',
                                                      edgecolor='r'))
                a.set_visible(False) # flipped on/off by `update` function.
                arrows.append(a)
        
        ax_scan_info = fig.add_axes([0.1, 0.8, 0.3, 0.1]) # l,b,w,h
        ax_scan_info.set_facecolor('w')
        scan_info_table = ax_scan_info.table(cellText=[
                ['Patient ID:', self.patient_id],
                ['Slice thickness:', '%.3f mm' % self.slice_thickness],
                ['Pixel spacing:', '%.3f mm' % self.pixel_spacing]
            ],
            loc='center', cellLoc='left'
        )
        # Remove the table cell borders.
        for cell in scan_info_table.properties()['child_artists']:
            cell.set_color('w')
        # Add title, remove ticks from scan info.
        ax_scan_info.set_title('Scan Info')
        ax_scan_info.set_xticks([])
        ax_scan_info.set_yticks([])

        # If annotation_groups are provided, give a info table for them.
        if annotation_groups is not None and nnods != 0:
            # The values here were chosen heuristically.
            ax_ann_grps = fig.add_axes([0.1, 0.45-nnods*0.01,
                                        0.3, 0.2+0.01*nnods]) 
            txt = [['Num Nodules:', str(nnods)]]
            for i in range(nnods):
                c = centroids[i]
                g = annotation_groups[i]
                txt.append(['Nodule %d:'%(i+1),
                            '%d annotations, near z=%.2f'%(len(g),c[2])])
            ann_grps_table = ax_ann_grps.table(cellText=txt, loc='center',
                                               cellLoc='left')
            # Remove cell borders.
            for cell in ann_grps_table.properties()['child_artists']:
                cell.set_color('w')
            # Add title, remove ticks from scan info.
            ax_ann_grps.set_title('Nodule Info')
            ax_ann_grps.set_xticks([])
            ax_ann_grps.set_yticks([])


        # Add the widgets.
        ax_slice = fig.add_axes([0.1, 0.1, 0.3, 0.05])
        ax_slice.set_facecolor('w')
        z = float(images[current_slice].ImagePositionPatient[-1])
        sslice = Slider(ax_slice, 'Z: %.3f'%z, 0, len(images)-1,
                         valinit=current_slice, valfmt=u'Slice: %d')

        def update(_):
            # Update image itself.
            current_slice = int(sslice.val)
            img.set_data(images[current_slice].pixel_array)

            # Update `z` label.
            z = float(images[current_slice].ImagePositionPatient[-1])
            sslice.label.set_text('Z: %.3f' % z)

            # Show annotation labels if possible.
            if annotation_groups is not None:
                for i in range(len(annotation_groups)):
                    dist = abs(z-centroids[i][2])
                    arrows[i].set_visible(dist <= 3*self.slice_thickness)

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
