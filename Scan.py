import os
import sys
import warnings

import pydicom as dicom
import numpy as np

import sqlalchemy as sq
from sqlalchemy.orm import relationship
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from ._Base import Base

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons

from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import connected_components
from .annotation_distance_metrics import metrics

from scipy.stats import mode

try:
    import configparser
except ImportError:
    import ConfigParser
    configparser = ConfigParser


def _get_config_filename():
    """
    Yields the platform-specific configuration filename
    """
    return 'pylidc.conf' if sys.platform.startswith('win') else '.pylidcrc'


def _get_config_path():
    """
    Yields the path to configuration file
    """
    return os.path.join(os.path.expanduser('~'))


def _get_config_file():
    return os.path.join(_get_config_path(),
                        _get_config_filename())


def _get_dicom_file_path_from_config_file():
    """
    Loads the dicom section of the configuration file
    """
    conf_file = _get_config_file()

    parser = configparser.SafeConfigParser()

    if os.path.exists(conf_file):
        parser.read(conf_file)
        
    try:
        return parser.get(section='dicom', option='path')
    except (configparser.NoSectionError,
            configparser.NoOptionError):
        msg = ("Could not find `dicom` configuration section or "
               " `path` configuration option under that section."
               "A template config file will be written to {}.")
        warnings.warn(msg.format(conf_file))

        parser.add_section('dicom')
        parser.set('dicom', 'path', '')

        with open(conf_file, 'w') as f:
            parser.write(f)

        return parser.get(section='dicom', option='path')


_off_limits = ['id','study_instance_uid','series_instance_uid',
               'patient_id','slice_thickness','pixel_spacing',
               'contrast_used','is_from_initial','sorted_dicom_file_names']


class Scan(Base):
    """
    The Scan model class refers to the top-level XML file from the LIDC.
    A scan has many :class:`pylidc.Annotation` objects, which correspond
    to the `unblindedReadNodule` XML attributes for the scan.

    Attributes
    ==========

    study_instance_uid: string
        DICOM attribute (0020,000D).

    series_instance_uid: string 
        DICOM attribute (0020,000E).

    patient_id: string
        Identifier of the form "LIDC-IDRI-dddd" where dddd is a string of 
        integers.

    slice_thickness: float
        DICOM attribute (0018,0050). Note that this may not be 
        equal to the `slice_spacing` attribute (see below).

    slice_zvals: ndarray
        The "z-values" for the slices of the scan (i.e.,
        the last coordinate of the ImagePositionPatient DICOM attribute)
        as a NumPy array sorted in increasing order.

    slice_spacing: float
        This computed property is the median of the difference
        between the slice coordinates, i.e., `scan.slice_zvals`.

        Note
        ----
        This attribute is typically (but not always!) the
        same as the `slice_thickness` attribute. Furthermore,
        the `slice_spacing` does NOT necessarily imply that all the 
        slices are spaced with spacing (although they often are).

    pixel_spacing: float
        Dicom attribute (0028,0030). This is normally two 
        values. All scans in the LIDC have equal resolutions 
        in the transverse plane, so only one value is used here.

    contrast_used: bool
        If the DICOM file for the scan had any Contrast tag, 
        this is marked as `True`.

    is_from_initial: bool 
        Indicates whether or not this PatientID was tagged as 
        part of the initial 399 release.

    sorted_dicom_file_names: string
        This attribute is no longer used, and can be ignored.

    Example
    -------
    A short example of `Scan` class usage::

        import pylidc as pl

        scans = pl.query(pl.Scan).filter(pl.Scan.slice_thickness <= 1)
        print(scans.count())
        # => 97

        scan = scans.first()
        print(scan.patient_id,
              scan.pixel_spacing,
              scan.slice_thickness,
              scan.slice_spacing)
        # => LIDC-IDRI-0066, 0.63671875, 0.6, 0.5

        print(len(scan.annotations))
        # => 11
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
            super(Scan, self).__setattr__(name,value)
    
    def get_path_to_dicom_files(self):
        """
        Get the path to where the DICOM files are stored for this scan, 
        relative to the root path set in the pylidc configuration file (i.e.,
        `~/.pylidc` in MAC and Linux).
        
        1. In older downloads, the data DICOM data would download as::

               [...]/LIDC-IDRI/LIDC-IDRI-dddd/uid1/uid2/dicom_file.dcm

           where [...] is the base path set in the pylidc configuration
           filee; uid1 is `Scan.study_instance_uid`; and, uid2
           is `Scan.series_instance_uid` .

        2. However, in more recent downloads, the data is downloaded like::

               [...]/LIDC-IDRI/LIDC-IDRI-dddd/???

           where "???" is some unknown folder hierarchy convention used
           by TCIA.

        We first check option 1. Otherwise, we check if the
        "LIDC-IDRI-dddd" folder exists in the root path. If so, then we 
        recursively search the "LIDC-IDRI-dddd" directory until we find
        the correct subfolder that contains a DICOM file with the correct
        `study_instance_uid` and `series_instance_uid`.

        Option 2 is less efficient than 1; however, option 2 is robust.
        """
        dicompath = _get_dicom_file_path_from_config_file()

        if not os.path.exists(dicompath):
            msg = ("Could not establish path to dicom files. Have you "
                   "specified the `path` option in the configuration "
                   "file {}?")
            raise RuntimeError(msg.format(_get_config_file()))

        base = os.path.join(dicompath, self.patient_id)

        if not os.path.exists(base):
            msg = "Couldn't find DICOM files for {} in {}"
            raise RuntimeError(msg.format(self, base))

        path = os.path.join(base,
                            self.study_instance_uid,
                            self.series_instance_uid)

        # Check if old path first. If not found, do recursive search.
        if not os.path.exists(path): # and base exists
            found = False
            for dpath,dnames,fnames in os.walk(base):
                # Skip if no files in current dir.
                if len(fnames) == 0: continue
                
                # Gather up DICOM files in dir (if any).
                dicom_file = [d for d in fnames if d.endswith(".dcm") and not d.startswith(".")]

                # Skip if no DICOM files.
                if len(dicom_file) == 0: continue

                # Grab the first DICOM file in the dir since they should
                # all have the same series/study ids.
                dicom_file = dicom_file[0]

                dimage = dicom.dcmread(os.path.join(dpath, dicom_file))

                seid = str(dimage.SeriesInstanceUID).strip()
                stid = str(dimage.StudyInstanceUID).strip()

                if seid == self.series_instance_uid and \
                   stid == self.study_instance_uid:
                    path = dpath
                    found = True
                    break

            if not found:
                raise IOError("Couldn't find DICOM files for %s."%self)

        return path

    def load_all_dicom_images(self, verbose=True):
        """
        Load all the DICOM images assocated with this scan and return as list.

        Parameters
        ----------
        verbose: bool
            Turn the loading method on/off.

        Example
        -------
        An example::

            import pylidc as pl
            import matplotlib.pyplot as plt

            scan = pl.query(pl.Scan).first()

            images = scan.load_all_dicom_images()
            zs = [float(img.ImagePositionPatient[2]) for img in images]
            print(zs[1] - zs[0], images[0].SliceThickness, scan.slice_thickness)
            
            plt.imshow(images[0].pixel_array, cmap=plt.cm.gray)
            plt.show()

        """
        if verbose: print("Loading dicom files ... This may take a moment.")

        path = self.get_path_to_dicom_files()
        fnames = [fname for fname in os.listdir(path)
                            if fname.endswith('.dcm') and not fname.startswith(".")]
        images = []
        for fname in fnames:
            image = dicom.dcmread(os.path.join(path,fname))

            seid = str(image.SeriesInstanceUID).strip()
            stid = str(image.StudyInstanceUID).strip()

            if seid == self.series_instance_uid and\
               stid == self.study_instance_uid:
                images.append(image)

        # ##############################################
        # Clean multiple z scans.
        #
        # Some scans contain multiple slices with the same `z` coordinate 
        # from the `ImagePositionPatient` tag.
        # The arbitrary choice to take the slice with lesser 
        # `InstanceNumber` tag is made.
        # This takes some work to accomplish...
        zs    = [float(img.ImagePositionPatient[-1]) for img in images]
        inums = [float(img.InstanceNumber) for img in images]
        inds = list(range(len(zs)))
        while np.unique(zs).shape[0] != len(inds):
            for i in inds:
                for j in inds:
                    if i!=j and zs[i] == zs[j]:
                        k = i if inums[i] > inums[j] else j
                        inds.pop(inds.index(k))

        # Prune the duplicates found in the loops above.
        zs     = [zs[i]     for i in range(len(zs))     if i in inds]
        images = [images[i] for i in range(len(images)) if i in inds]

        # Sort everything by (now unique) ImagePositionPatient z coordinate.
        sort_inds = np.argsort(zs)
        images    = [images[s] for s in sort_inds]
        # End multiple z clean.
        # ##############################################

        return images


    def cluster_annotations(self, metric='min', tol=None, factor=0.9,
                            min_tol=1e-1, return_distance_matrix=False,
                            verbose=True):
        """
        Estimate which annotations refer to the same physical 
        nodule in the CT scan. This method clusters all nodule Annotations for
        a Scan by computing a distance measure between the annotations.
        
        Parameters
        ------
        metric: string or callable, default 'min'
            If string, see::

                from pylidc.annotation_distance_metrics import 
                print(metrics metrics.keys())

            for available metrics. If callable, the provided function,
            should take two Annotation objects and return a float, i.e.,
            `isinstance( metric(ann1, ann2), float )`.

        tol: float, default=None
            A distance in millimeters. Annotations are grouped when 
            the minimum distance between their boundary contour points
            is less than `tol`. If `tol = None` (the default), then
            `tol = scan.pixel_spacing` is used.

        factor: float, default=0.9
            If `tol` resulted in any group of annotations with more than
            4 Annotations, then `tol` is multiplied by `factor` and the
            grouping is performed again.

        min_tol: float, default=0.1
            If `tol` is reduced below `min_tol` (see the `factor` parameter),
            then the routine exits because we conclude that the annotation 
            groups cannot be automatically reduced to have groups 
            with each group having `Annotations<=4` (as expected 
            with LIDC data).

        return_distance_matrix: bool, default False
            Optionally return the distance matrix that was used
            to produce the clusters.

        verbose: bool, default=True
            If True, a warning message is printed when `tol < min_tol` occurs.

        Return
        ------
        clusters: list of lists.
            `clusters[i]` is a list of :class:`pylidc.Annotation` objects
            that refer to the same physical nodule in the Scan. `len(clusters)` 
            estimates the number of unique physical nodules in the Scan.

        Note
        ----
        The "distance" matrix, `d[i,j]`, between all Annotations for 
        the Scan is first computed using the provided `metric` parameter.
        Annotations are said to be adjacent when `d[i,j]<=tol`. 
        Annotation groups are determined by finding the connected components 
        of the graph associated with this adjacency matrix.

        Example
        -------
        An example::

            import pylidc as pl
            
            scan = pl.query(pl.Scan).first()
            nodules = scan.cluster_annotations()

            print("This can has %d nodules." % len(nodules))
            # => This can has 4 nodules.
            
            for i,nod in enumerate(nodules):
                print("Nodule %d has %d annotations." % (i+1,len(nod)))
            # => Nodule 1 has 4 annotations.
            # => Nodule 2 has 4 annotations.
            # => Nodule 3 has 1 annotations.
            # => Nodule 4 has 4 annotations.

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
            if tol < min_tol:
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
                          key=lambda cluster: np.mean([ann.centroid[2]
                                                       for ann in cluster]))

        if return_distance_matrix:
            return clusters, D
        else:
            return clusters

    def visualize(self, annotation_groups=None):
        """
        Visualize the scan.

        Parameters
        ----------
        annotation_groups: list of lists of Annotation objects, default=None
            This argument should be supplied by the returned object from
            the `cluster_annotations` method.

        Example
        -------
        An example::

            import pylidc as pl
            
            scan = pl.query(pl.Scan).first()
            nodules = scan.cluster_annotations()
            
            scan.visualize(annotation_groups=nodules)

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
            centroids = [np.array([a.centroid for a in group]).mean(0)
                                          for group in annotation_groups]
            radii = [np.mean([a.diameter/2 for a in group])
                                        for group in annotation_groups]

            arrows = []
            for i in range(nnods):
                r = radii[i]
                c = centroids[i]
                s = '%d Annotations'%len(annotation_groups[i])
                a = ax_image.annotate(s,
                                      xy=(c[1]-r, c[0]-r),
                                      xytext=(c[1]-50, c[0]-50),
                                      bbox=dict(fc='w', ec='r'),
                                      arrowprops=dict(arrowstyle='->',
                                                      edgecolor='r'))
                a.set_visible(False) # flipped on/off by `update` function.
                arrows.append(a)
        
        ax_scan_info = fig.add_axes([0.1, 0.7, 0.3, 0.15]) # l,b,w,h
        ax_scan_info.set_facecolor('w')
        scan_info_table = ax_scan_info.table(cellText=[
                ['Patient ID:', self.patient_id],
                ['Slice thickness:', '%.3f mm' % self.slice_thickness],
                ['Pixel spacing:', '%.3f mm' % self.pixel_spacing],
                ['Manufacturer:', images[current_slice].Manufacturer],
                ['Model name:', images[current_slice].ManufacturerModelName],
                ['Convolution kernel:', images[current_slice].ConvolutionKernel],
            ],
            loc='center', cellLoc='left'
        )
        # Remove the table cell borders.
        for cell in scan_info_table.properties()['children']:
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
                            '%d annotations, near slice %d' \
                                    % (len(g), int(c[2].round()))])
            ann_grps_table = ax_ann_grps.table(cellText=txt, loc='center',
                                               cellLoc='left')
            # Remove cell borders.
            for cell in ann_grps_table.properties()['children']:
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
                    centroid_z = self.slice_zvals[int(centroids[i][2].round())]
                    dist = abs(z - centroid_z)
                    arrows[i].set_visible(dist <= 3*self.slice_spacing)
            fig.canvas.draw_idle()

        sslice.on_changed(update)
        update(None)
        plt.show()
        return sslice

    @property
    def slice_zvals(self):
        """
        The "z-values" for the slices of the scan (i.e.,
        the last coordinate of the ImagePositionPatient DICOM attribute)
        as a NumPy array sorted in increasing order.
        """
        return np.sort([z.val for z in self.zvals])

    @property
    def slice_spacing(self):
        """
        This computes the median of the difference
        between the slice coordinates, i.e., `scan.slice_zvals`.

        Note
        ----
        This attribute is typically (but not always!) the
        same as the `slice_thickness` attribute. Furthermore,
        the `slice_spacing` does NOT necessarily imply that all the 
        slices are spaced with spacing (although they often are).
        """
        return np.median(np.diff(self.slice_zvals))

    @property
    def spacings(self):
        """
        The spacings in the i, j, k image coordinate directions, as a 
        length 3 array.
        """
        return np.array([self.pixel_spacing,
                         self.pixel_spacing,
                         self.slice_spacing])

    def to_volume(self, verbose=True):
        """
        Return the scan as a 3D numpy array volume.
        """
        images = self.load_all_dicom_images(verbose=verbose)

        volume = np.stack(
            [
                x.pixel_array * x.RescaleSlope + x.RescaleIntercept
                for x in images
            ],
            axis=-1,
        ).astype(np.int16)
        return volume
