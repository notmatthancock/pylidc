import os, warnings
import sqlalchemy as sq
from sqlalchemy.orm import relationship
from ._Base import Base
from .Scan import Scan

import numpy as np
import matplotlib.pyplot as plt

# For contour to boolean mask function.
import matplotlib.path as mplpath

# For CT volume visualizer.
from matplotlib.patches import Rectangle
from matplotlib.widgets import Slider, Button, CheckButtons

# For diameter estimation.
from scipy.spatial.distance import pdist,squareform
from scipy.interpolate import RegularGridInterpolator

# For 3D visualizer.
from skimage.measure import mesh_surface_area

try:
    from skimage.measure import marching_cubes
except ImportError:
    # Old version compatible since marching_cubes replaced with marchin_cubes_lewiner in skimage 0.19.0
    from skimage.measure import marching_cubes_lewiner as marching_cubes
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import Delaunay
from scipy.ndimage.morphology import distance_transform_edt as dtrans


feature_names = \
   ('subtlety',
    'internalStructure',
    'calcification',
    'sphericity',
    'margin',
    'lobulation',
    'spiculation',
    'texture',
    'malignancy')

_off_limits = ['id','scan_id','_nodule_id','scan'] + \
              list(feature_names)

viz3dbackends = ['matplotlib', 'mayavi']

class Annotation(Base):
    """
    The Nodule model class holds the information from a single physicians 
    annotation of a nodule >= 3mm class with a particular scan. A nodule 
    has many contours, each of which refers to the contour drawn for 
    nodule in each scan slice.  

    Attributes
    ----------
    subtlety: int, range = {1,2,3,4,5}
        Difficulty of detection. Higher values indicate easier detection.

        1. 'Extremely Subtle'
        2. 'Moderately Subtle'
        3. 'Fairly Subtle'
        4. 'Moderately Obvious'
        5. 'Obvious'

    internalStructure: int, range = {1,2,3,4}
        Internal composition of the nodule.

        1. 'Soft Tissue'
        2. 'Fluid'
        3. 'Fat'
        4. 'Air'

    calcification: int, range = {1,2,3,4,6}
        Pattern of calcification, if present.

        1. 'Popcorn'
        2. 'Laminated'
        3. 'Solid'
        4. 'Non-central'
        5. 'Central'
        6. 'Absent'

    sphericity: int, range = {1,2,3,4,5}
        The three-dimensional shape of the nodule in terms of its roundness.

        1. 'Linear'
        2. 'Ovoid/Linear'
        3. 'Ovoid'
        4. 'Ovoid/Round'
        5. 'Round'

    margin: int, range = {1,2,3,4,5}
        Description of how well-defined the nodule margin is.

        1. 'Poorly Defined'
        2. 'Near Poorly Defined'
        3. 'Medium Margin'
        4. 'Near Sharp'
        5. 'Sharp'

    lobulation: int, range = {1,2,3,4,5}
        The degree of lobulation ranging from none to marked

        1. 'No Lobulation'
        2. 'Nearly No Lobulation'
        3. 'Medium Lobulation'
        4. 'Near Marked Lobulation'
        5. 'Marked Lobulation'

    spiculation: int, range = {1,2,3,4,5}
        The extent of spiculation present.

        1. 'No Spiculation'
        2. 'Nearly No Spiculation'
        3. 'Medium Spiculation'
        4. 'Near Marked Spiculation'
        5. 'Marked Spiculation'

    texture: int, range = {1,2,3,4,5}
        Radiographic solidity: internal texture (solid, ground glass, 
        or mixed). 

        1. 'Non-Solid/GGO'
        2. 'Non-Solid/Mixed'
        3. 'Part Solid/Mixed'
        4. 'Solid/Mixed'
        5. 'Solid'

    malignancy: int, range = {1,2,3,4,5}
        Subjective assessment of the likelihood of
        malignancy, assuming the scan originated from a 60-year-old male 
        smoker. 

        1. 'Highly Unlikely'
        2. 'Moderately Unlikely'
        3. 'Indeterminate'
        4. 'Moderately Suspicious'
        5. 'Highly Suspicious'

    Example
    -------
    A short usage example for the Annotation class::

        import pylidc as pl

        # Get the first annotation with spiculation value greater than 3.
        ann = pl.query(pl.Annotation)\\
                .filter(pl.Annotation.spiculation > 3).first()
        
        print(ann.spiculation)
        # => 4
        
        # Each nodule feature has a corresponding property 
        # to print the semantic value.
        print(ann.Spiculation)
        # => Medium-High Spiculation
        
        ann = anns.first()
        print("%.2f, %.2f, %.2f" % (ann.diameter,
                                    ann.surface_area,
                                    ann.volume))
        # => 17.98, 1221.40, 1033.70
    """
    __tablename__ = 'annotations'
    id            = sq.Column('id', sq.Integer, primary_key=True)
    scan_id       = sq.Column(sq.Integer, sq.ForeignKey('scans.id'))
    scan          = relationship('Scan', back_populates='annotations')
    _nodule_id    = sq.Column('_nodule_id', sq.String)

    # Physician-assigned diagnostic attributes.
    subtlety          = sq.Column('subtlety',          sq.Integer)
    internalStructure = sq.Column('internalStructure', sq.Integer)
    calcification     = sq.Column('calcification',     sq.Integer)
    sphericity        = sq.Column('sphericity',        sq.Integer)
    margin            = sq.Column('margin',            sq.Integer)
    lobulation        = sq.Column('lobulation',        sq.Integer)
    spiculation       = sq.Column('spiculation',       sq.Integer)
    texture           = sq.Column('texture',           sq.Integer)
    malignancy        = sq.Column('malignancy',        sq.Integer)

    def __repr__(self):
        return "Annotation(id=%d,scan_id=%d)" % (self.id, self.scan_id)

    def __setattr__(self, name, value):
        if name in _off_limits:
            msg = "Trying to assign read-only Annotation object attribute \
                   `%s` a value of `%s`." % (name,value)
            raise ValueError(msg)
        else:
            super(Annotation,self).__setattr__(name,value)

    ####################################
    # { Begin semantic attribute functions

    @property
    def Subtlety(self):
        """Semantic interpretation of `subtlety` value as string."""
        s = self.subtlety
        assert s in range(1,6), "Subtlety score out of bounds."
        if   s == 1: return 'Extremely Subtle'
        elif s == 2: return 'Moderately Subtle'
        elif s == 3: return 'Fairly Subtle'
        elif s == 4: return 'Moderately Obvious'
        elif s == 5: return 'Obvious'

    @property
    def InternalStructure(self):
        """Semantic interpretation of `internalStructure` value as string."""
        s = self.internalStructure
        assert s in range(1,5), "Internal structure score out of bounds."
        if   s == 1: return 'Soft Tissue'
        elif s == 2: return 'Fluid'
        elif s == 3: return 'Fat'
        elif s == 4: return 'Air'

    @property
    def Calcification(self):
        """Semantic interpretation of `calcification` value as string."""
        s = self.calcification
        assert s in range(1,7), "Calcification score out of bounds."
        if   s == 1: return 'Popcorn'
        elif s == 2: return 'Laminated'
        elif s == 3: return 'Solid'
        elif s == 4: return 'Non-central'
        elif s == 5: return 'Central'
        elif s == 6: return 'Absent'

    @property
    def Sphericity(self):
        """Semantic interpretation of `sphericity` value as string."""
        s = self.sphericity
        assert s in range(1,6), "Sphericity score out of bounds."
        if   s == 1: return 'Linear'
        elif s == 2: return 'Ovoid/Linear'
        elif s == 3: return 'Ovoid'
        elif s == 4: return 'Ovoid/Round'
        elif s == 5: return 'Round'

    @property
    def Margin(self):
        """Semantic interpretation of `margin` value as string."""
        s = self.margin
        assert s in range(1,6), "Margin score out of bounds."
        if   s == 1: return 'Poorly Defined'
        elif s == 2: return 'Near Poorly Defined'
        elif s == 3: return 'Medium Margin'
        elif s == 4: return 'Near Sharp'
        elif s == 5: return 'Sharp'

    @property
    def Lobulation(self):
        """Semantic interpretation of `lobulation` value as string."""
        s = self.lobulation
        assert s in range(1,6), "Lobulation score out of bounds."
        if   s == 1: return 'No Lobulation'
        elif s == 2: return 'Nearly No Lobulation'
        elif s == 3: return 'Medium Lobulation'
        elif s == 4: return 'Near Marked Lobulation'
        elif s == 5: return 'Marked Lobulation'

    @property
    def Spiculation(self):
        """Semantic interpretation of `spiculation` value as string."""
        s = self.spiculation
        assert s in range(1,6), "Spiculation score out of bounds."
        if   s == 1: return 'No Spiculation'
        elif s == 2: return 'Nearly No Spiculation'
        elif s == 3: return 'Medium Spiculation'
        elif s == 4: return 'Near Marked Spiculation'
        elif s == 5: return 'Marked Spiculation'

    @property
    def Texture(self):
        """Semantic interpretation of `texture` value as string."""
        s = self.texture
        assert s in range(1,6), "Texture score out of bounds."
        if   s == 1: return 'Non-Solid/GGO'
        elif s == 2: return 'Non-Solid/Mixed'
        elif s == 3: return 'Part Solid/Mixed'
        elif s == 4: return 'Solid/Mixed'
        elif s == 5: return 'Solid'

    @property
    def Malignancy(self):
        """Semantic interpretation of `malignancy` value as string."""
        s = self.malignancy
        assert s in range(1,6), "Malignancy score out of bounds."
        if   s == 1: return 'Highly Unlikely'
        elif s == 2: return 'Moderately Unlikely'
        elif s == 3: return 'Indeterminate'
        elif s == 4: return 'Moderately Suspicious'
        elif s == 5: return 'Highly Suspicious'

    # } End attribute functions
    ####################################

    def feature_vals(self, return_str=False):
        """
        Return all feature values as a numpy array in the order 
        presented in `feature_names`.

        Parameters
        ----------
        return_str: bool, default=False
            If True, a list of strings is also returned, corresponding
            to the meaning of each numerical feature value.

        Return
        ------
            fvals[, fstrs]: array[, list of strings]
                `fvals` is an array of numerical values corresponding to the 
                numerical feature values for the annotation. `fstrs` is a 
                list of semantic string interpretations of the numerical 
                values given in `fvals`.
        """
        fvals = np.array([getattr(self,f) for f in feature_names])
        if return_str:
            caps = [f.title() for f in feature_names]
            k = caps.index('Internalstructure')
            caps[k] = 'InternalStructure'
            return fvals, [getattr(self, c) for c in caps]
        else:
            return fvals

    def print_formatted_feature_table(self):
        """
        Print all feature values as a string table.
        """
        fnames = feature_names
        fvals, fstrings = self.feature_vals(True)

        print('%-18s   %-24s   %-2s'%('Feature', 'Meaning','#'))
        print('%-18s   %-24s   %-2s' % ('-', '-', '-'))

        for i in range(len(fnames)):
            print('%-18s | %-24s | %-2d'%(fnames[i].title(), 
                                          fstrings[i], fvals[i]))

    def bbox(self, pad=None):
        """
        Returns a tuple of Python `slice` objects that can be used to index
        into the image volume corresponding to the extent of the
        (padded) bounding box.

        Parameters
        ----------
        pad: int, list of ints, or float, default=None
            * If None (default), then no padding is used.
            * If an integer is provided, then the bounding box is padded
              uniformly by this integer amount.
            * If a list of integers is provided, then it is of the form::

                  [(i1,i2), (j1,j2), (k1,k2)]

              and indicates the pad amounts along each coordinate axis.
            * If a float is provided, then the slices are padded such
              that the bounding box occupies at least `pad` physical units
              (using the corresponding scan `pixel_spacing` and `slice_spacing`
              parameters). This means the returned Slice indices will
              yield a bounding box that is at least `pad` millimeters along
              each coordinate axis direction.

        Note
        ----
        In the various `pad` cases above, borders are handled so that if a 
        pad beyond the image borders is requested, then it is set 
        to the maximum (or minimum, depending on the direction)
        possible index.

        Return
        ------
        bb: 3-tuple of Python `slice` objects.
            `bb` is the corresponding bounding box (with desired padding) 
            in the CT image volume. `bb[i]` is a slice corresponding
            to the the extent of the bounding box along the 
            coordinate axis `i`.

        Example
        -------

        The example below illustrates the various `pad` argument types::

            import pylidc as pl
            
            ann = pl.query(pl.Annotation).first()
            vol = ann.scan.to_volume()
            
            print ann.bbox()
            # => (slice(151, 185, None), slice(349, 376, None), slice(44, 50, None))
            
            print(vol[ann.bbox()].shape)
            # => (34, 27, 6)
            
            print(vol[ann.bbox(pad=2)].shape)
            # => (38, 31, 10)
            
            print(vol[ann.bbox(pad=[(1,2), (3,0), (2,4)])].shape)
            # => (37, 30, 12)
            
            print(max(ann.bbox_dims()))
            # => 21.45
            
            print(vol[ann.bbox(pad=30.0)].shape)
            # => (48, 49, 12)
            
            print(ann.bbox_dims(pad=30.0))
            # => [30.55, 31.200000000000003, 33.0]
        """
        # Error checking ...
        if pad is not None:
            if not isinstance(pad, (int, list, float)):
                raise TypeError("`pad` is incorrect type.")
            if isinstance(pad, list):
                if len(pad) != 3:
                    raise ValueError("`pad` list length should be 3.")
                for p in pad:
                    msg = "`pad` list elements should be (int, int)"
                    if len(p) != 2:
                        raise ValueError(msg)
                    if not isinstance(p[0], int) or not isinstance(p[1], int):
                        raise TypeError(msg)

        # The index limits for the scan.
        limits = [(0,511), (0,511), (0,self.scan.slice_zvals.shape[0]-1)]

        cmatrix = self.contours_matrix
        imin,jmin,kmin = cmatrix.min(axis=0)
        imax,jmax,kmax = cmatrix.max(axis=0)

        # Adding the padding for each respective case, handling the
        # borders as needed.
        if isinstance(pad, int):
            imin = max(imin-pad, limits[0][0])
            imax = min(imax+pad, limits[0][1])
            jmin = max(jmin-pad, limits[1][0])
            jmax = min(jmax+pad, limits[1][1])
            kmin = max(kmin-pad, limits[2][0])
            kmax = min(kmax+pad, limits[2][1])
        elif isinstance(pad, list):
            imin = max(imin-pad[0][0], limits[0][0])
            imax = min(imax+pad[0][1], limits[0][1])
            jmin = max(jmin-pad[1][0], limits[1][0])
            jmax = min(jmax+pad[1][1], limits[1][1])
            kmin = max(kmin-pad[2][0], limits[2][0])
            kmax = min(kmax+pad[2][1], limits[2][1])
        elif isinstance(pad, float):
            # In this instance, we compute the extend the limits
            # until the required physical size is met (or until we can 
            # no long extend the index).
            rij = self.scan.pixel_spacing
            rk  = self.scan.slice_spacing

            # Check if the desired bbox size is not smaller than is possible.
            if isinstance(pad, float):
                minsize = max(self.bbox_dims(pad=None))
                if pad < minsize:
                    raise ValueError(("Requested `bbox` size (%.4f mm) is "
                                      "less than minimal possible size "
                                      "(%.4f mm).") % (pad, minsize))
            while (imax-imin)*rij < pad:
                imin -= 1 if imin > limits[0][0] else 0
                imax += 1 if imax < limits[0][1] else 0
                if imin == limits[0][0] and imax == limits[0][1]:
                    break
            while (jmax-jmin)*rij < pad:
                jmin -= 1 if jmin > limits[1][0] else 0
                jmax += 1 if jmax < limits[1][1] else 0
                if jmin == limits[1][0] and jmax == limits[1][1]:
                    break
            while (kmax-kmin)*rk  < pad:
                kmin -= 1 if kmin > limits[2][0] else 0
                kmax += 1 if kmax < limits[2][1] else 0
                if kmin == limits[2][0] and kmax == limits[2][1]:
                    break

        return (slice(imin,imax+1),
                slice(jmin,jmax+1),
                slice(kmin,kmax+1))


    def bbox_dims(self, pad=None):
        """
        Return the physical dimensions of the nodule bounding box in 
        millimeters along each coordinate axis.

        Parameters
        ----------
        pad: int, list, or float, default=None
            See :meth:`pylidc.Annotation.bbox` for a 
            description of this argument.

        Return
        ------
        dims: ndarray, shape=(3,)
            `dims[i]` is the length in millimeters of the bounding box along
            the coordinate axis `i`.

        Example
        -------
        An example where we compare the bounding box volume vs the nodule
        volume::

            import pylidc as pl

            ann = pl.query(pl.Annotation).first()

            print("%.2f mm^3, %.2f mm^3" % (ann.volume,
                                            np.prod(ann.bbox_dims())))
            # => 2439.30 mm^3, 5437.58 mm^3
        """
        res = [self.scan.pixel_spacing,]*2 + [self.scan.slice_spacing]
        return np.array([(b.stop-1-b.start)*r 
                            for r,b in zip(res, self.bbox(pad=pad))])


    def bbox_matrix(self, pad=None):
        """
        The `bbox` function returns a tuple of slices to be used to index
        into an image volume. On the other hand, `bbox_array` returns
        a 3x2 matrix where each row is the (start, stop) indices of the
        i, j, and k axes.

        Parameters
        ----------
        pad: int, list, or float
            See :meth:`pylidc.Annotation.bbox` for a 
            description of this argument.

        Note
        ----
        The indices return by `bbox_array` are *inclusive*, whereas
        the indices of the slice objects in the tuple return by `bbox`
        are offset by +1 in the "stop" index.

        Return
        ------
        bb_mat: ndarray, shape=(3,2)
            `bb_mat[i]` is the stop and start indices (inclusive) of the 
            bounding box along coordinate axis `i`.

        Example
        -------
        An example of the difference between `bbox` and `bbox_matrix`::

            import pylidc as pl
            ann = pl.query(pl.Annotation).first()
            
            bb = ann.bbox()
            bm = ann.bbox_matrix()
            
            print(all([bm[i,0] == bb[i].start for i in range(3)]))
            # => True
            
            print(all([bm[i,1]+1 == bb[i].stop for i in range(3)]))
            # => True
        """
        return np.array([[sl.start, sl.stop-1] for sl in self.bbox(pad=pad)])


    @property
    def centroid(self):
        """
        The center of mass of the nodule as determined by its 
        radiologist-drawn contours.

        Example
        -------
        An example of plotting the centroid on a CT image slice::

            import pylidc as pl
            import matplotlib.pyplot as plt
            
            ann = pl.query(pl.Annotation).first()
            i,j,k = ann.centroid

            vol = ann.scan.to_volume()
            
            plt.imshow(vol[:,:,int(k)], cmap=plt.cm.gray)
            plt.plot(j, i, '.r', label="Nodule centroid")
            plt.legend()
            plt.show()

        Return
        ------
        centr: ndarray, shape=(3,)
            `centr[i]` is the average index value of all contour index values
            for coordinate axis `i`.
        """
        return self.contours_matrix.mean(axis=0)

    @property
    def diameter(self):
        """
        Estimate the greatest axial plane diameter using the annotation's 
        contours. This estimation does not currently account for cases 
        where the diamter passes outside the boundary of the nodule, or 
        through cavities within the nodule.
        
        Return
        ------
        diam: float
            The maximal diameter as float, accounting for the axial-plane 
            resolution of the scan. The units are mm.
        """
        greatest_diameter = -np.inf
        i,j,k = 0,0,1 # placeholders for max indices
        for c,contour in enumerate(self.contours):
            contour_array = contour.to_matrix()[:,:2]*self.scan.pixel_spacing

            # There's some edge cases where the contour consists only of 
            # a single point, which we must ignore.
            if contour_array.shape[0]==1: continue
            
            # pdist computes the pairwise distances between the points.
            # squareform turns the condensed array into matrix where
            # entry i,j is ||point_i - point_j||.
            diameters = squareform(pdist(contour_array))
            diameter  = diameters.max()

            if diameter > greatest_diameter:
                greatest_diameter = diameter
                i = c
                j,k = np.unravel_index(diameters.argmax(), diameters.shape)

        return greatest_diameter

    @property
    def surface_area(self):
        """
        Estimate the surface area by summing the areas of a trianglation
        of the nodules surface in 3d. Returned units are mm^2.

        Return
        ------
        sa: float
            The estimated surface area in squared millimeters.
        """
        mask = self.boolean_mask()
        mask = np.pad(mask, [(1,1), (1,1), (1,1)], 'constant') # Cap the ends.
        mask = mask.astype(np.float)

        rij  = self.scan.pixel_spacing
        rk   = self.scan.slice_thickness
        verts, faces, _, _ = marching_cubes(mask, 0.5, spacing=(rij, rij, rk))
        return mesh_surface_area(verts, faces)

    @property
    def volume(self):
        """
        Estimate the volume of the annotated nodule, using the contour 
        annotations. Green's theorem (via the shoelace formula) is first 
        used to measure the area in each slice. This area is multiplied 
        by the distance between slices to obtain a volume for each slice, 
        which is then added or subtracted from the total volume, depending 
        on if the inclusion value for the contour. 
        
        The distance between slices is taken to be the distance from the 
        midpoint between the current `image_z_position` and the 
        `image_z_position` in one slice higher plus the midpoint between 
        the current `image_z_position` and the `image_z_position` of one 
        slice below. If the the `image_z_position` corresponds to an end 
        piece, we use the distance between the current `image_z_posiition` 
        and the `image_z_position` of one slice below or above for top or 
        bottom, respectively. If the annotation only has one contour, we 
        use the `slice_thickness` attribute of the scan.

        Return
        ------
        vol: float
            The estimated 3D volume of the annotated nodule. Units are cubic
            millimeters.
        """
        volume = 0.
        zvals  = np.unique([c.image_z_position for c in self.contours])

        # We pad a zval on the bottom that is the same distance from the
        # first zval to the second zval but below the first point. We do 
        # the same thing for the top zval.
        if len(self.contours) != 1:
            zlow  = zvals[ 0] - (zvals[1]-zvals[0])
            zhigh = zvals[-1] + (zvals[-1]-zvals[-2])
            zvals = np.r_[zlow, zvals, zhigh]
        else:
            zvals = None

        for i,contour in enumerate(self.contours):
            contour_array = contour.to_matrix() * self.scan.pixel_spacing
            x = contour_array[:,0]
            y = contour_array[:,1]
            # "Shoelace" formula for area.
            area = 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
            
            if zvals is not None:
                j = np.argmin(np.abs(contour.image_z_position-zvals))
                spacing_z = 0.5*(zvals[j+1]-zvals[j-1])
            else:
                spacing_z = self.scan.slice_thickness

            volume += (1. if contour.inclusion else -1.) * area * spacing_z
        return volume

    def visualize_in_3d(self, edgecolor='0.2', cmap='viridis',
                        step=1, figsize=(5,5), backend='matplotlib'):
        """
        Visualize in 3d a triangulation of the nodule's surface.

        Parameters
        ----------
        edgecolor: string color or rgb 3-tuple
            Sets edgecolors of triangulation.
            Ignored if backend != matplotlib.

        cmap: matplotlib colormap string.
            Sets the facecolors of the triangulation.
            See `matplotlib.cm.cmap_d.keys()` for all available.
            Ignored if backend != matplotlib.

        step: int, default=1
            The `step_size` parameter for the skimage marching_cubes function.
            Bigger values are quicker, but yield coarser surfaces.

        figsize: tuple, default=(5,5)
            Figure size for the displayed volume.

        backend: string
            The backend for visualization. Default is matplotlib.
            Execute `from pylidc.Annotation import viz3dbackends` to
            see available backends.

        Example
        -------
        A short example::

            ann = pl.query(pl.Annotation).first()
            ann.visualize_in_3d(edgecolor='green', cmap='autumn')
        """
        if backend not in viz3dbackends:
            raise ValueError("backend should be in %s." % viz3dbackends)

        if backend == 'matplotlib':
            if cmap not in plt.cm.cmap_d.keys():
                raise ValueError("Invalid `cmap`. See `plt.cm.cmap_d.keys()`.")

        # Pad to cap the ends for masks that hit the edge.
        mask = self.boolean_mask(pad=[(1,1), (1,1), (1,1)]) 

        rij  = self.scan.pixel_spacing
        rk   = self.scan.slice_thickness

        if backend == 'matplotlib':
            verts, faces, _, _= marching_cubes(mask.astype(np.float), 0.5,
                                               spacing=(rij, rij, rk),
                                               step_size=step)
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection='3d')

            t = np.linspace(0, 1, faces.shape[0])
            mesh = Poly3DCollection(verts[faces], 
                                    edgecolor=edgecolor,
                                    facecolors=plt.cm.cmap_d[cmap](t))
            ax.add_collection3d(mesh)

            ceil = max(self.bbox_dims(pad=[(1,1), (1,1), (1,1)]))
            ceil = int(np.round(ceil))
            
            ax.set_xlim(0, ceil)
            ax.set_xlabel('length (mm)')

            ax.set_ylim(0, ceil)
            ax.set_ylabel('length (mm)')

            ax.set_zlim(0, ceil)
            ax.set_zlabel('length (mm)')

            plt.tight_layout()
            plt.show()
        elif backend == 'mayavi':
            try:
                from mayavi import mlab
                sf = mlab.pipeline.scalar_field(mask.astype(np.float))
                sf.spacing = [rij, rij, rk]
                mlab.pipeline.iso_surface(sf, contours=[0.5])
                mlab.show()
            except ImportError:
                print("Mayavi could not be imported. Is it installed?")


    def visualize_in_scan(self, verbose=True):
        """
        Engage an interactive visualization of the slices of the scan 
        along with scan and annotation information.
        
        The visualization begins (but is not limited to) the first slice 
        where the nodule occurs (according to the annotation). Annotation
        contours are plotted on top of the images 
        for visualization and can be toggled on and off, using an interactive 
        check mark utility.

        Parameters
        ----------
        verbose: bool, default=True
            Turn the image loading statement on/off.
        """
        images = self.scan.load_all_dicom_images(verbose)
        
        # Preload contours and sort them by z pos.
        contours = sorted(self.contours, key=lambda c: c.image_z_position)
        fnames = self.scan.sorted_dicom_file_names.split(',')
        index_of_contour = [fnames.index(c.dicom_file_name) for c in contours]

        fig = plt.figure(figsize=(16,8))

        min_slice = min(index_of_contour)
        max_slice = max(index_of_contour)
        current_slice = min_slice

        ax_image = fig.add_axes([0.5,0.0,0.5,1.0])
        img = ax_image.imshow(images[current_slice].pixel_array,
                              cmap=plt.cm.gray)

        contour_lines = []
        # We draw all the contours initially and set the visibility
        # to False. This works better than trying create and destroy
        # plots every time we update the image.
        for i,c in enumerate(contours):
            arr = c.to_matrix()
            cc, = ax_image.plot(arr[:,1], arr[:,0], '-r')
            cc.set_visible(i==0) # Set the first contour visible.
            contour_lines.append( cc )
        ax_image.set_xlim(-0.5,511.5); ax_image.set_ylim(511.5,-0.5)
        ax_image.axis('off')
        
        # Add the scan info table
        ax_scan_info = fig.add_axes([0.1, 0.76, 0.3, 0.15])
        ax_scan_info.set_facecolor('w')
        scan_info_table = ax_scan_info.table(
            cellText=[
                ['Patient ID:', self.scan.patient_id],
                ['Slice thickness:', '%.3f mm' % self.scan.slice_thickness],
                ['Pixel spacing:', '%.3f mm'%self.scan.pixel_spacing],
                ['Manufacturer:', images[current_slice].Manufacturer],
                ['Model name:', images[current_slice].ManufacturerModelName],
                ['Convolution kernel:', images[current_slice].ConvolutionKernel],
            ],
            loc='center', cellLoc='left'
        )
        # Remove the cell borders.
        # It Seems like there should be an easier way to do this...
        for cell in scan_info_table.properties()['children']:
            cell.set_color('w')

        ax_scan_info.set_title('Scan Info')
        ax_scan_info.set_xticks([])
        ax_scan_info.set_yticks([])

        # Add annotations / features table.
        ax_annotation_info = fig.add_axes([0.1, 0.45, 0.3, 0.25])
        ax_annotation_info.set_facecolor('w')

        # Create the rows to be displayed in the annotations table.
        cell_text = []
        for f in feature_names:
            row = []
            fname = f.capitalize()
            if fname.startswith('Int'):
                fname = 'InternalStructure'

            row.append(fname)
            row.append(getattr(self,fname))
            row.append(getattr(self,f))

            cell_text.append(row)

        annotation_info_table = ax_annotation_info.table(
            cellText=cell_text,
            loc='center', cellLoc='left', colWidths=[0.45,0.45,0.1]
        )

        # Again, remove cell borders.
        for cell in annotation_info_table.properties()['children']:
            cell.set_color('w')

        ax_annotation_info.set_title('Annotation Info')
        ax_annotation_info.set_xticks([])
        ax_annotation_info.set_yticks([])

        # Add the checkbox for turning contours on / off.
        ax_contour_checkbox = fig.add_axes([0.1, 0.25, 0.1, 0.15])
        ax_contour_checkbox.set_facecolor('w')
        contour_checkbox = CheckButtons(ax_contour_checkbox,
                                        ('Show Contours',), (True,))
        contour_checkbox.is_checked = True

        # Add the widgets.
        ax_slice = fig.add_axes([0.1, 0.1, 0.3, 0.05])
        ax_slice.set_facecolor('w')
        txt = 'Z: %.3f'%float(images[current_slice].ImagePositionPatient[-1]) 
        sslice = Slider(ax_slice,
                        txt,
                        0,
                        len(images)-1,
                        valinit=current_slice,
                        valfmt=u'Slice: %d')

        def update(_):
            # Update image itself.
            current_slice = int(sslice.val)
            img.set_data(images[current_slice].pixel_array)
            txt = 'Z: %.3f'
            txt = txt % float(images[current_slice].ImagePositionPatient[-1])
            sslice.label.set_text(txt)

            if contour_checkbox.is_checked:
                for i,c in enumerate(contour_lines):
                    flag = ((index_of_contour[i] == current_slice) and 
                            (current_slice >= min_slice) and
                            (current_slice <= max_slice))
                    # Set contour visible if flag is True.
                    c.set_visible(flag)
            else:
                for c in contour_lines: c.set_visible(False)

            fig.canvas.draw_idle()

        def update_contours(_):
            contour_checkbox.is_checked = not contour_checkbox.is_checked
            update(None) # update requires an argument.

        sslice.on_changed(update)
        contour_checkbox.on_clicked(update_contours)

        plt.show()

    @property
    def contour_slice_zvals(self):
        """An array of unique z-coordinates for the contours."""
        return np.sort([c.image_z_position for c in self.contours])        

    @property
    def contour_slice_indices(self):
        """
        Returns an array of indices into the scan where each contour
        belongs. An example should clarify::

            import pylidc as pl
            
            ann = pl.query(pl.Annotation)
            
            zvals = ann.contour_slice_zvals
            kvals = ann.contour_slice_indices
            scan_zvals = ann.scan.slice_zvals
            
            for k,z in zip(kvals, zvals):
                # the two z values should the same (up to machine precision)
                print(k, z, scan_zvals[k]) 
        """
        return np.sort([c.image_k_position for c in self.contours])

    @property
    def contours_matrix(self):
        """
        All the contour index values a 3D numpy array.
        """
        return np.vstack([c.to_matrix(include_k=True)
                                for c in sorted(self.contours,
                                        key=lambda c: c.image_z_position)])

    def boolean_mask(self, pad=None, bbox=None, include_contour_points=False):
        """
        A boolean volume where 1 indicates nodule and 0 indicates
        non-nodule. The `mask` volume covers the extent of the voxels
        in the image volume given by `annotation.bbox`, i.e., the `mask`
        volume would be placed in the full image volume according to
        the `bbox` attribute.

        Parameters
        ----------
        pad: int, list, or float, default=None
            See :meth:`pylidc.Annotation.bbox` for a 
            description of this argument.

        bbox: 3x2 NumPy array, default=None
            If `bbox` is provided, then `pad` is ignored. This argument allows
            for more fine-tuned control of placement of the mask in a volume,
            or for pre-computation of bbox when working with multiple 
            Annotation object.

        Example
        -------
        An example::

            import pylidc as pl
            import matplotlib.pyplot as plt
            
            ann = pl.query(pl.Annotation).first()
            vol = ann.scan.to_volume()
            
            mask = ann.boolean_mask()
            bbox = ann.bbox()
            
            print("Avg HU inside nodule: %.1f" % vol[bbox][mask].mean())
            # => Avg HU inside nodule: -280.0

            print("Avg HU outside nodule: %.1f" % vol[bbox][~mask].mean())
            # => Avg HU outside nodule: -732.2
        """
        bb = self.bbox_matrix(pad=pad) if bbox is None else bbox

        czs = self.contour_slice_zvals
        cks = self.contour_slice_indices

        zs = self.scan.slice_zvals
        zs = zs[cks[0]:cks[-1]+1]

        # Lambda to map a z-value to its appropriate index in the volume.
        z_to_index = lambda z: dict(zip(czs,cks))[z] - bb[2,0]#cks[0]

        # Get dimensions, initialize mask.
        ni,nj,nk = np.diff(bb, axis=1).astype(int)[:,0] + 1
        mask = np.zeros((ni,nj,nk), dtype=np.bool)

        # We check if these points are enclosed within each contour 
        # for a given slice. `test_points` is a list of image coordinate 
        # points, offset by the bounding box.
        ii,jj = np.indices(mask.shape[:2])
        test_points = bb[:2,0] + np.c_[ii.flatten(), jj.flatten()]

        # First we "turn on" pixels enclosed by inclusion contours.
        for contour in self.contours:
            if contour.inclusion:
                zi = z_to_index(contour.image_z_position)
                C  = contour.to_matrix(include_k=False)

                # Turn the contour closed if it is not.
                if (C[0] != C[-1]).any():
                    C = np.append(C, C[0].reshape(1,2), axis=0)

                # Create path object and test all pixels
                # within the contour's bounding box.
                path = mplpath.Path(C, closed=True)
                contains_pts = path.contains_points(test_points)
                contains_pts = contains_pts.reshape(mask.shape[:2])

                # The logical or here prevents the cases where a single
                # slice contains multiple inclusion regions.
                mask[:,:,zi] = np.logical_or(mask[:,:,zi], contains_pts)

                if not include_contour_points:
                    # Remove the contour points themselves.
                    i, j = (C - bb[:2,0]).T
                    k = np.ones(C.shape[0], dtype=np.int)*zi
                    mask[i,j,k] = False

        # Second, we "turn off" pixels enclosed by exclusion contours.
        for contour in self.contours:
            if not contour.inclusion:
                zi = z_to_index(contour.image_z_position)
                C = contour.to_matrix(include_k=False)

                # Turn the contour closed if it is not.
                if (C[0] != C[-1]).any():
                    C = np.append(C, C[0].reshape(1,2), axis=0)

                path = mplpath.Path(C, closed=True)
                not_contains_pts = ~path.contains_points(test_points)
                not_contains_pts = not_contains_pts.reshape(mask.shape[:2])
                mask[:,:,zi] = np.logical_and(mask[:,:,zi], not_contains_pts)

                # Remove the contour points themselves.
                i, j = (C - bb[:2,0]).T
                k = np.ones(C.shape[0], dtype=np.int)*zi
                mask[i,j,k] = False

        return mask

    def _as_set(self):
        """
        Private function used to computed overlap between nodules of the 
        same scan. This function returns a set where is element is a 
        3-tuple referring to a voxel within the scan. If the voxel is 
        in the set, the nodule is considered to be defined there.
        
        Essentially this is a boolean mask stored as a set.
        """
        included = set()
        excluded = set()
        # Add all points lying within each inclusion contour to S.
        for contour in self.contours:
            contour_matrix = contour.to_matrix()[:,:2]
            # Turn the contour closed if it's not.
            if (contour_matrix[0] != contour_matrix[-1]).all():
                contour_matrix = np.append(contour_matrix,
                                           contour_matrix[0].reshape(1,2),
                                           axis=0)

            # Create path object and test all pixels 
            # within the contour's bounding box.
            path = mplpath.Path(contour_matrix, closed=True)
            mn = contour_matrix.min(axis=0)
            mx = contour_matrix.max(axis=0)
            x,y = np.mgrid[mn[0]:mx[0]+1, mn[1]:mx[1]+1]
            test_points = np.c_[x.flatten(), y.flatten()]
            points_in_contour = test_points[path.contains_points(test_points)]

            # Add the z coordinate.
            points_in_contour = np.c_[\
                points_in_contour,\
                np.ones(points_in_contour.shape[0])*contour.image_z_position
            ]

            # Now turn the numpy matrix into a list of tuples,
            # so we can add it to the corresponding set.
            points_in_contour = list(map(tuple, points_in_contour))

            # Update the corresponding set.
            if contour.inclusion:
                included.update(points_in_contour)
            else:
                excluded.update(points_in_contour)
        # Return the included points minus the excluded points.
        return included.difference( excluded )

    def uniform_cubic_resample(self, side_length=None, resample_vol=True,
                               irp_pts=None, return_irp_pts=False,
                               resample_img=True, verbose=True):
        """
        Get the CT value volume and respective boolean mask volume. The 
        volumes are interpolated and resampled to have uniform spacing of 1mm
        along each dimension. The resulting volumes are cubic of the 
        specified `side_length`. Thus, the returned volumes have dimensions,
        `(side_length+1,)*3` (since `side_length` is the spacing).

        TODO
        ----
        It would be nice if this function performed fully general 
        interpolation, i.e., not necessarily uniform spacing and allowing 
        different resample-resolutions along different coordinate axes.

        Parameters
        ----------
        side_length: integer, default=None
            The physical length of each side of the new cubic 
            volume in millimeters. The default, `None`, takes the
            max of the nodule's bounding box dimensions.

            If this parameter is not `None`, then it should be 
            greater than any bounding box dimension. If the specified 
            `side_length` requires a padding which results in an 
            out-of-bounds image index, then the image is padded with 
            the minimum CT image value.

        resample_vol: boolean, default=True
            If False, only the segmentation volume is resampled.

        irp_pts: 3-tuple from meshgrid
            If provided, the volume(s) will be resampled over these interpolation
            points, rather than the automatically calculated points. This allows
            for sampling segmentation volumes over a common coordinate-system.

        return_irp_pts: boolean, default=False
            If True, the interpolation points (ix,iy,iz) at which the volume(s)
            were resampled are returned. These can potentially be provided as
            an argument to `irp_pts` for separate selfotations that refer to the
            same nodule, allowing the segmentation volumes to be resampled in a
            common coordinate-system.

        verbose: boolean, default=True
            Turn the loading statement on / off.

        Return
        ------
        [ct_volume,] mask [, irp_pts]: ndarray, ndarray, list of ndarrays
            `ct_volume` and `mask` are the resampled CT and boolean 
            volumes, respectively. `ct_volume` and `irp_points` are optionally
            returned, depending on which flags are set (see above).

        Example
        -------
        An example::

            import numpy as np
            import matplotlib.pyplot as plt
            import pylidc as pl

            ann = pl.query(pl.Annotation).first()

            # resampled volumes will have uniform side length of 70mm and
            # uniform voxel spacing of 1mm.
            n = 70
            vol,mask = ann.uniform_cubic_resample(n)


            # Setup the plot.
            img = plt.imshow(np.zeros((n+1, n+1)), 
                             vmin=vol.min(), vmax=vol.max(),
                             cmap=plt.cm.gray)


            # View all the resampled image volume slices.
            for i in range(n+1):
                img.set_data(vol[:,:,i] * (mask[:,:,i]*0.6+0.2))

                plt.title("%02d / %02d" % (i+1, n))
                plt.pause(0.1)

        """
        bbox  = self.bbox_matrix()
        bboxd = self.bbox_dims()
        rij   = self.scan.pixel_spacing
        rk    = self.scan.slice_spacing

        imin,imax = bbox[0]
        jmin,jmax = bbox[1]
        kmin,kmax = bbox[2]

        xmin,xmax = imin*rij, imax*rij
        ymin,ymax = jmin*rij, jmax*rij

        zmin = self.scan.slice_zvals[kmin]
        zmax = self.scan.slice_zvals[kmax]

        # { Begin input checks.
        if side_length is None:
            side_length = np.ceil(bboxd.max())
        else:
            if not isinstance(side_length, int):
                raise TypeError('`side_length` must be an integer.')
            if side_length < bboxd.max():
                raise ValueError('`side_length` must be greater\
                                   than any bounding box dimension.')
        side_length = float(side_length)
        # } End input checks.

        # Load the images. Get the z positions.
        images = self.scan.load_all_dicom_images(verbose=verbose)
        img_zs = [float(img.ImagePositionPatient[-1]) for img in images]
        img_zs = np.unique(img_zs)

        # Get the z values of the contours.
        contour_zs = np.unique([c.image_z_position for c in self.contours])

        # Get the indices where the nodule stops and starts
        # with respect to the scan z values.
        #kmin = np.where(zmin == img_zs)[0][0]
        #kmax = np.where(zmax == img_zs)[0][0]

        # Initialize the boolean mask.
        mask = self.boolean_mask()

        ########################################################
        # { Begin interpolation grid creation.
        #   (The points at which the volumes will be resampled.)

        # Compute new interpolation grid points in x.
        d = 0.5*(side_length-(xmax - xmin))
        xhat, step = np.linspace(xmin-d, xmax+d,
                                 int(side_length)+1, retstep=True)
        assert abs(step-1) < 1e-5, "New x spacing != 1."

        # Do the same for y.
        d = 0.5*(side_length-(ymax - ymin))
        yhat, step = np.linspace(ymin-d, ymax+d,
                                 int(side_length)+1, retstep=True)
        assert abs(step-1) < 1e-5, "New y spacing != 1."

        # Do the same for z.
        d = 0.5*(side_length-(zmax - zmin))
        zhat, step = np.linspace(zmin-d, zmax+d,
                                 int(side_length)+1, retstep=True)
        assert abs(step-1) < 1e-5, "New z pixel spacing != 1."

        # } End interpolation grid creation.
        ########################################################

        ########################################################
        # { Begin grid creation.
        #   (The points at which the volumes are assumed to be sampled.)

        # a[x|y|z], b[x|y|z] are the start / stop indexes for the 
        # (non resample) sample grids along each respective axis.

        # It helps to draw a diagram. For example,
        #
        # *--*--*-- ...
        # x3 x4 x5
        #  *---*---*--- ...
        #  xhat0
        #
        # In this case, `ax` would be chosen to be 3
        # since this is the index directly to the left of 
        # `xhat[0]`. If `xhat[0]` is below any grid point,
        # then `ax` is the minimum possible index, 0. A similar
        # diagram helps with the `bx` index.

        T = np.arange(0, 512)*rij

        if xhat[0] <= 0:
            ax = 0
        else:
            ax = (T < xhat[0]).sum() - 1
        if xhat[-1] >= T[-1]:
            bx = 512
        else:
            bx = 512 - (T > xhat[-1]).sum() + 1

        if yhat[0] <= 0:
            ay = 0
        else:
            ay = (T < yhat[0]).sum() - 1
        if yhat[-1] >= T[-1]:
            by = 512
        else:
            by = 512 - (T > yhat[-1]).sum() + 1

        if zhat[0] <= img_zs[0]:
            az = 0
        else:
            az = (img_zs < zhat[0]).sum() - 1
        if zhat[-1] >= img_zs[-1]:
            bz = len(img_zs)
        else:
            bz = len(img_zs) - (img_zs > zhat[-1]).sum() + 1
        
        # These are the actual grids.
        x = T[ax:bx]
        y = T[ay:by]
        z = img_zs[az:bz]

        # } End grid creation.
        ########################################################


        # Create the non-interpolated CT volume.
        if resample_vol:
            ctvol = np.zeros(x.shape+y.shape+z.shape, dtype=np.float64)
            for k in range(z.shape[0]):
                ctvol[:,:,k] = images[k+az].pixel_array[ax:bx, ay:by]

        # We currently only have the boolean mask volume on the domain
        # of the bounding box. Thus, we must "place it" in the appropriately
        # sized volume (i.e., `ctvol.shape`). This is accomplished by
        # padding `mask`.
        padvals = [(imin-ax, bx-1-imax), # The `b` terms have a `+1` offset
                   (jmin-ay, by-1-jmax), # from being an index that is
                   (kmin-az, bz-1-kmax)] # corrected with the `-1` here.
        mask = np.pad(mask, pad_width=padvals,
                      mode='constant', constant_values=False)

        # Obtain minimum image value to use as const for interpolation.
        if resample_vol:
            fillval = min([img.pixel_array.min() for img in images])

        if irp_pts is None:
            ix,iy,iz = np.meshgrid(xhat, yhat, zhat, indexing='ij')
        else:
            ix,iy,iz = irp_pts
        IXYZ = np.c_[ix.flatten(), iy.flatten(), iz.flatten()]

        # Interpolate the nodule CT volume.
        if resample_vol:
            rgi = RegularGridInterpolator(points=(x, y, z), values=ctvol,
                                          bounds_error=False, fill_value=fillval)
            ictvol = rgi(IXYZ).reshape(ix.shape)

        # Interpolate the mask volume.
        rgi = RegularGridInterpolator(points=(x, y, z), values=mask,
                                      bounds_error=False, fill_value=False)
        imask = rgi(IXYZ).reshape(ix.shape) > 0

        if resample_vol:
            if return_irp_pts:
                return ictvol, imask, (ix,iy,iz)
            else:
                return ictvol, imask
        else:
            if return_irp_pts:
                return imask, (ix,iy,iz)
            else:
                return imask


# Add the relationship to the Scan model.
Scan.annotations = relationship('Annotation',
                                order_by=Annotation.id,
                                back_populates='scan')
