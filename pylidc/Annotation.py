import os
import sqlalchemy as sq
from sqlalchemy.orm import relationship
from ._Base import Base
from .Scan import Scan

import dicom
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
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import Delaunay
from scipy.ndimage.morphology import distance_transform_edt as dtrans
from skimage.measure import marching_cubes, mesh_surface_area

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

class Annotation(Base):
    """
    The Nodule model class holds the information from a single physicians 
    annotation of a nodule >= 3mm class with a particular scan. A nodule 
    has many contours, each of which refers to the contour drawn for 
    nodule in each scan slice.  

    Example:
        >>> import pylidc as pl
        >>> # Get the first annotation with spiculation value greater than 3.
        >>> ann = pl.query(pl.Annotation).filter(pl.Annotation.spiculation > 3).first()
        >>> print(ann.spiculation)
        >>> # => 4
        >>> # Each nodule feature has a helper function to print the semantic value.
        >>> print(ann.Spiculation())
        >>> # => Medium-High Spiculation
        >>> 
        >>> q = pl.query(pl.Annotation).join(pl.Scan).filter(pl.Scan.resolution_z <= 1, pl.Annotation.malignancy == 5)
        >>> print(q.count())
        >>> # => 58
        >>> ann = q.first()
        >>> print(ann.estimate_diameter(), ann.estimate_volume())
        >>> # => 17.9753270062 1240.43532257
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
    def Subtlety(self):
        """return subtlety value as string"""
        s = self.subtlety
        assert s in range(1,6), "Subtlety score out of bounds."
        if   s == 1: return 'Extremely Subtle'
        elif s == 2: return 'Moderately Subtle'
        elif s == 3: return 'Fairly Subtle'
        elif s == 4: return 'Moderately Obvious'
        elif s == 5: return 'Obvious'

    def InternalStructure(self):
        """return internalStructure value as string"""
        s = self.internalStructure
        assert s in range(1,5), "Internal structure score out of bounds."
        if   s == 1: return 'Soft Tissue'
        elif s == 2: return 'Fluid'
        elif s == 3: return 'Fat'
        elif s == 4: return 'Air'

    def Calcification(self):
        """return calcification value as string"""
        s = self.calcification
        assert s in range(1,7), "Calcification score out of bounds."
        if   s == 1: return 'Popcorn'
        elif s == 2: return 'Laminated'
        elif s == 3: return 'Solid'
        elif s == 4: return 'Non-central'
        elif s == 5: return 'Central'
        elif s == 6: return 'Absent'

    def Sphericity(self):
        """return sphericity value as string"""
        s = self.sphericity
        assert s in range(1,6), "Sphericity score out of bounds."
        if   s == 1: return 'Linear'
        elif s == 2: return 'Ovoid/Linear'
        elif s == 3: return 'Ovoid'
        elif s == 4: return 'Ovoid/Round'
        elif s == 5: return 'Round'

    def Margin(self):
        """return margin value as string"""
        s = self.margin
        assert s in range(1,6), "Margin score out of bounds."
        if   s == 1: return 'Poorly Defined'
        elif s == 2: return 'Near Poorly Defined'
        elif s == 3: return 'Medium Margin'
        elif s == 4: return 'Near Sharp'
        elif s == 5: return 'Sharp'

    def Lobulation(self):
        """return lobulation value as string"""
        s = self.lobulation
        assert s in range(1,6), "Lobulation score out of bounds."
        if   s == 1: return 'No Lobulation'
        elif s == 2: return 'Nearly No Lobulation'
        elif s == 3: return 'Medium Lobulation'
        elif s == 4: return 'Near Marked Lobulation'
        elif s == 5: return 'Marked Lobulation'

    def Spiculation(self):
        """return spiculation value as string"""
        s = self.spiculation
        assert s in range(1,6), "Spiculation score out of bounds."
        if   s == 1: return 'No Spiculation'
        elif s == 2: return 'Nearly No Spiculation'
        elif s == 3: return 'Medium Spiculation'
        elif s == 4: return 'Near Marked Spiculation'
        elif s == 5: return 'Marked Spiculation'

    def Texture(self):
        """return texture value as string"""
        s = self.texture
        assert s in range(1,6), "Texture score out of bounds."
        if   s == 1: return 'Non-Solid/GGO'
        elif s == 2: return 'Non-Solid/Mixed'
        elif s == 3: return 'Part Solid/Mixed'
        elif s == 4: return 'Solid/Mixed'
        elif s == 5: return 'Solid'

    def Malignancy(self):
        """return malignancy value as string"""
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

        return_str: bool, default False
            If True, a list of strings is also returned, corresponding
            to the meaning of each numerical feature value.
        """
        fvals = np.array([getattr(self,f) for f in feature_names])
        if return_str:
            caps = [f.title() for f in feature_names]
            k = caps.index('Internalstructure')
            caps[k] = 'InternalStructure'
            return fvals, [getattr(self, c)() for c in caps]
        else:
            return fvals

    def print_formatted_feature_table(self):
        """
        Return all feature values as a string table.
        """
        fnames = feature_names
        fvals, fstrings = self.feature_vals(True)

        print('%-18s   %-24s   %-2s'%('Feature', 'Meaning','#'))
        print('%-18s   %-24s   %-2s' % ('-', '-', '-'))

        for i in range(len(fnames)):
            print('%-18s | %-24s | %-2d'%(fnames[i].title(), 
                                          fstrings[i], fvals[i]))


    def bbox(self, image_coords=False):
        """
        Return a 3 by 2 matrix, corresponding to the bounding box of the 
        annotation within the scan. If `scan_slice` is a numpy array 
        containing aslice of the scan, each slice of the annotation is 
        contained within the box:

            bbox[1,0]:bbox[1,1]+1, bbox[0,0]:bbox[0,1]+1

        If `image_coords` is `True` then each annotation slice is 
        instead contained within:
            
            bbox[0,0]:bbox[0,1]+1, bbox[1,0]:bbox[1,1]+1

        The last row of `bbox` give the inclusive lower and upper 
        bounds of the `image_z_position`.
        """
        matrix = self.contours_to_matrix()
        bbox   = np.c_[matrix.min(axis=0), matrix.max(axis=0)]
        return bbox if not image_coords else bbox[[1,0,2]]

    def bbox_dimensions(self, image_coords=False):
        """
        Return the dimensions of the nodule bounding box in mm.
        """
        bb = self.bbox(image_coords)
        df = np.diff(bb)[:,0]
        df[:2] = df[:2]*self.scan.pixel_spacing
        return df

    def centroid(self, image_coords=True):
        """
        Return the center of mass of the nodule as determined by its 
        radiologist-drawn contours.
        """
        return self.contours_to_matrix(image_coords).mean(axis=0)

    def estimate_diameter(self, return_indices=False):
        """
        Estimate the greatest axial plane diameter using the annotation's 
        contours. This estimation does not currently account for cases 
        where the diamter passes outside the boundary of the nodule, or 
        through cavities within the nodule.
        
        TODO?: The greatest diameter perpendicular to the greatest 
        diameter could be computed here as well.

        return_indices: bool, default False
            If `True`, a 3-tuple of indices is return along with the 
            maximum diameter, `(i,j,k)`, where `i` is the index of the 
            contour where the max occurs, and `j` and `k` refer to the 
            two contour points between which is the maximum diameter.

        returns: float (or float,Contour)
            Returns the diameter as float, accounting for the axial-plane 
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

        if not return_indices:
            return greatest_diameter
        else:
            return greatest_diameter, (i,j,k)

    def estimate_surface_area(self):
        """
        Estimate the surface area by summing the areas of a trianglation
        of the nodules surface in 3d. Returned units are mm^2.
        """
        mask = self.get_boolean_mask()
        mask = np.pad(mask, [(1,1), (1,1), (1,1)], 'constant') # Cap the ends.
        dist = dtrans(mask) - dtrans(~mask)

        rxy  = self.scan.pixel_spacing
        rz   = self.scan.slice_thickness
        verts, faces = marching_cubes(dist, 0, spacing=(rxy, rxy, rz))
        return mesh_surface_area(verts, faces)

    def estimate_volume(self):
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

        returns: float
            The estimated 3D volume of the annotated nodule. Units are mm^3.
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

    def visualize_in_3d(self, edgecolor='0.2', cmap='viridis'):
        """
        Visualize in 3d a triangulation of the nodule's surface.

        edgecolor: string color or rgb 3-tuple
            Sets edgecolors of triangulation.

        cmap: matplotlib colormap string.
            Sets the facecolors of the triangulation.
            See `matplotlib.cm.cmap_d.keys()` for all available.

        Example:
            >>> ann = pl.query(pl.Annotation).first()
            >>> ann.visualize_in_3d(edgecolor='green', cmap='autumn')
        """
        if cmap not in plt.cm.cmap_d.keys():
            raise ValueError("Invalid `cmap`. See `plt.cm.cmap_d.keys()`.")

        mask = self.get_boolean_mask()
        mask = np.pad(mask, [(1,1), (1,1), (1,1)], 'constant') # Cap the ends.
        dist = dtrans(mask) - dtrans(~mask)

        rxy  = self.scan.pixel_spacing
        rz   = self.scan.slice_thickness
        verts, faces = marching_cubes(dist, 0, spacing=(rxy, rxy, rz))
        maxes = np.ceil(verts.max(axis=0))

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        t = np.linspace(0, 1, faces.shape[0])
        mesh = Poly3DCollection(verts[faces], 
                                edgecolor=edgecolor,
                                facecolors=plt.cm.cmap_d[cmap](t))
        ax.add_collection3d(mesh)

        ax.set_xlim(0, maxes[0])
        ax.set_xlabel('length (mm)')

        ax.set_ylim(0, maxes[1])
        ax.set_ylabel('length (mm)')

        ax.set_zlim(0, maxes[2])
        ax.set_zlabel('length (mm)')

        plt.tight_layout()
        plt.show()


    def visualize_in_scan(self, verbose=True):
        """
        Interactive visualization of the slices of the scan along with scan 
        and annotation information. The visualization begins 
        (but is not limited to) the first slice where the nodule occurs 
        (according to the annotation). Contours are plotted atop the images 
        for visualization and can be toggled on and off.
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
        # We draw all the contours initally and set the visibility
        # to False. This works better than trying create and destroy
        # plots every time we update the image.
        for i,c in enumerate(contours):
            arr = c.to_matrix()
            cc, = ax_image.plot(arr[:,0], arr[:,1], '-r')
            cc.set_visible(i==0) # Set the first contour visible.
            contour_lines.append( cc )
        ax_image.set_xlim(-0.5,511.5); ax_image.set_ylim(511.5,-0.5)
        ax_image.axis('off')
        
        # Add the scan info table
        ax_scan_info = fig.add_axes([0.1, 0.8, 0.3, 0.1])
        ax_scan_info.set_facecolor('w')
        scan_info_table = ax_scan_info.table(
            cellText=[
                ['Patient ID:', self.scan.patient_id],
                ['Slice thickness:', '%.3f mm' % self.scan.slice_thickness],
                ['Pixel spacing:', '%.3f mm'%self.scan.pixel_spacing]
            ],
            loc='center', cellLoc='left'
        )
        # Remove the cell borders.
        # It Seems like there should be an easier way to do this...
        for cell in scan_info_table.properties()['child_artists']:
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
            row.append(getattr(self,fname)())
            row.append(getattr(self,f))

            cell_text.append(row)

        annotation_info_table = ax_annotation_info.table(
            cellText=cell_text,
            loc='center', cellLoc='left', colWidths=[0.45,0.45,0.1]
        )

        # Again, remove cell borders.
        for cell in annotation_info_table.properties()['child_artists']:
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
            txt='Z: %.3f'%float(images[current_slice].ImagePositionPatient[-1])
            sslice.label.set_text(txt)
            if contour_checkbox.is_checked:
                for i,c in enumerate(contour_lines):
                    flag = (index_of_contour[i] == current_slice)
                    flag = flag and (current_slice >= min_slice)
                    flag = flag and (current_slice <= max_slice)
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

    def contours_to_matrix(self, image_coords=True):
        """
        Return all the contours in a 3D numpy array.

        image_coords: bool, default True
            If True, the first two coordinates of each point are given 
            in image coordinates (i.e., 2d array index). If False, these 
            values are scaled by the respective `pixel_spacing` attribute
            of the Annotation's Scan.
        """
        pts = np.vstack([c.to_matrix() for c in self.contours])
        if not image_coords:
            pts[:,:2] *= self.scan.pixel_spacing
        return pts
            

    def get_boolean_mask(self, return_bbox=False):
        """
        Return a boolean volume which corresponds to the bounding box 
        containing the nodule annotation. The slices of the volume are 
        ordered by increasing `image_z_position` of the contour 
        annotations.
        
        Note that this method doesn't account for a case where the nodule 
        contour annotations "skip a slice".
        
        returns: mask, bounding_box
            `mask` is the boolean volume. In the original 
            512 x 512 x num_slices dicom volume, `mask` is a boolean 
            mask over the region, `bbox[i,0]:bbox[i,1]+1`, i=0,1,2
        """
        bbox = self.bbox()
        zs = np.unique([c.image_z_position for c in self.contours])
        z_to_index = dict(zip(zs,range(len(zs))))

        # Get dimensions, initialize mask.
        nx,ny = np.diff(bbox[:2], axis=1).astype(int) + 1
        nx = int(nx); ny = int(ny)
        nz = int(zs.shape[0])
        mask = np.zeros((nx,ny,nz), dtype=np.bool)

        # We check if these points are enclosed within each contour 
        # for a given slice. `test_points` is a list of image coordinate 
        # points, offset by the bounding box.
        test_points = bbox[:2,0] + np.c_[ np.where(~mask[:,:,0]) ]

        # First we "turn on" pixels enclosed by inclusion contours.
        for contour in self.contours:
            if contour.inclusion:
                zi = z_to_index[contour.image_z_position]
                contour_matrix = contour.to_matrix()[:,:2]

                # Turn the contour closed if it's not.
                if (contour_matrix[0] != contour_matrix[-1]).any():
                    contour_matrix = np.append(contour_matrix,
                                               contour_matrix[0].reshape(1,2),
                                               axis=0)

                # Create path object and test all pixels
                # within the contour's bounding box.
                path = mplpath.Path(contour_matrix, closed=True)
                contains_pts = path.contains_points(test_points)
                contains_pts = contains_pts.reshape(mask.shape[:2])
                # The logical or here prevents the cases where a single
                # slice contains multiple inclusion regions.
                mask[:,:,zi] = np.logical_or(mask[:,:,zi], contains_pts)

        # Second, we "turn off" pixels enclosed by exclusion contours.
        for contour in self.contours:
            if not contour.inclusion:
                zi = z_to_index[contour.image_z_position]
                contour_matrix = contour.to_matrix()[:,:2]

                # Turn the contour closed if it's not.
                if (contour_matrix[0] != contour_matrix[-1]).any():
                    contour_matrix = np.append(contour_matrix,
                                               contour_matrix[0].reshape(1,2),
                                               axis=0)

                path = mplpath.Path(contour_matrix, closed=True)
                not_contains_pts = ~path.contains_points(test_points)
                not_contains_pts = not_contains_pts.reshape(mask.shape[:2])
                mask[:,:,zi] = np.logical_and(mask[:,:,zi], not_contains_pts)

        # The first and second axes have to 
        # be swapped because of the reshape.
        if return_bbox:
            return mask.swapaxes(0,1), bbox[[1,0,2]]
        else:
            return mask.swapaxes(0,1)

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

    def uniform_cubic_resample(self, side_length=None, verbose=True):
        """
        Get the CT value volume and respective boolean mask volume. The 
        volumes are interpolated and resampled to have uniform spacing of 1mm
        along each dimension. The resulting volumes are cubic of the 
        specified `side_length`. Thus, the returned volumes have dimensions,
        `(side_length+1,)*3` (since `side_length` is the spacing).


        side_length: integer, default None
            The physical length of each side of the new cubic 
            volume in millimeters. The default, `None`, takes the
            max of the nodule's bounding box dimensions.

            If this parameter is not `None`, then it should be 
            greater than any bounding box dimension. If the specified 
            `side_length` requires a padding which results in an 
            out-of-bounds image index, then the image is padded with 
            the minimum CT image value.

        verbose: boolean, default True
            Turn the loading statement on / off.

        returns: ct_volume, mask
            `ct_volume` and `mask` are the resampled CT and boolean 
            volumes, respectively.

        Example:
            >>> self = pl.query(pl.Annotation).first()
            >>> ct_volume, mask = ann.uniform_cubic_resample(side_length=70)
            >>> print(ct_volume.shape, mask.shape)
            >>> # => (71, 71, 71), (71, 71, 71)
            >>> # (Nodule is centered at (35,35,35).)
            >>>
            >>> import matplotlib.pyplot as plt
            >>> plt.imshow( ct_volume[:,:,35] * (0.2 + 0.8*mask[:,:,35]) )
            >>> plt.show()
        """
        bbox  = self.bbox(image_coords=True)
        bboxd = self.bbox_dimensions(image_coords=True)
        rxy   = self.scan.pixel_spacing

        imin,imax = bbox[0].astype(int)
        jmin,jmax = bbox[1].astype(int)

        xmin,xmax = imin*rxy, imax*rxy
        ymin,ymax = jmin*rxy, jmax*rxy
        zmin,zmax = bbox[2]

        # { Begin input checks.
        if side_length is None:
            side_length = np.ceil(bboxd.max())
        else:
            if not isinstance(side_length, int):
                raise ValueError('`side_length` must be an integer.')
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
        kmin = np.where(zmin == img_zs)[0][0]
        kmax = np.where(zmax == img_zs)[0][0]

        # Initialize the boolean mask.
        mask = self.get_boolean_mask()

        ########################################################
        # { Begin mask corrections.

        # This block handles the case where 
        # the contour selfotations "skip a slice".
        if mask.shape[2] != (kmax-kmin+1):
            old_mask = mask.copy()
            
            # Create the new mask with appropriate z-length.
            mask = np.zeros((old_mask.shape[0],
                             old_mask.shape[1],
                             kmax-kmin+1), dtype=np.bool)

            # Map z's to an integer.
            z_to_index = dict(zip(
                            img_zs[kmin:kmax+1],
                            range(img_zs[kmin:kmax+1].shape[0])
                         ))

            # Map each slice to its correct location.
            for k in range(old_mask.shape[2]):
                mask[:, :, z_to_index[contour_zs[k]] ] = old_mask[:,:,k]

            # Get rid of the old one.
            del old_mask

        # } End mask corrections.
        ########################################################

        ########################################################
        # { Begin interpolation grid creation.
        #   (The points at which the volumes will be resampled.)

        # Compute new interpolation grid points in x.
        d = 0.5*(side_length-(xmax - xmin))
        xhat, step = np.linspace(xmin-d, xmax+d, side_length+1, retstep=True)
        assert abs(step-1) < 1e-5, "New x spacing != 1."

        # Do the same for y.
        d = 0.5*(side_length-(ymax - ymin))
        yhat, step = np.linspace(ymin-d, ymax+d, side_length+1, retstep=True)
        assert abs(step-1) < 1e-5, "New y spacing != 1."

        # Do the same for y.
        d = 0.5*(side_length-(zmax - zmin))
        zhat, step = np.linspace(zmin-d, zmax+d, side_length+1, retstep=True)
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

        T = np.arange(0, 512)*rxy

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
        ctvol = np.zeros(x.shape+y.shape+z.shape, dtype=np.float64)
        for k in range(z.shape[0]):
            ctvol[:,:,k] = images[k+az].pixel_array[ax:bx, ay:by]

        # We currently only have the boolean mask volume on the domain
        # of the bounding box. Thus, we must "place it" in the appropriately
        # sized volume (i.e., `ctvol.shape`). This is accomplished by
        # padding `mask`.
        padvals = [(imin-ax, bx-1-imax), # The `b` terms have a `+1` offset
                   (jmin-ay, by-1-jmax), # from being an index that is
                   (kmin-az, bz-1-kmax)] # correct with the `-1` here.
        mask = np.pad(mask, pad_width=padvals,
                      mode='constant', constant_values=False)

        # Obtain minimum image value to use as const for interpolation.
        fillval = min([img.pixel_array.min() for img in images])

        ix,iy,iz = np.meshgrid(xhat, yhat, zhat, indexing='ij')
        IXYZ = np.c_[ix.flatten(), iy.flatten(), iz.flatten()]

        # Interpolate the nodule CT volume.
        rgi = RegularGridInterpolator(points=(x, y, z), values=ctvol,
                                      bounds_error=False, fill_value=fillval)
        ictvol = rgi(IXYZ).reshape(ix.shape)

        # Interpolate the mask volume.
        rgi = RegularGridInterpolator(points=(x, y, z), values=mask,
                                      bounds_error=False, fill_value=False)
        imask = rgi(IXYZ).reshape(ix.shape)

        return ictvol, imask


# Add the relationship to the Scan model.
Scan.annotations = relationship('Annotation',
                                order_by=Annotation.id,
                                back_populates='scan')
