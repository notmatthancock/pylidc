import os
import sqlalchemy as sq
from sqlalchemy.orm import relationship
from ._Base import Base
from .Scan import Scan

import dicom
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mplpath
from matplotlib.patches import Rectangle
from matplotlib.widgets import Slider, Button, CheckButtons
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import Delaunay
from scipy.spatial.distance import pdist,squareform
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import mode

_char_to_word_ = ('Low', 'Medium-Low', 'Medium', 'Medium-High', 'High')
_all_characteristics_ = \
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
              list(_all_characteristics_)

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
        >>> print ann.spiculation
        >>> # => 4
        >>> # Each nodule characteristic has a helper function to print the semantic value.
        >>> print ann.Spiculation()
        >>> # => Medium-High Spiculation
        >>> 
        >>> q = pl.query(pl.Annotation).join(pl.Scan).filter(pl.Scan.resolution_z <= 1, pl.Annotation.malignancy == 5)
        >>> print q.count()
        >>> # => 58
        >>> ann = q.first()
        >>> print ann.estimate_diameter(), ann.estimate_volume()
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
        """return subtlety value as semantic string"""
        s = self.subtlety
        assert s in range(1,6), "Subtlety score out of bounds."
        return _char_to_word_[::-1][ s-1 ] + ' Subtlety'
    def InternalStructure(self):
        """return internalStructure value as semantic string"""
        s = self.internalStructure
        assert s in range(1,5), "Internal structure score out of bounds."
        if   s == 1: return 'Soft Tissue'
        elif s == 2: return 'Fluid'
        elif s == 3: return 'Fat'
        elif s == 4: return 'Air'
    def Calcification(self):
        """return calcification value as semantic string"""
        s = self.calcification
        assert s in range(1,7), "Calcification score out of bounds."
        if   s == 1: return 'Popcorn'
        elif s == 2: return 'Laminated'
        elif s == 3: return 'Solid'
        elif s == 4: return 'Non-central'
        elif s == 5: return 'Central'
        elif s == 6: return 'Absent'
    def Sphericity(self):
        """return sphericity value as semantic string"""
        s = self.sphericity
        assert s in range(1,6), "Sphericity score out of bounds."
        if   s == 1: return 'Linear'
        elif s == 2: return 'Ovoid Linear'
        elif s == 3: return 'Ovoid'
        elif s == 4: return 'Ovoid Round'
        elif s == 5: return 'Round'
    def Margin(self):
        """return margin value as semantic string"""
        s = self.margin
        assert s in range(1,6), "Margin score out of bounds."
        if   s == 1: return 'Poor'
        elif s == 2: return 'Near Poor'
        elif s == 3: return 'Medium'
        elif s == 4: return 'Near Sharp'
        elif s == 5: return 'Sharp'
    def Lobulation(self):
        """return lobulation value as semantic string"""
        s = self.lobulation
        assert s in range(1,6), "Lobulation score out of bounds."
        return _char_to_word_[ s-1 ] + ' Lobulation'
    def Spiculation(self):
        """return spiculation value as semantic string"""
        s = self.spiculation
        assert s in range(1,6), "Spiculation score out of bounds."
        return _char_to_word_[ s-1 ] + ' Spiculation'
    def Texture(self):
        """return texture value as semantic string"""
        s = self.texture
        assert s in range(1,6), "Texture score out of bounds."
        if   s == 1: return 'Non-Solid / Ground Glass Opacity Texture'
        elif s == 2: return 'Non-Solid or Mixed Texture'
        elif s == 3: return 'Part Solid or Mixed Texture'
        elif s == 4: return 'Mixed or Solid Texure'
        elif s == 5: return 'Solid Texture'
    def Malignancy(self):
        """return malignancy value as semantic string"""
        s = self.malignancy
        assert s in range(1,6), "Malignancy score out of bounds."
        return _char_to_word_[ s-1 ] + ' Malignancy'
    # } End semantic attribute functions
    ####################################

    def all_characteristics_as_string(self):
        """
        Return all characteristic values as a string table.
        """
        chars1 = _all_characteristics_
        chars2 = [ch.title() for ch in chars1]
        chars2[chars2.index('Internalstructure')] = 'InternalStructure'

        s = ('%-18s   %-24s   %-2s'%('Characteristic', 'Semantic value','#'))
        s+= '\n'
        s+= ('%-18s   %-24s   %-2s' % ('-', '-', '-')) + '\n'

        for i in range(len(chars1)):
            attrs = (chars2[i],\
                     getattr(self,chars2[i])(),
                     getattr(self,chars1[i]))
            s += '%-18s | %-24s | %-2d' % attrs
            s += '\n'
        return s[:-1] # cut the trailing newline character

    def all_characteristics_as_array(self):
        """
        Return all characteristic values as a numpy array in the order 
        presented in `pylidc._all_characteristics_`.
        """
        return np.array([getattr(self,char) for char in _all_characteristics_])

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

    def centroid(self):
        """
        Return the center of mass of the nodule as determined by its 
        radiologist-drawn contours. Note that the first two components 
        are the mean in image coordinates, while the last component of 
        the centroid is the mean of `image_z_position`s.
        """
        return self.contours_to_matrix().mean(axis=0)

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
    
    def visualize_in_3d(self,**kwargs):
        """
        This method is for rough visualization only, and could look nasty 
        if the annotated nodule has a shape that deviates significantly 
        from spherical. Any keyword argument that is accepted by matplotlib's 
        `plot_trisurf` function can passed as an argument to this function.

        The surface of the annotated nodule is triangulated by transforming 
        the annotated contours into spherical coordinates, and triangulating 
        the azimuth and zenith angles. Again, this could look quite bad if 
        the nodule deviates from a roughly spherical shape.
        """
        fig = plt.figure(figsize=(7,7))
        ax  = fig.add_subplot(111, projection='3d')

        points = np.vstack([
            c.to_matrix() for c in self.contours if c.inclusion
        ])
        points[:,:2] = points[:,:2] * self.scan.pixel_spacing

        # Center the points at the origin for 
        # spherical coordinates conversion.
        points = points - points.mean(axis=0)

        # Triangulate the azimuth and zenith transformation.
        azimuth = np.arctan2(points[:,1],points[:,0])
        zenith  = np.arccos(points[:,2] / np.linalg.norm(points,axis=1))
        azi_zen = np.c_[azimuth.flatten(),zenith.flatten()]
        triangles = Delaunay(azi_zen).simplices

        # Start the points at 0 on every axis.
        # This lets the axis ticks to be interpreted as length in mm.
        points = points - points.min(axis=0)

        ax.set_xlabel('length (mm)')
        ax.set_ylabel('length (mm)')
        ax.set_zlabel('length (mm)')

        # Plot the points.
        ax.plot_trisurf(points[:,0], points[:,1], points[:,2],
                        triangles=triangles, **kwargs)
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
        ax_scan_info.set_axis_bgcolor('w')
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

        # Add annotations / characteristics table.
        ax_annotation_info = fig.add_axes([0.1, 0.45, 0.3, 0.25])
        ax_annotation_info.set_axis_bgcolor('w')

        # Create the rows to be displayed in the annotations table.
        cell_text = []
        for c in _all_characteristics_:
            row = []
            cname = c.capitalize()
            if cname.startswith('Int'):
                cname = 'InternalStructure'

            row.append(cname)
            row.append(getattr(self,cname)())
            row.append(getattr(self,c))

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
        ax_contour_checkbox.set_axis_bgcolor('w')
        contour_checkbox = CheckButtons(ax_contour_checkbox,
                                        ('Show Contours',), (True,))
        contour_checkbox.is_checked = True

        # Add the widgets.
        ax_slice = fig.add_axes([0.1, 0.1, 0.3, 0.05])
        ax_slice.set_axis_bgcolor('w')
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

    def contours_to_matrix(self):
        """
        Return all the contours in a 3D numpy array. Note that the first 
        two columns are in image coordinates while the latter is given as 
        the `image_z_position`. Thus, the image resolution is not accounted 
        for in the first two columns.
        """
        return np.vstack([c.to_matrix() for c in self.contours])

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
            points_in_contour = map(tuple, points_in_contour)

            # Update the corresponding set.
            if contour.inclusion:
                included.update(points_in_contour)
            else:
                excluded.update(points_in_contour)
        # Return the included points minus the excluded points.
        return included.difference( excluded )


# Add the relationship to the Scan model.
Scan.annotations = relationship('Annotation',
                                order_by=Annotation.id,
                                back_populates='scan')
