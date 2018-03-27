import sqlalchemy as sq
from sqlalchemy.orm import relationship
import numpy as np
from ._Base import Base
from .Scan import Scan
from .Annotation import Annotation

_off_limits = ['id','annotation_id','annotation',
               'inclusion','image_z_position','dicom_file_name','coords']

class Contour(Base):
    """
    The Contour class holds the coordinate data for the vertices of
    a :class:`pylidc.Annotation` object for a single slice in the CT volume.

    inclusion: bool
        If True, the area inside the contour is included as part of 
        the nodule. If False, the area inside the contour is excluded 
        from the nodule.

    image_z_position: float
        This is the `imageZposition` defined via the xml annnotations 
        for this contour. It is the z-coordinate of dicom 
        attribute, `(0020,0032)`.
    
    dicom_file_name: string
        This is the name of the corresponding dicom file for the scan 
        to which this contour belongs, having the same `image_z_position`.

    coords: string
        These are the sequential (x,y) coordinates of the curve, stored 
        as a string. It is better to access these coordinates using the 
        `to_matrix` method, which returns a numpy array rather than 
        a string.

    Example:
        >>> import pylidc as pl
        >>> import matplotlib.pyplot as plt
        >>>
        >>> ann = pl.query(pl.Annotation).first()
        >>> vol = ann.scan.to_volume()
        >>> ii,jj = ann.contours[0].to_matrix(include_k=False).T
        >>> 
        >>> plt.imshow(vol[:,:,46], cmap=plt.cm.gray)
        >>> plt.plot(jj, ii, '-r', lw=1, label="Nodule Boundary")
        >>> plt.legend()
        >>> plt.show()
    """
    __tablename__    = 'contours'
    id               = sq.Column('id', sq.Integer, primary_key=True)
    annotation_id    = sq.Column(sq.Integer, sq.ForeignKey('annotations.id'))
    annotation       = relationship('Annotation', back_populates='contours')
    inclusion        = sq.Column('inclusion', sq.Boolean)
    image_z_position = sq.Column('image_z_position', sq.Float)
    dicom_file_name  = sq.Column('dicom_file_name', sq.String)
    coords           = sq.Column('coords', sq.String)

    def __repr__(self):
        return "Contour(id=%d,annotation_id=%d)"%(self.id, self.annotation_id)

    def __setattr__(self, name, value):
        if name in _off_limits:
            msg = "Trying to assign read-only Contour object attribute \
                   `%s` a value of `%s`." % (name,value)
            raise ValueError(msg)
        else:
            super(Contour,self).__setattr__(name,value)

    def to_matrix(self, include_k=True):
        """
        Return the contour-annotation coordinates as a matrix where 
        each row contains an (i,j,k) *index* coordinate into the image volume.
        
        include_k: bool, default=True
            Set `include_k=False` to omit the `k` axis coordinate. In this 
            case, each row of the returned matrix is an (i,j) *index* 
            coordinate into the `k`th image. The index `k` can be 
            recovered via:
                z = contour.image_z_position
                k = contour.annotation.contour_slice_indexes[z]
        """
        # The reversal [::-1] is because the coordinates from the LIDC XML
        # are stored as (x,y), not (i,j).
        ij = np.array([[int(cc) for cc in c.split(',')][::-1]
                        for c in self.coords.split('\n')])
        if not include_k:
            return ij
        else:
            k  = np.ones(ij.shape[0])
            zs = self.annotation.contour_slice_zvals
            kk = np.abs(zs-self.image_z_position).argmin()
            k *= self.annotation.contour_slice_indices[kk]
            return np.c_[ij, k].astype(np.int)
    
Annotation.contours = relationship('Contour',
                                   order_by=Contour.id,
                                   back_populates='annotation')
