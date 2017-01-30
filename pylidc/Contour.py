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
    The Contour model class holds a contour for a single scan slice 
    of a Nodule object, as drawn by the annotating radiologist.

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
        >>> scan = pl.query(pl.Scan).first()
        >>> ann  = scan.annotations[0]

        >>> for c in ann.contours:
        >>>     print(c.image_z_position, c.to_matrix().mean(axis=0))

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

    def to_matrix(self):
        """
        Return the contour-annotation coordinates as a matrix where 
        each row contains an (x,y,z) coordinate.
        """
        xy = np.array([list(map(int,c.split(','))) for c in self.coords.split('\n')])
        z  = np.ones(xy.shape[0])*self.image_z_position
        return np.c_[xy,z]
    
Annotation.contours = relationship('Contour',
                                   order_by=Contour.id,
                                   back_populates='annotation')
