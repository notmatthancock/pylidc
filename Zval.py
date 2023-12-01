import os, warnings
import sqlalchemy as sq
from sqlalchemy.orm import relationship
from ._Base import Base
from .Scan import Scan

_off_limits = ['id','scan_id','scan','val'] 

class Zval(Base):
    """
    """
    __tablename__ = 'zvals'
    id            = sq.Column('id', sq.Integer, primary_key=True)
    scan_id       = sq.Column(sq.Integer, sq.ForeignKey('scans.id'))
    scan          = relationship('Scan', back_populates='zvals')
    val           = sq.Column('val', sq.Float)

    def __repr__(self):
        return "Zval(id=%d,scan_id=%d,val=%f)" % (self.id, self.scan_id,
                                                  self.val)

    def __setattr__(self, name, value):
        if name in _off_limits:
            msg = "Trying to assign read-only Annotation object attribute \
                   `%s` a value of `%s`." % (name,value)
            raise ValueError(msg)
        else:
            super(Zval,self).__setattr__(name,value)

    def __float__(self): return self.val

# Add the relationship to the Scan model.
Scan.zvals = relationship('Zval', order_by=Zval.id, back_populates='scan')
