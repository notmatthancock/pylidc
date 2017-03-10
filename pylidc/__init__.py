"""
--------------------------------------------------------
Author: Matt Hancock, not.matt.hancock@gmail.com
--------------------------------------------------------

This python module implements an (ORM) object relational mapping 
to an sqlite database containing the annotation information from 
the XML files provided by the LIDC dataset. The purpose of this 
module is to make for easier data querying and to include 
functional aspects of the data models in addition to pure 
attribute information, e.g., computing nodule centroids from 
contour attribtues.

The ORM is implemented using sqlalchemy. There are three data models:

Scan, Annotation, and Contour

The relationships are "one to many" for each model going left 
to right, i.e., scans have many annotations and annotations 
have many contours.

For more information, see the model classes themselves.
"""
from __future__ import print_function as _pf

__version__ = '0.1.2'

# Hidden stuff.
import os as _os
import pkg_resources as _pr
from sqlalchemy import create_engine as _create_engine
from sqlalchemy.orm import sessionmaker as _sessionmaker

_dbpath  = _pr.resource_filename('pylidc', 'pylidc.sqlite')
_engine  = _create_engine('sqlite:///'+_dbpath)
_session = _sessionmaker(bind=_engine)()

# Public stuff.
from .Scan       import Scan
from .Scan       import dicompath
from .Annotation import Annotation
from .Contour    import Contour

def query(*args):
    """
    Wraps the sqlalchemy session object. Some example usage:
    
        >>> import pylidc as pl
        >>> qu = pl.query(pl.Scan).filter(pl.Scan.slice_thickness <= 1.)
        >>> print qu.count()
        >>> # => 97
        >>> scan = qu.first()
        >>> print len(scan.annotations)
        >>> # => 11
        >>> qu = pl.query(pl.Annotation).filter((pl.Annotation.malignancy > 3), (pl.Annotation.spiculation < 3))
        >>> print qu.count()
        >>> # => 1083
        >>> annotation = qu.first()
        >>> print annotation.estimate_volume()
        >>> # => 5230.33874999
    """
    return _session.query(*args)
