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
__version__ = 0.1

# Hidden stuff.
import os as _os
from sqlalchemy import create_engine as _create_engine
from sqlalchemy.orm import sessionmaker as _sessionmaker
from ._Configuration import _Configuration

_module_path = _os.path.dirname(_os.path.abspath(__file__))
_path        = _os.path.join(_module_path,'db','pylidc.sqlite')
_engine      = _create_engine('sqlite:///'+_path)
_session     = _sessionmaker(bind=_engine)()

# Public stuff.
from .Scan import Scan
from .Annotation import Annotation
from .Annotation import _all_characteristics_ 
from .Contour import Contour

def query(*args):
    """
    Wraps the sqlalchemy session object. Some example usage:
    
        >>> import pylidc as pl
        >>> qu = pl.query(pl.Scan).filter(pl.Scan.resolution_z <= 1.)
        >>> print qu.count()
        >>> # => 97
        >>> scan = qu.first()
        >>> print len(scan.annotations)
        >>> # => 11
        >>> qu = pl.query(pl.Annotation).filter((pl.Annotation.malignancy > 3) & (pl.Annotation.spiculation < 3))
        >>> print qu.count()
        >>> # => 1083
        >>> annotation = qu.first()
        >>> print annotation.estimate_volume()
        >>> # => 5230.33874999
    """
    return _session.query(*args)

def get_path_to_dicom_files():
    """
    Get the root path to where the LIDC dicom data is stored. Note 
    that the path is stored in an sqlite database, and so it will 
    persist after it has been set.

    Functions for visualization will look in: `path_to_dicom_files/[scan.patient_id]/[scan.study_instance_uid]/[scan.series_instance_uid]/*.dcm`.

    Variables in brackets above will be set according to the pylidc.Scan 
    or pylidc.Annotation object that the function is called on. If the 
    scans were downloaded using the TCIA download manager, the path should 
    probably look like: `place/where/you/store/your/data/LIDC-IDRI`.
    """
    cfg_obj = _session.query(_Configuration)
    cfg_obj = cfg_obj.filter(_Configuration.key=='path_to_dicom_files').first()

    # It should never be the case that the configuration is missing,
    # but check for it and add it if missing anyway.
    if cfg_obj is None:
        _session.add(_Configuration(key='path_to_dicom_files',value=''))
        _session.commit()
        return ''
    else:
        return cfg_obj.value

def set_path_to_dicom_files(path):
    """
    Set the root path to where the LIDC dicom data is stored. Note that 
    the path is stored in an sqlite database, and so it will persist 
    after it has been set.

    path: string or None (default None)
        If a string is provided, the path will be updated. If no argument 
        is provided, the current root path is returned.

    Functions for visualization will look in: `path_to_dicom_files/[scan.patient_id]/[scan.study_instance_uid]/[scan.series_instance_uid]/*.dcm`

    Variables in brackets above will be set according to the scan or 
    annotation that the function is called on. If the scans were downloaded 
    using the TCIA download manager, the path should probably 
    look like: `place/where/you/store/your/data/LIDC-IDRI`.
    """
    path = str(path)

    cfg_obj = _session.query(_Configuration)
    cfg_obj = cfg_obj.filter(_Configuration.key=='path_to_dicom_files').first()

    # It should never be the case that the configuration is missing, 
    # but check for it and add it if missing anyway.
    if cfg_obj is None:
        cfg_obj = _Configuration(key='path_to_dicom_files',value=path)
        _session.add(cfg_obj)
    else:
        cfg_obj.value = path
    _session.commit()
    print("Path updated successfully to `%s`"%path)
