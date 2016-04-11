from ._Base import Base
import sqlalchemy as sq

class _Configuration(Base):
    """
    The Configuration model only consists of a key and a value.
    For each configuration, key refers to the name of the configuration,
    and value refers to the corresponding value.

    Currently, the only configuration option actually stored in the databaseis the path to where dicom files are located.
    """
    __tablename__ = 'configurations'
    id            = sq.Column('id', sq.Integer, primary_key=True)
    key           = sq.Column('key', sq.String, unique=True)
    value         = sq.Column('value', sq.String)
