''' utils for maintaining database with sqlalchemy '''
from sqlalchemy import create_engine, Column, MetaData, Table, Integer, Date, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session
#from marshmallow_schemas.schema_utils import password


ENGINE = create_engine('mysql+pymysql://root:Westhamphillie$2008@localhost/retrosheets', echo=False)
engine = create_engine("sqlite:///myexample.db")  # Access the DB Engine
if not engine.dialect.has_table(engine, 'PlateAppearance'):  # If table don't exist, Create.
    metadata = MetaData(engine)
    # Create a table with the appropriate Columns
BASE = declarative_base(bind=ENGINE)


def get_session():
    '''
    gets a sqlalchemy session -- useful for interfacing with the database
    Docs: https://docs.sqlalchemy.org/en/13/orm/session.html
    '''
    return Session(bind=ENGINE)