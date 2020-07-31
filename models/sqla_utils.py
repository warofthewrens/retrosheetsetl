''' utils for maintaining database with sqlalchemy '''
from sqlalchemy import create_engine, Column, MetaData, Table, Integer, Date, String, Float, event, exc
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session
import os
#from marshmallow_schemas.schema_utils import password


ENGINE = create_engine('mysql+pymysql://warofthewrens:DeJesus18Carlton32@35.245.199.134:3306/retrosheet', echo=False)
PLAYOFF_ENGINE = create_engine('mysql+pymysql://warofthewrens:DeJesus18Carlton32@35.245.199.134:3306/playoffs', echo=False)

engine = create_engine("sqlite:///myexample.db")  # Access the DB Engine
if not engine.dialect.has_table(engine, 'PlateAppearance'):  # If table don't exist, Create.
    metadata = MetaData(engine)
    # Create a table with the appropriate Columns
BASE = declarative_base(bind=ENGINE)
PLAYOFFBASE = declarative_base(bind=PLAYOFF_ENGINE)

# taken from https://docs.sqlalchemy.org/en/13/core/pooling.html
@event.listens_for(engine, "connect")
def connect(dbapi_connection, connection_record):
    connection_record.info['pid'] = os.getpid()

@event.listens_for(engine, "checkout")
def checkout(dbapi_connection, connection_record, connection_proxy):
    pid = os.getpid()
    if connection_record.info['pid'] != pid:
        connection_record.connection = connection_proxy.connection = None
        raise exc.DisconnectionError(
                "Connection record belongs to pid %s, "
                "attempting to check out in pid %s" %
                (connection_record.info['pid'], pid)
        )

def get_session(is_playoff=False):
    '''
    gets a sqlalchemy session -- useful for interfacing with the database
    Docs: https://docs.sqlalchemy.org/en/13/orm/session.html

    '''
    if is_playoff:
        return Session(bind=PLAYOFF_ENGINE)
    return Session(bind=ENGINE)