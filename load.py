''' load functions for nhl data tables '''
from models.sqla_utils import BASE, get_session
from models.plate_appearance import PlateAppearance
from models.game import Game
from sqlalchemy import MetaData
MODELS = [PlateAppearance, Game]


def create_tables():
    ''' creates all tables in the tables list '''
    BASE.metadata.create_all(tables=[x.__table__ for x in MODELS], checkfirst=True)


def load_data(results):
    '''
    takes in parsed data from transform and uses sqlalchemy to load into the database; note we dump the results
    of the parsed schema directly into our SQLAlchemy models, then commit them to the DB!
    @param dict results: results of transforming raw data
    '''
    # create a session: https://docs.sqlalchemy.org/en/13/orm/session.html
    session = get_session()
    for model in MODELS:
        data = results[model.__tablename__]
        i = 0
        # Here is where we convert directly the dictionary output of our marshmallow schema into sqlalchemy
        for row in data:
            if i % 1000 == 0:
                print('loading...', i)
            i+=1
            session.merge(model(**row))

    session.commit()