''' load functions for nhl data tables '''
from models.sqla_utils import BASE, get_session
from models.plate_appearance import PlateAppearance
from models.game import Game
from models.run import Run
from models.base_running_event import BaseRunningEvent
from sqlalchemy import MetaData
import concurrent.futures
MODELS = [PlateAppearance, Game, Run, BaseRunningEvent]



def create_tables():
    ''' creates all tables in the tables list '''
    BASE.metadata.create_all(tables=[x.__table__ for x in MODELS], checkfirst=True)

def merge(session, model, row, i):
    return model(**row)

def load_data(results):
    print('loading...')
    session = get_session()
    for model in MODELS:
        print(model)
        data = results[model.__tablename__]
        i = 0
        # Here is where we convert directly the dictionary output of our marshmallow schema into sqlalchemy
        objs = []
        for row in data:
            objs.append(merge(session, model, row, i))
            i += 1
        # results = [executor.submit(merge, session, model, row, i) for row in data]
        # objs = []
        # for result in concurrent.futures.as_completed(results):
        #     objs.append(result.result())
        # for row in data:
        session.bulk_save_objects(objs)
    session.commit()
    print('loaded')

    