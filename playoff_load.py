''' load functions for mlb data tables '''
from models.sqla_utils import PLAYOFFBASE, get_session
from models.playoff_plate_appearance import PlateAppearance
from models.playoff_game import Game
from models.playoff_run import Run
from models.playoff_br_event import BaseRunningEvent
from sqlalchemy import MetaData
import concurrent.futures
MODELS = [PlateAppearance, Game, Run, BaseRunningEvent]


def create_tables():
    ''' creates all tables in the tables list '''
    PLAYOFFBASE.metadata.create_all(tables=[x.__table__ for x in MODELS], checkfirst=True)

def load_data(results):
    '''
    Load playoff data into the SQL database
    @param results - a dictionary of lists of dictionaries containing the PlateAppearance, Game, Run, BaseRunningEvent data for the playoffs
    '''
    print('loading...')
    # Get the Playoff session
    session = get_session(True)
    for model in MODELS:
        print(model)
        data = results[model.__tablename__]
        i = 0
        # Here is where we convert directly the dictionary output of our marshmallow schema into sqlalchemy
        objs = []
        games = set([])
        for row in data:
            session.merge(model(**row))
            
            i += 1
        
    session.commit()
    print('loaded')