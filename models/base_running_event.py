from sqlalchemy import Column
from sqlalchemy.dialects.mysql import MEDIUMINT, INTEGER, TINYINT, SMALLINT, VARCHAR, DATETIME, BOOLEAN, DATE
from models.sqla_utils import BASE

class BaseRunningEvent(BASE):
    __tablename__ = 'BaseRunningEvent'

    event_id = Column(INTEGER(12), primary_key=True)
    game_id = Column(VARCHAR(12))
    date = Column(DATE)
    running_team = Column(VARCHAR(3))
    pitching_team = Column(VARCHAR(3))
    event = Column(VARCHAR(2))
    base = Column(VARCHAR(2))
    runner = Column(VARCHAR(8))
    pitcher = Column(VARCHAR(8))
    catcher = Column(VARCHAR(8))
    inning = Column(SMALLINT(3))
    outs = Column(TINYINT(1))