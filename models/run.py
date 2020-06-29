from sqlalchemy import Column
from sqlalchemy.dialects.mysql import MEDIUMINT, INTEGER, TINYINT, SMALLINT, VARCHAR, DATETIME, BOOLEAN, DATE
from models.sqla_utils import BASE

class Run(BASE):
    __tablename__ = 'Run'

    run_id = Column(INTEGER(12), primary_key=True, auto_increment=True, default=0)
    game_id = Column(VARCHAR(12), primary_key=True)
    year = Column(INTEGER(5))
    date = Column(DATE)
    scoring_team = Column(VARCHAR(3))
    conceding_team = Column(VARCHAR(3))
    scoring_player = Column(VARCHAR(8))
    batter = Column(VARCHAR(8))
    responsible_pitcher = Column(VARCHAR(8))
    is_earned = Column(BOOLEAN(3))
    is_team_earned = Column(BOOLEAN(3))
    is_rbi = Column(BOOLEAN(3))
    is_sp = Column(BOOLEAN(3))
    inning = Column(SMALLINT(3))
    outs = Column(TINYINT(1))