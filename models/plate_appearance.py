from sqlalchemy import Column
from sqlalchemy.dialects.mysql import MEDIUMINT, INTEGER, TINYINT, SMALLINT, VARCHAR, DATETIME, BOOLEAN
from models.sqla_utils import BASE

class PlateAppearance(BASE):
    __tablename__ = 'PlateAppearance'

    game_id = Column(VARCHAR(12), primary_key=True, auto_increment=False)
    pa_id = Column(INTEGER(6), primary_key=True, auto_increment=True, default=0)
    batter_id = Column(VARCHAR(8))
    pitcher_id = Column(VARCHAR(8))  
    play = Column(VARCHAR(150))
    inning = Column(SMALLINT(3))
    batting_team_home = Column(BOOLEAN(3))
    outs = Column(TINYINT(1))
    balls = Column(TINYINT(1))
    strikes = Column(TINYINT(1))
    sequence = Column(VARCHAR(50))
    away_runs = Column(SMALLINT(3))
    home_runs = Column(SMALLINT(3))  
    first_runner_id = Column(VARCHAR(8))
    first_dest = Column(VARCHAR(3))
    second_runner_id = Column(VARCHAR(8))
    second_dest = Column(VARCHAR(3))
    third_runner_id = Column(VARCHAR(8))
    third_dest = Column(VARCHAR(3))
    field_pos = Column(SMALLINT(3))
    lineup_pos = Column(SMALLINT(3))
    event_type = Column(SMALLINT(3))
    ab_flag = Column(BOOLEAN(3))
    hit_val = Column(SMALLINT(3))
    sac_bunt = Column(BOOLEAN(3))
    sac_fly = Column(BOOLEAN(3))
    outs_on_play = Column(SMALLINT(3))
    rbi = Column(SMALLINT(3))
    wp = Column(BOOLEAN(3))
    pb = Column(BOOLEAN(3))
    fielder_id = Column(VARCHAR(8))
    ball_type = Column(VARCHAR(1))
    bunt_flag = Column(BOOLEAN(3))
    foul_flag = Column(BOOLEAN(3))
    hit_loc = Column(SMALLINT(3))
    first_error = Column(VARCHAR(8))
    second_error = Column(VARCHAR(3))
    third_error = Column(VARCHAR(3))
    num_errors = Column(VARCHAR(3))
    batter_dest = Column(VARCHAR(3))
    
    first_po = Column(VARCHAR(8))
    second_po = Column(VARCHAR(8))
    third_po = Column(VARCHAR(8))
    first_ast = Column(VARCHAR(8))
    second_ast = Column(VARCHAR(8))
    third_ast = Column(VARCHAR(8))
    fourth_ast = Column(VARCHAR(8))
    fifth_ast = Column(VARCHAR(8))