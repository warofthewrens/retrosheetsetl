from sqlalchemy import Column, ForeignKey
from sqlalchemy.dialects.mysql import MEDIUMINT, INTEGER, TINYINT, SMALLINT, VARCHAR, DATETIME, BOOLEAN, DATE, FLOAT
from sqlalchemy.orm import relationship
from models.team import Team
from models.sqla_utils import BASE

class Player(BASE):
    __tablename__ = 'Player'

    player_id = Column(VARCHAR(8), primary_key=True)
    player_name = Column(VARCHAR(50))
    team = Column(VARCHAR(3), primary_key=True)
    year = Column(INTEGER(4), primary_key=True)
    GS = Column(INTEGER(3))
    GP = Column(INTEGER(3))
    PA = Column(INTEGER(4))
    AB = Column(INTEGER(4))
    S = Column(INTEGER(3))
    D = Column(INTEGER(3))
    T = Column(INTEGER(3))
    HR = Column(INTEGER(3))
    TB = Column(INTEGER(3))
    H = Column(INTEGER(3))
    R = Column(INTEGER(3))
    RBI = Column(INTEGER(3))
    SB = Column(INTEGER(3))
    CS = Column(INTEGER(3))
    BB = Column(INTEGER(3))
    IBB = Column(INTEGER(3))
    SO = Column(INTEGER(3))
    HBP = Column(INTEGER(3))
    SF = Column(INTEGER(3))
    SH = Column(INTEGER(3))
    AVG = Column(FLOAT(5))
    OBP = Column(FLOAT(5))
    SLG = Column(FLOAT(5))
    OPS = Column(FLOAT(5))
    wOBA = Column(FLOAT(5))
    wRAA = Column(FLOAT(5))
    BF = Column(INTEGER(4))
    IP = Column(FLOAT(5))
    Ha = Column(INTEGER(4))
    HRa = Column(INTEGER(4))
    TBa = Column(INTEGER(4))
    BBa = Column(INTEGER(4))
    IBBa = Column(INTEGER(3))
    K = Column(INTEGER(3))
    HBPa = Column(INTEGER(3))
    IFFB = Column(INTEGER(3))
    BK = Column(INTEGER(3))
    W = Column(INTEGER(2))
    L = Column(INTEGER(2))
    SV = Column(INTEGER(2))
    TR = Column(INTEGER(3))
    ER = Column(INTEGER(3))
    RA = Column(FLOAT(4))
    ERA = Column(FLOAT(4))
    FIP = Column(FLOAT(4))
    iFIP = Column(FLOAT(4))
    FIPR9 = Column(FLOAT(4))
    pFIPR9 = Column(FLOAT(4))
    dRPW = Column(FLOAT(4))
    RAAP9 = Column(FLOAT(4))
    WPGAA = Column(FLOAT(4))
    WPGAR = Column(FLOAT(5))
    WAR = Column(FLOAT(5))

    #player_teams = relationship('Team', primaryjoin= team == Team.team)
    #
    # player_teams = relationship('Team', primaryjoin='and_(Player.team == Team.team, Player.year == Team.year)')

