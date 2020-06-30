from sqlalchemy import Column
from sqlalchemy.dialects.mysql import MEDIUMINT, INTEGER, TINYINT, SMALLINT, VARCHAR, DATETIME, BOOLEAN, DATE, FLOAT
from models.sqla_utils import BASE

class Team(BASE):
    __tablename__ = 'team'

    team = Column(VARCHAR(3), primary_key=True)
    year = Column(INTEGER(4), primary_key=True)
    W = Column(INTEGER(3))
    L = Column(INTEGER(3))
    win_pct = Column(FLOAT(5))
    homeW = Column(INTEGER(2))
    homeL = Column(INTEGER(2))
    awayW = Column(INTEGER(2))
    awayL = Column(INTEGER(2))
    RS = Column(INTEGER(4))
    RA = Column(INTEGER(4))
    DIFF = Column(INTEGER(4))
    exp_win_pct = Column(FLOAT(5))
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
    SO = Column(INTEGER(3))
    HBP = Column(INTEGER(3))
    SF = Column(INTEGER(3))
    SH = Column(INTEGER(3))
    AVG = Column(FLOAT(5))
    OBP = Column(FLOAT(5))
    SLG = Column(FLOAT(5))
    OPS = Column(FLOAT(5))
    BF = Column(INTEGER(4))
    IP = Column(FLOAT(5))
    Ha = Column(INTEGER(4))
    HRa = Column(INTEGER(4))
    TBa = Column(INTEGER(4))
    BBa = Column(INTEGER(4))
    IBBa = Column(INTEGER(3))
    K = Column(INTEGER(3))
    HBPa = Column(INTEGER(3))
    BK = Column(INTEGER(3))
    SV = Column(INTEGER(2))
    TR = Column(INTEGER(3))
    ER = Column(INTEGER(3))
    RAA = Column(FLOAT(4))
    ERA = Column(FLOAT(4))
    SpIP = Column(FLOAT(5))
    RpIP = Column(FLOAT(5))
    SpER = Column(INTEGER(4))
    RpER = Column(INTEGER(4))
    SpTR = Column(INTEGER(4))
    RpTR = Column(INTEGER(4))
    SpERA = Column(FLOAT(5))
    RpERA = Column(FLOAT(5))



