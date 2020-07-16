from sqlalchemy import Column
from sqlalchemy.dialects.mysql import MEDIUMINT, INTEGER, TINYINT, SMALLINT, VARCHAR, DATETIME, BOOLEAN, DATE
from models.sqla_utils import BASE

class League(BASE):
    __tablename__ = 'league'

    year = Column(INTEGER(4), primary_key=True)
    H = Column(INTEGER(5))
    HR = Column(INTEGER(5))
    BB = Column(INTEGER(5))
    K = Column(INTEGER(5))
    HBP = Columm(INTEGER(5))
    IP = Column(INTEGER(5))
    ER = Column(INTEGER(5))
    TR = Column(INTEGER(5))
    SpER = Column(INTEGER(5))
    SpTR = Column(INTEGER(5))
    SpIP = Column(INTEGER(5))
    RpER = Column(INTEGER(5))
    RpTR = Column(INTEGER(5))
    RpIP = Column(INTEGER(5))
    lgRAA = Column(FLOAT(5))
    lgERA = Column(FLOAT(5))
    lgFIP = Column(FLOAT(5))
    lgIFIP = Column(FLOAT(5))
    lgSpERA = Column(FLOAT(5))
    lgSpRAA = Column(FLOAT(5))
    lgSpFIP = Column(FLOAT(5))
    lgRpERA = Column(FLOAT(5))
    lgRpRAA = Column(FLOAT(5))
    lgRpFIP = Column(FLOAT(5))



