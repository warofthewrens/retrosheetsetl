from sqlalchemy import Column, ForeignKey
from sqlalchemy.dialects.mysql import MEDIUMINT, INTEGER, TINYINT, SMALLINT, VARCHAR, DATETIME, BOOLEAN, DATE, FLOAT
from sqlalchemy.orm import relationship
from models.sqla_utils import BASE

class TeamPosition(BASE):
    __tablename__ = 'TeamPosition'

    year = Column(INTEGER(4), primary_key=True)
    team = Column(VARCHAR(4), primary_key=True)
    position_code = Column(INTEGER(4), primary_key=True)
    position = Column(VARCHAR(20))
    wRAA = Column(FLOAT(5))
    PA_first = Column(VARCHAR(8))
    PA_first_wRAA = Column(FLOAT(5))
    PA_first_PA = Column(FLOAT(5))
    PA_second = Column(VARCHAR(8))
    PA_second_wRAA = Column(FLOAT(5))
    PA_second_PA = Column(FLOAT(4))
    PA_third = Column(VARCHAR(8))
    PA_third_wRAA = Column(FLOAT(5))
    PA_third_PA = Column(FLOAT(4))