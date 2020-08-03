from sqlalchemy import Column, ForeignKey
from sqlalchemy.dialects.mysql import MEDIUMINT, INTEGER, TINYINT, SMALLINT, VARCHAR, DATETIME, BOOLEAN, DATE, FLOAT
from sqlalchemy.orm import relationship
from models.sqla_utils import BASE

class StarterPosition(BASE):
    __tablename__ = 'StarterPosition'

    team = Column(VARCHAR(3), primary_key=True)
    year = Column(INTEGER(4), primary_key=True)
    starter_1 = Column(VARCHAR(8))
    starter_1_WAR = Column(FLOAT(5))
    starter_2 = Column(VARCHAR(8))
    starter_2_WAR = Column(FLOAT(5))
    starter_3 = Column(VARCHAR(8))
    starter_3_WAR = Column(FLOAT(5))
    starter_4 = Column(VARCHAR(8))
    starter_4_WAR = Column(FLOAT(5))
    starter_5 = Column(VARCHAR(8))
    starter_5_WAR = Column(FLOAT(5))