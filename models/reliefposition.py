from sqlalchemy import Column, ForeignKey
from sqlalchemy.dialects.mysql import MEDIUMINT, INTEGER, TINYINT, SMALLINT, VARCHAR, DATETIME, BOOLEAN, DATE, FLOAT
from sqlalchemy.orm import relationship
from models.sqla_utils import BASE

class ReliefPosition(BASE):
    __tablename__ = 'ReliefPosition'

    team = Column(VARCHAR(3), primary_key=True)
    year = Column(INTEGER(4), primary_key=True)
    closer = Column(VARCHAR(8))
    closer_WAR = Column(FLOAT(5))
    relief_1 = Column(VARCHAR(8))
    relief_1_WAR = Column(FLOAT(5))
    relief_2 = Column(VARCHAR(8))
    relief_2_WAR = Column(FLOAT(5))
    relief_3 = Column(VARCHAR(8))
    relief_3_WAR = Column(FLOAT(5))
    relief_4 = Column(VARCHAR(8))
    relief_4_WAR = Column(FLOAT(5))

