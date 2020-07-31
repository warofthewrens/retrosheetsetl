from sqlalchemy import Column, ForeignKey
from sqlalchemy.dialects.mysql import MEDIUMINT, INTEGER, TINYINT, SMALLINT, VARCHAR, DATETIME, BOOLEAN, DATE, FLOAT
from sqlalchemy.orm import relationship
from models.sqla_utils import BASE

class Series(BASE):
    __tablename__ = 'Series'

    series_id = Column(VARCHAR(8), primary_key=True)
    year = Column(INTEGER(4))
    series = Column(VARCHAR(4))
    winning_team = Column(VARCHAR(3))
    losing_team = Column(VARCHAR(3))
    team_with_home_team_adv = Column(VARCHAR(3))