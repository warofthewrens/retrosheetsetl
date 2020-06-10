from marshmallow import Schema, fields, pre_load
from datetime import datetime

#TODO: much better validation
class Info(Schema):
    visiting_team = fields.String(data_key='visteam')
    home_team = fields.String(data_key='hometeam')
    site = fields.String()
    date = fields.DateTime(format='%Y/%m/%d')
    number = fields.Integer()
    start_time = fields.String(data_key='starttime')
    daynight = fields.Boolean(data_key='daynight')
    usedh = fields.Boolean()
    umphome = fields.String()
    ump1b = fields.String()
    ump2b = fields.String()
    ump3b = fields.String()
    how_scored = fields.String(data_key='howscored')
    pitches = fields.String()
    oscorer = fields.String()
    temp = fields.Integer()
    wind_dir = fields.String(data_key='winddir')
    wind_speed = fields.Integer(data_key='windspeed')
    field_cond = fields.String(data_key='fieldcond')
    precip = fields.String()
    sky = fields.String()
    time_of_game = fields.Integer(data_key='timeofgame')
    attendance = fields.Integer()
    winning_pitcher = fields.String(data_key='wp')
    losing_pitcher = fields.String(data_key='lp')
    save = fields.String(allow_none=True, missing=None)


    @pre_load
    def is_day(self, data, **kwargs):
        data['daynight'] = (data['daynight'] == 'day')
        return data