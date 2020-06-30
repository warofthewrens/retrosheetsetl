from marshmallow import Schema, fields, pre_load
from datetime import datetime

#TODO: much better validation
class Info(Schema):
    visiting_team = fields.String(data_key='visteam', missing=None)
    home_team = fields.String(data_key='hometeam', missing=None)
    site = fields.String(missing=None)
    date = fields.DateTime(format='%Y/%m/%d', missing=None)
    number = fields.Integer(missing=None)
    start_time = fields.String(data_key='starttime', missing=None)
    daynight = fields.Boolean(data_key='daynight', missing=None)
    usedh = fields.Boolean(missing=None)
    umphome = fields.String(missing=None)
    ump1b = fields.String(missing=None)
    ump2b = fields.String(missing=None)
    ump3b = fields.String(missing=None)
    umplf = fields.String(missing=None)
    umprf = fields.String(missing=None)
    how_scored = fields.String(data_key='howscored', missing=None)
    pitches = fields.String(missing=None)
    oscorer = fields.String(missing=None)
    temp = fields.Integer(missing=None)
    wind_dir = fields.String(data_key='winddir', missing=None)
    wind_speed = fields.Integer(data_key='windspeed', missing=None)
    field_cond = fields.String(data_key='fieldcond', missing=None)
    precip = fields.String(missing=None)
    sky = fields.String(missing = None)
    time_of_game = fields.Integer(data_key='timeofgame', missing = None)
    attendance = fields.Integer(missing=None)
    winning_pitcher = fields.String(data_key='wp', missing=None)
    losing_pitcher = fields.String(data_key='lp', missing=None)
    save = fields.String(allow_none=True, missing=None)


    @pre_load
    def is_day(self, data, **kwargs):
        data['daynight'] = (data['daynight'] == 'day')
        return data