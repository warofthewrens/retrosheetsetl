from marshmallow import Schema, fields, validate, pre_dump, post_dump

class BaseRunningEvent(Schema):
    game_id = fields.String()
    date = fields.DateTime(format='%Y/%m/%d')
    running_team = fields.String()
    pitching_team = fields.String()
    event = fields.String()
    base = fields.String()
    runner = fields.String()
    pitcher = fields.String()
    catcher = fields.String()
    inning = fields.Integer()
    outs = fields.Integer()
    
