from marshmallow import Schema, fields

class TeamPosition(Schema):

    year = fields.Integer()
    team = fields.String()
    position_code = fields.Integer()
    position = fields.String()
    wRAA = fields.Float()
    PA_first = fields.String()
    PA_first_wRAA = fields.Float()
    PA_first_PA = fields.Float()
    PA_second = fields.String()
    PA_second_wRAA = fields.Float()
    PA_second_PA = fields.Float()
    PA_third = fields.String()
    PA_third_wRAA = fields.Float()
    PA_third_PA = fields.Float()
