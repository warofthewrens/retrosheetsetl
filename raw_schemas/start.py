from marshmallow import Schema, fields

class Start(Schema):
    player_id = fields.String()
    name = fields.String()
    is_away = fields.Boolean()
    bat_pos = fields.Integer()
    field_pos = fields.Integer()