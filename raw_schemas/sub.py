from marshmallow import Schema, fields

class Sub(Schema):
    player_id = fields.String()
    name = fields.String()
    is_home = fields.Boolean()
    bat_pos = fields.Integer()
    field_pos = fields.Integer()
    play_idx = fields.Integer()