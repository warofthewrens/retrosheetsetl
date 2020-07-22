from marshmallow import Schema, fields, pre_load

class Start(Schema):
    player_id = fields.String()
    name = fields.String()
    is_home = fields.Boolean()
    bat_pos = fields.Integer()
    field_pos = fields.Integer()

    @pre_load
    def fix_player(self, data, **kwargs):
        data['player_id'] = data['player_id'][-8:]
        return data