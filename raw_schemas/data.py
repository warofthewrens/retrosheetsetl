from marshmallow import Schema, fields

class Data(Schema):
    data_type = fields.String(data_key='type')
    pitcher_id = fields.String()
    data = fields.Integer()