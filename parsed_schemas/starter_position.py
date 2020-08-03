from marshmallow import Schema, fields

class StarterPosition(Schema):
    team = fields.String()
    year = fields.Integer()
    starter_1 = fields.String()
    starter_1_WAR = fields.Float()
    starter_2 = fields.String()
    starter_2_WAR = fields.Float()
    starter_3 = fields.String()
    starter_3_WAR = fields.Float()
    starter_4 = fields.String()
    starter_4_WAR = fields.Float()
    starter_5 = fields.String()
    starter_5_WAR = fields.Float()