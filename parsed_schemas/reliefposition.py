from marshmallow import Schema, fields

class ReliefPosition(Schema):
    team = fields.String()
    year = fields.Integer()
    closer = fields.String()
    closer_WAR = fields.Float()
    relief_1 = fields.String()
    relief_1_WAR = fields.Float()
    relief_2 = fields.String()
    relief_2_WAR = fields.Float()
    relief_3 = fields.String()
    relief_3_WAR = fields.Float()
    relief_4 = fields.String()
    relief_4_WAR = fields.Float()

