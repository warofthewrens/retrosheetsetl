from marshmallow import Schema, fields

class League(Schema):

    year = fields.Integer()
    H = fields.Integer()
    HR = fields.Integer()
    BB = fields.Integer()
    K = fields.Integer()
    HBP = fields.Integer()
    IFFB = fields.Integer()
    IP = fields.Integer()
    ER = fields.Integer()
    TR = fields.Integer()
    SpER = fields.Integer()
    SpTR = fields.Integer()
    SpIP = fields.Integer()
    RpER = fields.Integer()
    RpTR = fields.Integer()
    RpIP = fields.Integer()
    lgRAA = fields.Float()
    lgERA = fields.Float()
    lgFIP = fields.Float()
    lgiFIP = fields.Float()
    ciFIP = fields.Float()
    WARadj = fields.Float()
    lgSpERA = fields.Float()
    lgSpRAA = fields.Float()
    lgSpFIP = fields.Float()
    lgRpERA = fields.Float()
    lgRpRAA = fields.Float()
    lgRpFIP = fields.Float()
    

