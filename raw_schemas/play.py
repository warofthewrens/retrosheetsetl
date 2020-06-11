from marshmallow import Schema, fields, validate

def validate_count(count):
    # print(count)
    return
    

def validate_pitches(pitches):
    # print(pitches)
    return

def validate_play(play):
    # print(play)
    return

class Play(Schema):
    inning = fields.Integer(validate=validate.Range(1, 30))
    is_home = fields.Boolean()
    batter_id = fields.String()
    count = fields.String(validate=validate_count)
    pitches = fields.String(validate=validate_pitches)
    play = fields.String(validate=validate_play)
