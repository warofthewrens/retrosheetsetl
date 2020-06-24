from marshmallow import Schema, fields, validate, pre_dump, post_dump

class Run(Schema):
    scoring_team = fields.String()
    conceding_team = fields.String()
    scoring_player = fields.String()
    batter = fields.String()
    responsible_pitcher = fields.String()
    is_earned = fields.Boolean()
    is_team_earned = fields.Boolean()
    is_rbi = fields.Boolean()
    inning = fields.Integer()
    outs = fields.Integer()

