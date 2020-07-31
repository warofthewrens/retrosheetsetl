from marshmallow import Schema, fields

class Series(Schema):

    series_id = fields.String()
    year = fields.Integer()
    series = fields.String()
    winning_team = fields.String()
    losing_team = fields.String()
    team_with_home_field_advantage = fields.String()