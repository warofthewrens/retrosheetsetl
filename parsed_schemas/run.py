from marshmallow import Schema, fields, validate, pre_dump, post_dump

class Run(Schema):
    run_id = fields.Integer()
    game_id = fields.String()
    year = fields.Integer()
    date = fields.DateTime(format='%Y/%m/%d')
    scoring_team = fields.String()
    conceding_team = fields.String()
    scoring_player = fields.String()
    batter = fields.String()
    responsible_pitcher = fields.String()
    is_earned = fields.Boolean()
    is_team_earned = fields.Boolean()
    is_rbi = fields.Boolean()
    is_sp = fields.Boolean()
    inning = fields.Integer()
    outs = fields.Integer()

    @pre_dump
    def get_year(self, data, **kwargs):
        data['year'] = data['date'].year
        return data
    

