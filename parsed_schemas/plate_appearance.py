from marshmallow import Schema, fields, validate, pre_dump
import re

def expand(data):
    # print(data)
    # play_arr = data['play'].split('/')

    # for item in play_arr:
    #     if 
    return data


class PlateAppearance(Schema):
    game_id = fields.String()
    inning = fields.Integer()
    batting_team_home = fields.Boolean(data_val = 'is_home')
    outs = fields.Integer()
    balls = fields.Integer()
    strikes = fields.Integer()
    sequence = fields.String(data_val = 'pitches')
    away_runs = fields.Integer()
    home_runs = fields.Integer()
    batter_id = fields.String()
    # batter_hand = fields.String()
    pitcher_id = fields.String()
    # pitcher_hand = fields.String()
    # positions
    first_runner_id = fields.String()
    second_runner_id = fields.String()
    third_runner_id = fields.String()
    field_pos = fields.Integer()
    lineup_pos = fields.Integer()
    event_type = fields.Integer()
    ab_flag = fields.Boolean()
    hit_val = fields.Integer(validate = validate.Range(1, 4))
    sac_bunt = fields.Boolean()
    sac_fly = fields.Boolean()
    outs_on_play = fields.Integer()
    rbi = fields.Integer()
    wp = fields.Boolean()
    pb = fields.Boolean()
    fielder_id = fields.String()
    ball_type = fields.String()
    bunt_flag = fields.Boolean()
    foul_flag = fields.Boolean()
    hit_loc = fields.Integer()
    error_player_id = fields.String()
    batter_dest = fields.String()
    first_dest = fields.String()
    second_dest = fields.String()
    third_dest = fields.String()
    first_po = fields.String()
    second_po = fields.String()
    first_ast = fields.String()
    second_ast = fields.String()

    @pre_dump
    def expand_play_str(self, data, **kwargs):
        data = expand(data)
        return data

    @pre_dump
    def count(self, data, **kwargs):
        data['balls'] = int(data['count'][0])
        data['strikes'] = int(data['count'][1])
        return data