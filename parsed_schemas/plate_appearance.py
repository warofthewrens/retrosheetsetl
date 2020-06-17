from marshmallow import Schema, fields, validate, pre_dump, post_dump
import re

runner_dest = {'B' : 'batter_dest',
               '1' : 'first_dest',
               '2' : 'second_dest',
               '3' : 'third_dest'}


class PlateAppearance(Schema):
    pa_id = fields.Integer()
    play = fields.String()
    game_id = fields.String()
    inning = fields.Integer()
    is_home = fields.Boolean(data_key = 'batting_team_home')
    outs = fields.Integer()
    balls = fields.Integer()
    strikes = fields.Integer()
    pitches = fields.String(data_key='sequence')
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
    hit_val = fields.Integer()
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
    first_error = fields.String()
    second_error = fields.String()
    third_error = fields.String()
    num_errors = fields.Integer()
    batter_dest = fields.String()
    first_dest = fields.String()
    second_dest = fields.String()
    third_dest = fields.String()
    first_po = fields.String()
    second_po = fields.String()
    third_po = fields.String()
    first_ast = fields.String()
    second_ast = fields.String()
    third_ast = fields.String()
    fourth_ast = fields.String()
    fifth_ast = fields.String()

    @pre_dump
    def expand_play_str(self, data, **kwargs):
        data['game_id'] = self.context['game_id']['game_id']
        data['date'] = self.context['date']
        data['outs'] = self.context['outs']
        data['home_runs'] = self.context['home_runs']
        data['away_runs'] = self.context['away_runs']
        data['first_runner_id'] = self.context['runners_before'][1]
        data['second_runner_id'] = self.context['runners_before'][2]
        data['third_runner_id'] = self.context['runners_before'][3]
        return data

    @pre_dump
    def count(self, data, **kwargs):
        data['balls'] = int(data['count'][0])
        data['strikes'] = int(data['count'][1])
        return data
    
    @post_dump
    def update_state(self, data, **kwargs):
        self.context['outs'] += data['outs_on_play']
        if data['third_dest'] in set(['3']):
            self.context['runners_before'][int(data['third_dest'])] = self.context['runners_before'][3]
        if data['third_dest'] in set(['O', 'H']):
            self.context['runners_before'][3] = ''
        if data['second_dest'] in set(['2', '3']):
            self.context['runners_before'][int(data['second_dest'])] = self.context['runners_before'][2]
        if data['second_dest'] in set(['3', 'O', 'H']):
            self.context['runners_before'][2] = ''
        if data['first_dest'] in set(['1', '2', '3']):
            self.context['runners_before'][int(data['first_dest'])] = self.context['runners_before'][1]
        if data['first_dest'] in set(['2', '3', 'O', 'H']):
            self.context['runners_before'][1] = ''
        if data['batter_dest'] in set(['1', '2', '3']):
            self.context['runners_before'][int(data['batter_dest'])] = data['batter_id']

        

        if self.context['outs'] == 3:
            self.context['runners_before'][1] = ''
            self.context['runners_before'][2] = ''
            self.context['runners_before'][3] = ''
            self.context['outs'] = 0
        
        self.context['po'] = 0
        self.context['ast'] = 0
        return data
        
        
        
