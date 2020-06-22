from marshmallow import Schema, fields, validate, pre_dump, post_dump
import re

runner_dest = {'B' : 'batter_dest',
               '1' : 'first_dest',
               '2' : 'second_dest',
               '3' : 'third_dest'}

pos_dict = {
    2: 'catcher',
    3: 'first_base',
    4: 'second_base',
    5: 'third_base',
    6: 'shortstop',
    7: 'left_field',
    8: 'center_field',
    9: 'right_field'
}

class PlateAppearance(Schema):
    pa_id = fields.Integer()
    play = fields.String()
    game_id = fields.String()
    date = fields.DateTime(format='%Y/%m/%d')
    batter_id = fields.String()
    batter_team = fields.String()
    batter_hand = fields.String()
    pitcher_id = fields.String()
    pitcher_team = fields.String()
    pitcher_hand = fields.String()
    inning = fields.Integer()
    is_home = fields.Boolean(data_key = 'batting_team_home')
    outs = fields.Integer()
    balls = fields.Integer()
    strikes = fields.Integer()
    pitches = fields.String(data_key='sequence')
    away_runs = fields.Integer()
    home_runs = fields.Integer()
    first_runner_id = fields.String()
    second_runner_id = fields.String()
    third_runner_id = fields.String()
    field_pos = fields.Integer()
    lineup_pos = fields.Integer()
    event_type = fields.Integer()
    ab_flag = fields.Boolean()
    pa_flag = fields.Boolean()
    sp_flag = fields.Boolean()
    hit_val = fields.Integer()
    sac_bunt = fields.Boolean()
    sac_fly = fields.Boolean()
    outs_on_play = fields.Integer()
    rbi = fields.Integer()
    runs_on_play = fields.Integer()
    first_scorer = fields.String()
    second_scorer = fields.String()
    third_scorer = fields.String()
    fourth_scorer = fields.String()
    first_runner_event = fields.String()
    second_runner_event = fields.String()
    third_runner_event = fields.String()
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
    catcher = fields.String()
    first_base = fields.String()
    second_base = fields.String()
    third_base = fields.String()
    shortstop = fields.String()
    left_field = fields.String()
    center_field = fields.String()
    right_field = fields.String()

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
        if data['is_home']:
            data['sp_flag'] = self.context['lineups']['away_field_pos']['sp'] == data['pitcher_id']
            data['batter_team'] = self.context['home_team']
            data['pitcher_team'] = self.context['away_team']
        else:
            data['sp_flag'] = self.context['lineups']['home_field_pos']['sp'] == data['pitcher_id']
            data['batter_team'] = self.context['away_team']
            data['pitcher_team'] = self.context['home_team']
        for pos in range(2, 10):
            if data['is_home']:
                data[pos_dict[pos]] = self.context['lineups']['away_field_pos'][pos]
            else:
                data[pos_dict[pos]] = self.context['lineups']['home_field_pos'][pos]
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
            self.context['responsible_pitchers'][data['third_dest']] = self.context['responsible_pitchers']['3']
        if data['third_dest'] in set(['O', 'H']):
            self.context['runners_before'][3] = ''
            self.context['responsible_pitchers']['3'] = ''
        if data['second_dest'] in set(['2', '3']):
            self.context['runners_before'][int(data['second_dest'])] = self.context['runners_before'][2]
            self.context['responsible_pitchers'][data['second_dest']] = self.context['responsible_pitchers']['2']
        if data['second_dest'] in set(['3', 'O', 'H']):
            self.context['runners_before'][2] = ''
            self.context['responsible_pitchers']['2'] = ''
        if data['first_dest'] in set(['1', '2', '3']):
            self.context['runners_before'][int(data['first_dest'])] = self.context['runners_before'][1]
            self.context['responsible_pitchers'][data['first_dest']] = self.context['responsible_pitchers']['1']
        if data['first_dest'] in set(['2', '3', 'O', 'H']):
            self.context['runners_before'][1] = ''
            self.context['responsible_pitchers']['1'] = ''
        if data['batter_dest'] in set(['1', '2', '3']):
            self.context['runners_before'][int(data['batter_dest'])] = data['batter_id']
            self.context['responsible_pitchers'][data['batter_dest']] = self.context['responsible_pitchers']['B']

        if data['batting_team_home']:
            self.context['home_runs'] += data['runs_on_play']
        else:
            self.context['away_runs'] += data['runs_on_play']

        if self.context['outs'] == 3:
            self.context['runners_before'][1] = ''
            self.context['runners_before'][2] = ''
            self.context['runners_before'][3] = ''
            self.context['outs'] = 0
            self.context['responsible_pitchers']['1'] = ''
            self.context['responsible_pitchers']['2'] = ''
            self.context['responsible_pitchers']['3'] = ''
            self.context['responsible_pitchers']['B'] = ''
        self.context['po'] = 0
        self.context['ast'] = 0
        return data
        
        
        
