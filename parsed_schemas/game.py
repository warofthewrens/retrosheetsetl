from marshmallow import Schema, fields, validate, pre_dump, post_dump

pos_dict = {
    1: 'pitcher',
    2: 'catcher',
    3: 'first',
    4: 'second',
    5: 'third',
    6: 'short',
    7: 'left',
    8: 'center',
    9: 'right',
    10: 'dh'
}

class Game(Schema):
    game_id = fields.String()
    series_id = fields.String()
    year = fields.Integer()
    date = fields.DateTime()
    away_team = fields.String()
    away_team_runs = fields.Integer()
    home_team = fields.String()
    home_team_runs = fields.Integer()
    winning_team = fields.String()
    losing_team = fields.String()
    innings = fields.Integer()
    site = fields.String()
    start_time = fields.String()
    is_day = fields.Boolean()
    temp = fields.Integer()
    wind_dir = fields.String()
    wind_speed = fields.Integer()
    field_cond = fields.String()
    precip = fields.String()
    sky = fields.String()
    time_of_game = fields.Integer()
    attendance = fields.Integer()
    winning_pitcher = fields.String()
    losing_pitcher = fields.String()
    save = fields.String()
    starting_pitcher_home = fields.String()
    starting_catcher_home = fields.String()
    starting_first_home = fields.String()
    starting_second_home = fields.String()
    starting_third_home = fields.String()
    starting_short_home = fields.String()
    starting_left_home = fields.String()
    starting_center_home = fields.String()
    starting_right_home = fields.String()
    starting_pitcher_away = fields.String()
    starting_catcher_away = fields.String()
    starting_first_away = fields.String()
    starting_second_away = fields.String()
    starting_third_away = fields.String()
    starting_short_away = fields.String()
    starting_left_away = fields.String()
    starting_center_away = fields.String()
    starting_right_away = fields.String()
    starting_dh_home = fields.String()
    starting_dh_away = fields.String()
    home_team_er = fields.Integer()
    away_team_er = fields.Integer()
    starting_pitcher_home_er = fields.Integer()
    starting_pitcher_home_r = fields.Integer()
    starting_pitcher_away_er = fields.Integer()
    starting_pitcher_away_r = fields.Integer()
    relief_pitcher1 = fields.String()
    relief_pitcher1_er = fields.Integer()
    relief_pitcher1_r = fields.Integer()
    relief_pitcher2 = fields.String()
    relief_pitcher2_er = fields.Integer()
    relief_pitcher2_r = fields.Integer()
    relief_pitcher3 = fields.String()
    relief_pitcher3_er = fields.Integer()
    relief_pitcher3_r = fields.Integer()
    relief_pitcher4 = fields.String()
    relief_pitcher4_er = fields.Integer()
    relief_pitcher4_r = fields.Integer()
    relief_pitcher5 = fields.String()
    relief_pitcher5_er = fields.Integer()
    relief_pitcher5_r = fields.Integer()
    relief_pitcher6 = fields.String()
    relief_pitcher6_er = fields.Integer()
    relief_pitcher6_r = fields.Integer()
    relief_pitcher7 = fields.String()
    relief_pitcher7_er = fields.Integer()
    relief_pitcher7_r = fields.Integer()
    relief_pitcher8 = fields.String()
    relief_pitcher8_er = fields.Integer()
    relief_pitcher8_r = fields.Integer()
    relief_pitcher9 = fields.String()
    relief_pitcher9_er = fields.Integer()
    relief_pitcher9_r = fields.Integer()
    relief_pitcher10 = fields.String()
    relief_pitcher10_er = fields.Integer()
    relief_pitcher10_r = fields.Integer()
    relief_pitcher11 = fields.String()
    relief_pitcher11_er = fields.Integer()
    relief_pitcher11_r = fields.Integer()
    relief_pitcher12 = fields.String()
    relief_pitcher12_er = fields.Integer()
    relief_pitcher12_r = fields.Integer()
    relief_pitcher13 = fields.String()
    relief_pitcher13_er = fields.Integer()
    relief_pitcher13_r = fields.Integer()
    relief_pitcher14 = fields.String()
    relief_pitcher14_er = fields.Integer()
    relief_pitcher14_r = fields.Integer()
    relief_pitcher15 = fields.String()
    relief_pitcher15_er = fields.Integer()
    relief_pitcher15_r = fields.Integer()
    relief_pitcher16 = fields.String()
    relief_pitcher16_er = fields.Integer()
    relief_pitcher16_r = fields.Integer()
    relief_pitcher17 = fields.String()
    relief_pitcher17_er = fields.Integer()
    relief_pitcher17_r = fields.Integer()
    relief_pitcher18 = fields.String()
    relief_pitcher18_er = fields.Integer()
    relief_pitcher18_r = fields.Integer()
    relief_pitcher19 = fields.String()
    relief_pitcher19_er = fields.Integer()
    relief_pitcher19_r = fields.Integer()
    


    @pre_dump
    def handle_data(self, game, **kwargs):
        #print(self.context)
        
        for player in game['lineup']:
            if player['is_home']:
                game['starting_' + pos_dict[player['field_pos']]+ '_home'] = player['player_id']
            else:
                game['starting_' + pos_dict[player['field_pos']]+ '_away'] = player['player_id']
        game.pop('lineups', None)
        i = 1
        for pitcher in game['data']:
            if pitcher['pitcher_id'] == game['starting_pitcher_home']:
                field = 'starting_pitcher_home'
            elif pitcher['pitcher_id'] == game['starting_pitcher_away']:
                field = 'starting_pitcher_away'
            else:
                field = 'relief_pitcher' + str(i)
                i+=1
            game[field] = pitcher['pitcher_id']
            game[field + '_er'] = pitcher['data']
            game[field + '_r'] = self.context['runs_allowed'][pitcher['pitcher_id']]
            if i > 19:
                print("WEEWOOWEEWOO")
        game.pop('data', None)

        game['sky'] = game['info']['sky']
        game['precip'] = game['info']['precip']
        game['attendance'] = game['info']['attendance']
        game['temp'] = game['info']['temp']
        game['date'] = game['info']['date']
        game['start_time'] = game['info']['start_time']
        game['time_of_game'] = game['info']['time_of_game']
        game['field_cond'] = game['info']['field_cond']
        game['winning_pitcher'] = game['info']['winning_pitcher']
        game['losing_pitcher'] = game['info']['losing_pitcher']
        game['wind_dir'] = game['info']['wind_dir']
        game['site'] = game['info']['site']
        game['home_team'] = game['info']['home_team']
        game['away_team'] = game['info']['visiting_team']
        game['is_day'] = game['info']['daynight']
        game['wind_speed'] = game['info']['wind_speed']
        game['save'] = game['info']['save']
        game.pop('info', None)

        game['home_team_runs'] = self.context['home_runs']
        game['away_team_runs'] = self.context['away_runs']
        if game['home_team_runs'] > game['away_team_runs']:
            game['winning_team'] = game['home_team']
            game['losing_team'] = game['away_team']
        elif game['home_team_runs'] == game['away_team_runs']:
            game['winning_team'] = 'TIE'
            game['losing_team'] = 'TIE'
        else:
            game['winning_team'] = game['away_team']
            game['losing_team'] = game['home_team']
        game['innings'] = self.context['inning']
        game['year'] = game['date'].year
        return game
    