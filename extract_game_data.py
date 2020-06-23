import pandas as pd
import time
from models.sqla_utils import ENGINE
from extract import extract_roster_team

team_set = set(['ANA', 'ARI', 'ATL', 'BAL', 'BOS', 'CHA', 'CHN', 'CIN', 'CLE', 'COL', 'DET', 'HOU',
            'KCA', 'LAN', 'MIA', 'MIL', 'MIN', 'NYA', 'NYN', 'OAK', 'PHI', 'PIT', 'SEA',
            'SFN', 'SLN', 'SDN', 'TBA', 'TEX', 'TOR', 'WAS'])

rosters = {}

def get_roster_team():
    for team in team_set:
        rosters.update(extract_roster_team('2019' + team))
    return rosters

def get_game_data():
    pa_data_df = pd.read_sql_table(
        'plateappearance',
        con=ENGINE
    )
    game_data_df = pd.read_sql_table(
        'game',
        con=ENGINE
    )
    rosters = get_roster_team()
    players = rosters.keys()
    player_dicts = []
    i = 0
    for player,team in players:
        if i % 50 == 0:
            print(player)
        player_dict = {}
        batter_team_bool = (pa_data_df.batter_id == player) & (pa_data_df.batter_team == team) 
        pitcher_team_bool = (pa_data_df.pitcher_id == player) & (pa_data_df.pitcher_team == team) 
        pitcher_runs =  game_data_df[player == game_data_df.starting_pitcher_home].starting_pitcher_home_r.sum() + \
                        game_data_df[player == game_data_df.starting_pitcher_away].starting_pitcher_away_r.sum() + \
                        game_data_df[player == game_data_df.relief_pitcher1].relief_pitcher1_r.sum() + \
                        game_data_df[player == game_data_df.relief_pitcher2].relief_pitcher2_r.sum() + \
                        game_data_df[player == game_data_df.relief_pitcher3].relief_pitcher3_r.sum() + \
                        game_data_df[player == game_data_df.relief_pitcher4].relief_pitcher4_r.sum() + \
                        game_data_df[player == game_data_df.relief_pitcher5].relief_pitcher5_r.sum() + \
                        game_data_df[player == game_data_df.relief_pitcher6].relief_pitcher6_r.sum() + \
                        game_data_df[player == game_data_df.relief_pitcher7].relief_pitcher7_r.sum() + \
                        game_data_df[player == game_data_df.relief_pitcher8].relief_pitcher8_r.sum() + \
                        game_data_df[player == game_data_df.relief_pitcher9].relief_pitcher9_r.sum() + \
                        game_data_df[player == game_data_df.relief_pitcher10].relief_pitcher10_r.sum() + \
                        game_data_df[player == game_data_df.relief_pitcher11].relief_pitcher11_r.sum() + \
                        game_data_df[player == game_data_df.relief_pitcher12].relief_pitcher12_r.sum() + \
                        game_data_df[player == game_data_df.relief_pitcher13].relief_pitcher13_r.sum() + \
                        game_data_df[player == game_data_df.relief_pitcher14].relief_pitcher14_r.sum() + \
                        game_data_df[player == game_data_df.relief_pitcher15].relief_pitcher15_r.sum() + \
                        game_data_df[player == game_data_df.relief_pitcher16].relief_pitcher16_r.sum() + \
                        game_data_df[player == game_data_df.relief_pitcher17].relief_pitcher17_r.sum() + \
                        game_data_df[player == game_data_df.relief_pitcher18].relief_pitcher18_r.sum() + \
                        game_data_df[player == game_data_df.relief_pitcher19].relief_pitcher19_r.sum()
        pitcher_earned_runs =  game_data_df[player == game_data_df.starting_pitcher_home].starting_pitcher_home_er.sum() + \
                        game_data_df[player == game_data_df.starting_pitcher_away].starting_pitcher_away_er.sum() + \
                        game_data_df[player == game_data_df.relief_pitcher1].relief_pitcher1_er.sum() + \
                        game_data_df[player == game_data_df.relief_pitcher2].relief_pitcher2_er.sum() + \
                        game_data_df[player == game_data_df.relief_pitcher3].relief_pitcher3_er.sum() + \
                        game_data_df[player == game_data_df.relief_pitcher4].relief_pitcher4_er.sum() + \
                        game_data_df[player == game_data_df.relief_pitcher5].relief_pitcher5_er.sum() + \
                        game_data_df[player == game_data_df.relief_pitcher6].relief_pitcher6_er.sum() + \
                        game_data_df[player == game_data_df.relief_pitcher7].relief_pitcher7_er.sum() + \
                        game_data_df[player == game_data_df.relief_pitcher8].relief_pitcher8_er.sum() + \
                        game_data_df[player == game_data_df.relief_pitcher9].relief_pitcher9_er.sum() + \
                        game_data_df[player == game_data_df.relief_pitcher10].relief_pitcher10_er.sum() + \
                        game_data_df[player == game_data_df.relief_pitcher11].relief_pitcher11_er.sum() + \
                        game_data_df[player == game_data_df.relief_pitcher12].relief_pitcher12_er.sum() + \
                        game_data_df[player == game_data_df.relief_pitcher13].relief_pitcher13_er.sum() + \
                        game_data_df[player == game_data_df.relief_pitcher14].relief_pitcher14_er.sum() + \
                        game_data_df[player == game_data_df.relief_pitcher15].relief_pitcher15_er.sum() + \
                        game_data_df[player == game_data_df.relief_pitcher16].relief_pitcher16_er.sum() + \
                        game_data_df[player == game_data_df.relief_pitcher17].relief_pitcher17_er.sum() + \
                        game_data_df[player == game_data_df.relief_pitcher18].relief_pitcher18_er.sum() + \
                        game_data_df[player == game_data_df.relief_pitcher19].relief_pitcher19_er.sum()
        player_dict['PA'] = pa_data_df[batter_team_bool & (pa_data_df.pa_flag)].pa_flag.count()
        player_dict['AB'] = pa_data_df[batter_team_bool & (pa_data_df.ab_flag)].ab_flag.count()
        player_dict['1B'] = pa_data_df[batter_team_bool & (pa_data_df.hit_val == 1)].hit_val.count()
        player_dict['2B'] = pa_data_df[batter_team_bool & (pa_data_df.hit_val == 2)].hit_val.count()
        player_dict['3B'] = pa_data_df[batter_team_bool & (pa_data_df.hit_val == 3)].hit_val.count()
        player_dict['HR'] = pa_data_df[batter_team_bool & (pa_data_df.hit_val == 4)].hit_val.count()
        player_dict['TB'] = pa_data_df[batter_team_bool & (pa_data_df.hit_val > 0)].hit_val.sum()
        player_dict['H'] = player_dict['1B'] + player_dict['2B'] + player_dict['3B'] + player_dict['HR']
        # player_dict['R'] = pa_data_df[(pa_data_df.first_runner_id == player & pa_data_df.batter_team == team & pa_data_df.first_runner_dest == 'H')
        #                                 | pa_data_df.second_runner_id == player & pa_data_df.batter_team == team & pa_data_df.second_runner_dest == 'H'
        #                                 | pa_data_df.third_runner_id == player & pa_data_df.batter_team == team & pa_data_df.third_runner_dest == 'H'].count()
        player_dict['RBI'] = pa_data_df[batter_team_bool & pa_data_df.rbi > 0].rbi.sum()
        # player_dict['SB'] = pa_data_df[(pa_data_df.first_runner_id == player & pa_data_df.batter_team == team & pa_data_df.first_runner_event == 'S')
        #                                 | pa_data_df.second_runner_id == player & pa_data_df.batter_team == team & pa_data_df.second_runner_event == 'S'
        #                                 | pa_data_df.third_runner_id == player & pa_data_df.batter_team == team & pa_data_df.third_runner_event == 'S'].count()
        # player_dict['CS'] = pa_data_df[(pa_data_df.first_runner_id == player & pa_data_df.batter_team == team & pa_data_df.first_runner_event == 'C')
        #                                 | pa_data_df.second_runner_id == player & pa_data_df.batter_team == team & pa_data_df.second_runner_event == 'C'
        #                                 | pa_data_df.third_runner_id == player & pa_data_df.batter_team == team & pa_data_df.third_runner_event == 'C'].count()
        player_dict['BB'] = pa_data_df[batter_team_bool & ((pa_data_df.event_type == 14) | (pa_data_df.event_type == 15))].pa_flag.count()
        player_dict['SO'] = pa_data_df[batter_team_bool & (pa_data_df.event_type == 3)].pa_flag.count()
        player_dict['HBP'] = pa_data_df[batter_team_bool & (pa_data_df.event_type == 16)].pa_flag.count()
        player_dict['SF'] = pa_data_df[batter_team_bool & pa_data_df.sac_fly].pa_flag.count()
        player_dict['SH'] = pa_data_df[batter_team_bool & pa_data_df.sac_bunt].pa_flag.count()
        if player_dict['AB'] > 0:
            player_dict['AVG'] = player_dict['H'] / player_dict['AB']
            player_dict['OBP'] = (player_dict['H'] + player_dict['BB'] + player_dict['HBP']) / (player_dict['AB'] + player_dict['BB'] + player_dict['HBP'] + player_dict['SF'])
            player_dict['SLG'] = player_dict['TB']/player_dict['AB']
            player_dict['OPS'] = player_dict['OBP'] + player_dict['SLG']
        player_dict['BF'] = pa_data_df[pitcher_team_bool & pa_data_df.pa_flag].pa_flag.count()
        player_dict['IP'] = pa_data_df[pitcher_team_bool].outs_on_play.sum()/3
        player_dict['Ha'] = pa_data_df[pitcher_team_bool & (pa_data_df.hit_val>0)].hit_val.count()
        player_dict['HRa'] = pa_data_df[pitcher_team_bool & (pa_data_df.hit_val == 4)].hit_val.count()
        player_dict['TBa'] = pa_data_df[pitcher_team_bool & (pa_data_df.hit_val>0)].hit_val.sum()
        player_dict['BBa'] = pa_data_df[pitcher_team_bool & ((pa_data_df.event_type == 14) | (pa_data_df.event_type == 15))].event_type.count()
        player_dict['IBBa'] = pa_data_df[pitcher_team_bool & (pa_data_df.event_type == 15)].event_type.count()
        player_dict['K'] = pa_data_df[pitcher_team_bool & (pa_data_df.event_type == 3)].event_type.count()
        player_dict['HBPa'] = pa_data_df[pitcher_team_bool & (pa_data_df.event_type == 16)].event_type.count()
        player_dict['BK'] = pa_data_df[pitcher_team_bool & (pa_data_df.event_type == 11)].event_type.count()
        player_dict['W'] = game_data_df[(player == game_data_df.winning_pitcher) & (team == game_data_df.winning_team)].winning_team.count()
        player_dict['L'] = game_data_df[(player == game_data_df.losing_pitcher) & (team == game_data_df.losing_team)].losing_team.count()
        player_dict['S'] = game_data_df[(player == game_data_df.save) & (team == game_data_df.winning_team)].winning_team.count()
        player_dict['Ra'] = pitcher_runs
        player_dict['ERa'] = pitcher_earned_runs
        if player_dict['IP'] > 0:
            player_dict['RA'] = (player_dict['Ra'] / player_dict['IP']) * 9
            player_dict['ERA'] = (player_dict['ERa'] / player_dict['IP']) * 9
        player_dict['player_id'] = player
        player_dict['team'] = player
        # player_dict = {player: player_dict}
        player_dicts.append(player_dict)
        
        i += 1
    for player in player_dicts:
        if player['IP'] > 50:
            print(player)
start = time.time()
get_game_data()
end = time.time()
print('total time', end - start)