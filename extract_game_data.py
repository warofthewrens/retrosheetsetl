import pandas as pd
import time
from models.sqla_utils import ENGINE, BASE, get_session
from models.player import Player
from extract import extract_roster_team

MODELS = [Player]

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
    run_data_df = pd.read_sql_table(
        'run',
        con=ENGINE
    )
    br_data_df = pd.read_sql_table(
        'base_running_event',
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
        player_dict['S'] = pa_data_df[batter_team_bool & (pa_data_df.hit_val == 1)].hit_val.count()
        player_dict['D'] = pa_data_df[batter_team_bool & (pa_data_df.hit_val == 2)].hit_val.count()
        player_dict['T'] = pa_data_df[batter_team_bool & (pa_data_df.hit_val == 3)].hit_val.count()
        player_dict['HR'] = pa_data_df[batter_team_bool & (pa_data_df.hit_val == 4)].hit_val.count()
        player_dict['TB'] = pa_data_df[batter_team_bool & (pa_data_df.hit_val > 0)].hit_val.sum()
        player_dict['H'] = player_dict['1B'] + player_dict['2B'] + player_dict['3B'] + player_dict['HR']
        player_dict['R'] = run_data_df[run_data_df.scoring_player == player].scoring_player.count()
        player_dict['RBI'] = pa_data_df[batter_team_bool & pa_data_df.rbi > 0].rbi.sum()
        player_dict['SB'] = br_data_df[br_data_df.runner == player & br_data_df.event == 'S'].runner.count()
        player_dict['CS'] = br_data_df[br_data_df.runner == player & br_data_df.event == 'C'].runner.count()
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
        else:
            player_dict['AVG'], player_dict['OBP'], player_dict['SLG'], player_dict['OPS'] = 0
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
        player_dict['SV'] = game_data_df[(player == game_data_df.save) & (team == game_data_df.winning_team)].winning_team.count()
        player_dict['Ra'] = run_data_df[run_data_df.responsible_pitcher == player].responsible_pitcher.count()
        player_dict['ERa'] = run_data_df[run_data_df.responsible_pitcher == player & run_data_df.is_earned].responsible_pitcher.count()
        if player_dict['IP'] > 0:
            player_dict['RA'] = (player_dict['Ra'] / player_dict['IP']) * 9
            player_dict['ERA'] = (player_dict['ERa'] / player_dict['IP']) * 9
        else:
            player_dict['RA'], player_dict['ERA'] = 0
        player_dict['player_id'] = player
        player_dict['team'] = player
        # player_dict = {player: player_dict}
        player_dicts.append(player_dict)
        
        i += 1
    
    for player in player_dicts:
        if player['IP'] > 50:
            print(player)
    return player_dicts

def load(results):
    BASE.metadata.create_all(tables=[x.__table__ for x in MODELS], checkfirst=True)
    session = get_session()
    for model in MODELS:
        data = results[model.__tablename__]
        i = 0
        # Here is where we convert directly the dictionary output of our marshmallow schema into sqlalchemy
        for row in data:
            if i % 1000 == 0:
                print('loading...', i)
            i+=1
            session.merge(model(**row))
start = time.time()
parsed_data = get_game_data()
rows = {table: [] for table in ['player']}
rows['player'].extend(parsed_data)
load(rows)
end = time.time()
print('total time', end - start)