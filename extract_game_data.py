import pandas as pd
import pymysql
from os import walk
import time
import shutil
from models.sqla_utils import ENGINE, BASE, get_session
from models.player import Player
from parsed_schemas.player import Player as p
from extract import extract_roster_team, extract_game_data_by_year
import concurrent.futures
import multiprocessing
import pickle

MODELS = [Player]

team_set = set(['ANA', 'ARI', 'ATL', 'BAL', 'BOS', 'CHA', 'CHN', 'CIN', 'CLE', 'COL', 'DET', 'HOU',
            'KCA', 'LAN', 'MIA', 'MIL', 'MIN', 'NYA', 'NYN', 'OAK', 'PHI', 'PIT', 'SEA',
            'SFN', 'SLN', 'SDN', 'TBA', 'TEX', 'TOR', 'WAS'])

rosters = {}

#TODO: return dfs to correct place here


# pa_data_df = pd.DataFrame
# game_data_df = pd.DataFrame
# run_data_df = pd.DataFrame
# br_data_df = pd.DataFrame

def get_rosters(year, data_zip):
    for team in team_set:
        rosters.update(extract_roster_team(year + team, data_zip))
    return rosters

def get_player_data(player, team, year, pa_data_df, game_data_df, run_data_df, br_data_df):
    
    player_dict = {}
    pa_year = pa_data_df.year == int(year)
    game_year = game_data_df.year == int(year)
    run_year = run_data_df.year == int(year)
    br_year = (br_data_df.year == int(year)) & (br_data_df.running_team == team)
    team_scored = run_data_df.scoring_team == team
    team_scored_against = run_data_df.conceding_team == team
    batter_team_bool = (pa_data_df.batter_id == player) & (pa_data_df.batter_team == team) 
    pitcher_team_bool = (pa_data_df.pitcher_id == player) & (pa_data_df.pitcher_team == team) 
    player_dict['PA'] = pa_data_df[batter_team_bool & (pa_data_df.pa_flag) & pa_year].pa_flag.count()
    player_dict['AB'] = pa_data_df[batter_team_bool & (pa_data_df.ab_flag) & pa_year].ab_flag.count()
    player_dict['S'] = pa_data_df[batter_team_bool & (pa_data_df.hit_val == 1) & pa_year].hit_val.count()
    player_dict['D'] = pa_data_df[batter_team_bool & (pa_data_df.hit_val == 2) & pa_year].hit_val.count()
    player_dict['T'] = pa_data_df[batter_team_bool & (pa_data_df.hit_val == 3) & pa_year].hit_val.count()
    player_dict['HR'] = pa_data_df[batter_team_bool & (pa_data_df.hit_val == 4) & pa_year].hit_val.count()
    player_dict['TB'] = pa_data_df[batter_team_bool & (pa_data_df.hit_val > 0) & pa_year].hit_val.sum()
    player_dict['H'] = player_dict['S'] + player_dict['D'] + player_dict['T'] + player_dict['HR']
    player_dict['R'] = run_data_df[(run_data_df.scoring_player == player) & run_year & team_scored].scoring_player.count()
    player_dict['RBI'] = pa_data_df[batter_team_bool & (pa_data_df.rbi > 0) & pa_year].rbi.sum()
    player_dict['SB'] = br_data_df[(br_data_df.runner == player) & (br_data_df.event == 'S') & br_year].runner.count()
    player_dict['CS'] = br_data_df[(br_data_df.runner == player) & (br_data_df.event == 'C') & br_year].runner.count()
    player_dict['BB'] = pa_data_df[batter_team_bool & ((pa_data_df.event_type == 14) | (pa_data_df.event_type == 15)) & pa_year].pa_flag.count()
    player_dict['SO'] = pa_data_df[batter_team_bool & (pa_data_df.event_type == 3) & pa_year].pa_flag.count()
    player_dict['HBP'] = pa_data_df[batter_team_bool & (pa_data_df.event_type == 16) & pa_year].pa_flag.count()
    player_dict['SF'] = pa_data_df[batter_team_bool & pa_data_df.sac_fly & pa_year].pa_flag.count()
    player_dict['SH'] = pa_data_df[batter_team_bool & pa_data_df.sac_bunt & pa_year].pa_flag.count()
    if player_dict['AB'] > 0:
        player_dict['AVG'] = player_dict['H'] / player_dict['AB'] + 0.0
        player_dict['OBP'] = (player_dict['H'] + player_dict['BB'] + player_dict['HBP']) / (player_dict['AB'] + player_dict['BB'] + player_dict['HBP'] + player_dict['SF'])
        player_dict['SLG'] = player_dict['TB']/player_dict['AB']
        player_dict['OPS'] = player_dict['OBP'] + player_dict['SLG']
    else:
        player_dict['AVG'], player_dict['OBP'], player_dict['SLG'], player_dict['OPS'] = 0, 0, 0, 0
    player_dict['BF'] = pa_data_df[pitcher_team_bool & pa_data_df.pa_flag & pa_year].pa_flag.count()
    player_dict['IP'] = float((pa_data_df[pitcher_team_bool & (pa_data_df.outs_on_play > 0) & pa_year].outs_on_play.sum() + 0.0)/3.0)
    player_dict['Ha'] = pa_data_df[pitcher_team_bool & (pa_data_df.hit_val>0) & pa_year].hit_val.count()
    player_dict['HRa'] = pa_data_df[pitcher_team_bool & (pa_data_df.hit_val == 4) & pa_year].hit_val.count()
    player_dict['TBa'] = pa_data_df[pitcher_team_bool & (pa_data_df.hit_val>0) & pa_year].hit_val.sum()
    player_dict['BBa'] = pa_data_df[pitcher_team_bool & ((pa_data_df.event_type == 14) | (pa_data_df.event_type == 15)) & pa_year].event_type.count()
    player_dict['IBBa'] = pa_data_df[pitcher_team_bool & (pa_data_df.event_type == 15) & pa_year].event_type.count()
    player_dict['K'] = pa_data_df[pitcher_team_bool & (pa_data_df.event_type == 3) & pa_year].event_type.count()
    player_dict['HBPa'] = pa_data_df[pitcher_team_bool & (pa_data_df.event_type == 16) & pa_year].event_type.count()
    player_dict['BK'] = pa_data_df[pitcher_team_bool & (pa_data_df.event_type == 11) & pa_year].event_type.count()
    player_dict['W'] = game_data_df[(player == game_data_df.winning_pitcher) & (team == game_data_df.winning_team) & game_year].winning_team.count()
    player_dict['L'] = game_data_df[(player == game_data_df.losing_pitcher) & (team == game_data_df.losing_team) & game_year].losing_team.count()
    player_dict['SV'] = game_data_df[(player == game_data_df.save) & (team == game_data_df.winning_team) & game_year].winning_team.count()
    player_dict['TR'] = run_data_df[(run_data_df.responsible_pitcher == player) & team_scored_against & run_year].responsible_pitcher.count()
    player_dict['ER'] = run_data_df[(run_data_df.responsible_pitcher == player) & team_scored_against & run_data_df.is_earned & run_year].responsible_pitcher.count()
    if player_dict['IP'] > 0:
        player_dict['RA'] = (player_dict['TR'] / player_dict['IP']) * 9
        player_dict['ERA'] = (player_dict['ER'] / player_dict['IP']) * 9
        player_dict['FIP'] = (13 * player_dict['HR'] + 3 * player_dict['BB'] - 2 * player_dict['K'])
    else:
        player_dict['RA'], player_dict['ERA'] = 0, 0
    player_dict['player_id'] = player
    player_dict['team'] = team
    player_dict['year'] = year
    player_dict['player_name'] = rosters[(player, team)]['player_first_name'] + ' ' + rosters[(player, team)]['player_last_name']
    return player_dict

def get_game_data(year, pa_data_df, game_data_df, run_data_df, br_data_df):
    roster_files = set([])
    
    data_zip, data_td = extract_game_data_by_year(year)
    
    f = []
    for (dirpath, dirnames, filenames) in walk(data_td):
        f.extend(filenames)
        break
    shutil.rmtree(data_td)
    for team_file in f:
        if team_file[-4:] == '.ROS':
            roster_files.add(team_file)
    
    for team in roster_files:
        rosters.update(extract_roster_team(team, data_zip))

    players = rosters.keys()
    player_dicts = []
    i = 0
    for player, team in players:
        player_dict = get_player_data(player, team, year, pa_data_df, game_data_df, run_data_df, br_data_df)
        new_player = p().dump(player_dict)
        player_dicts.append(new_player)
        if (i % 25 == 0):
            print(new_player['player_name'])
        i += 1

    # for player,team in players:
    #     if i % 50 == 0:
    #         print(player)
        
    #     player_dict = get_player_data(player, team, year)
    #     # player_dict = {player: player_dict}
        
    #     i += 1
    if len(pickle.dumps(player_dicts)) > 2 * 10 ** 9:
        raise RuntimeError('return data cannot be sent')
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
            if row['AB'] > 0 or row['IP'] > 0:
                session.merge(model(**row))
    session.commit()

def etl_player_data(year):
    rosters = {}
    pa_query = '''select * from plateappearance where year = ''' + year
    print('plate_appearance')
    pa_start = time.time()
    pa_data_df = pd.concat(list(pd.read_sql_query(
        pa_query,
        con=ENGINE,
        chunksize = 10000
    )))
    game_data_df = pd.read_sql_table(
        'game',
        con=ENGINE
    )
    run_data_df = pd.read_sql_table(
        'run',
        con=ENGINE
    )
    br_data_df = pd.read_sql_table(
        'baserunningevent',
        con=ENGINE
    )
    print('done bigger chunk', time.time() - pa_start)
    parsed_data = get_game_data(year, pa_data_df, game_data_df, run_data_df, br_data_df)
    rows = {table: [] for table in ['Player']}
    rows['Player'].extend(parsed_data)
    load(rows)

# etl_player_data('2003')
# etl_player_data('2004')
# etl_player_data('2005')