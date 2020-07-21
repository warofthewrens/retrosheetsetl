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
from extract_fangraphs import extract_fangraphs

MODELS = [Player]

team_set = set(['ANA', 'ARI', 'ATL', 'BAL', 'BOS', 'CHA', 'CHN', 'CIN', 'CLE', 'COL', 'DET', 'HOU',
            'KCA', 'LAN', 'MIA', 'MIL', 'MIN', 'NYA', 'NYN', 'OAK', 'PHI', 'PIT', 'SEA',
            'SFN', 'SLN', 'SDN', 'TBA', 'TEX', 'TOR', 'WAS'])

rosters = {}

def get_rosters(year, data_zip):
    for team in team_set:
        rosters.update(extract_roster_team(year + team, data_zip))
    return rosters

def get_player_data(player, team, year, pa_data_df, game_data_df, run_data_df, br_data_df, woba_weights):
    '''
    convert a combination of player, plate appearance, run, and game data into team level 
    @param player - 8 character player id of player
    @param team - three letter string team code
    @param year - string of appropriate year
    @param pa_data_df - dataframe containing every plate appearance from @year
    @param player_data_df - dataframe containing player statistics
    @param run_data_df - dataframe containing info for each run scored in @year
    @param game_data_df - dataframe containing info on every game played in the season
    @param woba_weights - dataframe containing the woba weight for every batting event
    '''
    player_dict = {}
    pa_year = pa_data_df.year == int(year)
    game_year = game_data_df.year == int(year)
    run_year = run_data_df.year == int(year)
    br_year = (br_data_df.year == int(year)) & (br_data_df.running_team == team)
    infield = set([1, 2, 3, 4, 5, 6])
    team_scored = run_data_df.scoring_team == team
    team_scored_against = run_data_df.conceding_team == team
    position = (game_data_df.starting_pitcher_home == player) | (game_data_df.starting_catcher_home == player) | (game_data_df.starting_first_home == player) | (game_data_df.starting_second_home == player) | (game_data_df.starting_third_home == player)  | (game_data_df.starting_short_home == player) | (game_data_df.starting_left_home == player) | (game_data_df.starting_right_home == player) | (game_data_df.starting_center_home == player) | (game_data_df.starting_pitcher_away == player) | (game_data_df.starting_catcher_away == player) | (game_data_df.starting_first_away == player) | (game_data_df.starting_second_away == player)| (game_data_df.starting_third_away == player) | (game_data_df.starting_short_away == player) | (game_data_df.starting_left_away == player) | (game_data_df.starting_right_away == player) | (game_data_df.starting_center_away == player)
    infield_fly = (pa_data_df.hit_loc <= 6) & ((pa_data_df.ball_type == 'F') | (pa_data_df.ball_type == 'P'))
    batter_team_bool = (pa_data_df.batter_id == player) & (pa_data_df.batter_team == team) 
    pitcher_team_bool = (pa_data_df.pitcher_id == player) & (pa_data_df.pitcher_team == team)
    player_dict['GS'] = game_data_df[position & game_year].year.count()
    player_dict['GP'] = len(pa_data_df[(pitcher_team_bool | batter_team_bool) & pa_year].game_id.unique())
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
    player_dict['IBB'] = pa_data_df[batter_team_bool & (pa_data_df.event_type == 15) & pa_year].pa_flag.count()
    player_dict['SO'] = pa_data_df[batter_team_bool & (pa_data_df.event_type == 3) & pa_year].pa_flag.count()
    player_dict['HBP'] = pa_data_df[batter_team_bool & (pa_data_df.event_type == 16) & pa_year].pa_flag.count()
    player_dict['SF'] = pa_data_df[batter_team_bool & pa_data_df.sac_fly & pa_year].pa_flag.count()
    player_dict['SH'] = pa_data_df[batter_team_bool & pa_data_df.sac_bunt & pa_year].pa_flag.count()
    if player_dict['AB'] > 0:
        player_dict['AVG'] = player_dict['H'] / player_dict['AB'] + 0.0
        player_dict['OBP'] = (player_dict['H'] + player_dict['BB'] + player_dict['HBP']) / (player_dict['AB'] + player_dict['BB'] + player_dict['HBP'] + player_dict['SF'])
        player_dict['SLG'] = player_dict['TB']/player_dict['AB']
        player_dict['OPS'] = player_dict['OBP'] + player_dict['SLG']
        bb = woba_weights.wBB * (player_dict['BB'] - player_dict['IBB'])
        hbp = (woba_weights.wHBP * player_dict['HBP'])
        hits = (woba_weights.w1B * player_dict['S']) + (woba_weights.w2B * player_dict['D']) + (woba_weights.w3B + player_dict['T']) + (woba_weights.wHR * player_dict['HR'])
        baserunning = (woba_weights.runSB * player_dict['SB']) + (woba_weights.runCS * player_dict['CS'])
        sabr_PA = (player_dict['AB'] + player_dict['BB'] + player_dict['HBP'] + player_dict['SF'])
        sabr_PA_no_IBB = sabr_PA - player_dict['IBB']
        player_dict['wOBA'] = (bb + hbp + hits + baserunning)/sabr_PA_no_IBB
        player_dict['wRAA'] = ((player_dict['wOBA'] - woba_weights.wOBA)/(woba_weights.wOBAScale)) * sabr_PA
    else:
        player_dict['AVG'], player_dict['OBP'], player_dict['SLG'], player_dict['OPS'], player_dict['wOBA'], player_dict['wRAA'] = 0, 0, 0, 0, 0, 0
    player_dict['BF'] = pa_data_df[pitcher_team_bool & pa_data_df.pa_flag & pa_year].pa_flag.count()
    player_dict['IP'] = float((pa_data_df[pitcher_team_bool & (pa_data_df.outs_on_play > 0) & pa_year].outs_on_play.sum() + 0.0)/3.0)
    player_dict['Ha'] = pa_data_df[pitcher_team_bool & (pa_data_df.hit_val>0) & pa_year].hit_val.count()
    player_dict['HRa'] = pa_data_df[pitcher_team_bool & (pa_data_df.hit_val == 4) & pa_year].hit_val.count()
    player_dict['TBa'] = pa_data_df[pitcher_team_bool & (pa_data_df.hit_val>0) & pa_year].hit_val.sum()
    player_dict['BBa'] = pa_data_df[pitcher_team_bool & ((pa_data_df.event_type == 14) | (pa_data_df.event_type == 15)) & pa_year].event_type.count()
    player_dict['IBBa'] = pa_data_df[pitcher_team_bool & (pa_data_df.event_type == 15) & pa_year].event_type.count()
    player_dict['K'] = pa_data_df[pitcher_team_bool & (pa_data_df.event_type == 3) & pa_year].event_type.count()
    player_dict['HBPa'] = pa_data_df[pitcher_team_bool & (pa_data_df.event_type == 16) & pa_year].event_type.count()
    player_dict['IFFB'] = pa_data_df[pitcher_team_bool & infield_fly & pa_year].hit_loc.count()
    player_dict['BK'] = pa_data_df[pitcher_team_bool & (pa_data_df.event_type == 11) & pa_year].event_type.count()
    player_dict['W'] = game_data_df[(player == game_data_df.winning_pitcher) & (team == game_data_df.winning_team) & game_year].winning_team.count()
    player_dict['L'] = game_data_df[(player == game_data_df.losing_pitcher) & (team == game_data_df.losing_team) & game_year].losing_team.count()
    player_dict['SV'] = game_data_df[(player == game_data_df.save) & (team == game_data_df.winning_team) & game_year].winning_team.count()
    player_dict['TR'] = run_data_df[(run_data_df.responsible_pitcher == player) & team_scored_against & run_year].responsible_pitcher.count()
    player_dict['ER'] = run_data_df[(run_data_df.responsible_pitcher == player) & team_scored_against & run_data_df.is_earned & run_year].responsible_pitcher.count()
    if player_dict['IP'] > 0:
        player_dict['RA'] = (player_dict['TR'] / player_dict['IP']) * 9
        player_dict['ERA'] = (player_dict['ER'] / player_dict['IP']) * 9
        player_dict['FIP'] = ((13 * player_dict['HRa'] + (3 * (player_dict['BBa'] + player_dict['HBPa'])) - 2 * (player_dict['K']))) / player_dict['IP']
        player_dict['iFIP'] = ((13 * player_dict['HRa'] + (3 * (player_dict['BBa'] + player_dict['HBPa'])) - 2 * (player_dict['K'] + player_dict['IFFB']))) / player_dict['IP']
    else:
        player_dict['RA'], player_dict['ERA'], player_dict['FIP'], player_dict['iFIP'] = 0,0,0,0
    
    player_dict['player_id'] = player
    player_dict['team'] = team
    player_dict['year'] = year
    player_dict['player_name'] = rosters[(player, team)]['player_first_name'] + ' ' + rosters[(player, team)]['player_last_name']
    return player_dict

def get_game_data(year, pa_data_df, game_data_df, run_data_df, br_data_df):
    '''
    @param year - string of appropriate year
    @param pa_data_df - dataframe containing every plate appearance from @year
    @param player_data_df - dataframe containing player statistics
    @param run_data_df - dataframe containing info for each run scored in @year
    @param game_data_df - dataframe containing info on every game played in the season
    @param woba_weights - dataframe containing the woba weight for every batting event
    '''
    global rosters
    roster_files = set([])
    
    # scrape fangraphs for wOBA weights
    woba_df = extract_fangraphs()
    woba_weights = woba_df[woba_df.Season == int(year)]

    # extract rosters from retrosheet
    data_zip, data_td = extract_game_data_by_year(year)
    f = []
    for (dirpath, dirnames, filenames) in walk(data_td):
        f.extend(filenames)
        break
    shutil.rmtree(data_td)
    for team_file in f:
        if team_file[-4:] == '.ROS':
            roster_files.add(team_file)
    rosters = {}
    for team in roster_files:
        rosters.update(extract_roster_team(team, data_zip))

    # Collect player data from plate appearance, game and run data for every player
    players = rosters.keys()
    print(len(players))
    player_dicts = []
    i = 0
    for player, team in players:
        player_dict = get_player_data(player, team, year, pa_data_df, game_data_df, run_data_df, br_data_df, woba_weights)
        new_player = p().dump(player_dict)
        print(new_player['iFIP'])
        player_dicts.append(new_player)
        if (i % 25 == 0):
            print(new_player)
        i += 1

    if len(pickle.dumps(player_dicts)) > 2 * 10 ** 9:
        raise RuntimeError('return data cannot be sent')
    return player_dicts

def load(results):
    '''
    load all of the player data into the SQL database
    @param results - dictionary of lists of dictionaries containing all the individual player rows of data
    '''
    BASE.metadata.create_all(tables=[x.__table__ for x in MODELS], checkfirst=True)
    session = get_session()
    for model in MODELS:
        data = results[model.__tablename__]
        i = 0
        # Here is where we convert directly the dictionary output of our marshmallow schema into sqlalchemy
        objs = []
        for row in data:
            if i % 1000 == 0:
                print('loading...', i)
            i+=1
            if row['AB'] > 0 or row['IP'] > 0:
                objs.append(model(**row))
                #session.merge(model(**row))
        
        session.bulk_save_objects(objs)
    session.commit()

def etl_player_data(year):
    '''
    wrapper for all the functions extracting, transforming, and loading player data from the data 
    already in the SQL database for a given year
    @param year - string containing the year for which to collect data
    '''
    rosters = {}
    pa_query = '''select * from plateappearance where year = ''' + year
    print('plate_appearance')
    pa_start = time.time()

    # Extract the plate appearance data
    pa_data_df = pd.concat(list(pd.read_sql_query(
        pa_query,
        con=ENGINE,
        chunksize = 10000
    )))

    # Extract game data
    game_data_df = pd.read_sql_table(
        'game',
        con=ENGINE
    )

    # Extract run data
    run_data_df = pd.read_sql_table(
        'run',
        con=ENGINE
    )

    # Extract base running events
    br_data_df = pd.read_sql_table(
        'baserunningevent',
        con=ENGINE
    )
    print('done bigger chunk', time.time() - pa_start)

    # Transform the data into player data
    parsed_data = get_game_data(year, pa_data_df, game_data_df, run_data_df, br_data_df)
    rows = {table: [] for table in ['Player']}
    rows['Player'].extend(parsed_data)

    # Load data into SQL Database
    load(rows)



# etl_player_data('2004')
# etl_player_data('2005')