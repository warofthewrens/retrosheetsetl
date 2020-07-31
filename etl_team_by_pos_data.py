import pandas as pd
import pymysql
from os import walk
import time
import shutil
from models.sqla_utils import ENGINE, BASE, get_session
from models.teamposition import TeamPosition
from models.reliefposition import ReliefPosition
from parsed_schemas.teamposition import TeamPosition as t
from parsed_schemas.reliefposition import ReliefPosition as r
from extract import extract_roster_team, extract_game_data_by_year
from extract_fangraphs import extract_fangraphs, extract_park_factors

MODELS = [TeamPosition]
RELIEFMODELS = [ReliefPosition]

expand_team = {
    'ARI' : 'Diamondbacks',
    'ATL' : 'Braves',
    'BAL' : 'Orioles',
    'BOS' : 'Red Sox',
    'CAL' : 'Angels',
    'ANA' : 'Angels',
    'CHA' : 'White Sox',
    'CHN' : 'Cubs',
    'CIN' : 'Reds',
    'CLE' : 'Indians',
    'COL' : 'Rockies',
    'DET' : 'Tigers',
    'FLA' : 'Marlins',
    'FLO' : 'Marlins',
    'HOU' : 'Astros',
    'KCA' : 'Royals',
    'LAN' : 'Dodgers',
    'MIL' : 'Brewers',
    'MIN' : 'Twins',
    'MIA' : 'Marlins',
    'MON' : 'Expos',
    'NYA' : 'Yankees',
    'NYN' : 'Mets',
    'OAK' : 'Athletics',
    'PHI' : 'Phillies',
    'PIT' : 'Pirates',
    'SDN' : 'Padres',
    'SEA' : 'Mariners',
    'SFN' : 'Giants',
    'SLN' : 'Cardinals',
    'TBA' : 'Rays',
    'TBD' : 'Devil Rays',
    'TEX' : 'Rangers',
    'TOR' : 'Blue Jays',
    'WAS' : 'Nationals'
}

positions = {
    2: 'Catcher',
    3: 'First Baseman',
    4: 'Second Baseman',
    5: 'Third Baseman',
    6: 'Shortstop',
    7: 'Left Field',
    8: 'Center Field',
    9: 'Right Field',
    10: 'Designate Hitter'
}

def get_player_position_wRAA(position_pa, player, woba_weights, pf_weights):
    player_dict = {}
    player_pos = position_pa[position_pa.batter_id == player]
    player_dict['PA'] = player_pos[player_pos.pa_flag == True].pa_flag.count()
    player_dict['AB'] = player_pos[player_pos.ab_flag == True].pa_flag.count()
    player_dict['S'] = player_pos[player_pos.hit_val == 1].hit_val.count()
    player_dict['D'] = player_pos[player_pos.hit_val == 2].hit_val.count()
    player_dict['T'] = player_pos[player_pos.hit_val == 3].hit_val.count()
    player_dict['HR'] = player_pos[player_pos.hit_val == 4].hit_val.count()
    player_dict['BB'] = player_pos[player_pos.event_type == 14].event_type.count()
    player_dict['IBB'] = player_pos[player_pos.event_type == 15].event_type.count()
    player_dict['HBP'] = player_pos[player_pos.event_type == 16].event_type.count()
    player_dict['SF'] = player_pos[player_pos.sac_fly == True].sac_fly.count()
    player_dict['SH'] = player_pos[player_pos.sac_bunt == True].sac_bunt.count()

    bb = woba_weights.wBB * (player_dict['BB'] - player_dict['IBB'])
    hbp = (woba_weights.wHBP * player_dict['HBP'])
    hits = (woba_weights.w1B * player_dict['S']) + (woba_weights.w2B * player_dict['D']) + (woba_weights.w3B + player_dict['T']) + (woba_weights.wHR * player_dict['HR'])

    sabr_PA = (player_dict['AB'] + player_dict['BB'] + player_dict['HBP'] + player_dict['SF'])
    sabr_PA_no_IBB = sabr_PA - player_dict['IBB']
    player_dict['PPFp'] = 1/((pf_weights['Basic (5yr)'] / 100).item())
    player_dict['wOBA'] = ((bb + hbp + hits)/sabr_PA_no_IBB)
    player_dict['wOBAadj'] = player_dict['wOBA'].item() * player_dict['PPFp']
    player_dict['wRAA'] = (((player_dict['wOBAadj'] - woba_weights.wOBA)/(woba_weights.wOBAScale)) * sabr_PA) 
    return player_dict['wRAA']

def get_team_data(team, year, position_code, pa_data_df, woba_weights, pf_weights):
    '''
    convert a combination of player, plate appearance, run, and game data into team level data
    @param team - three letter string team code
    @param year - string of appropriate year
    @param pa_data_df - dataframe containing every plate appearance from @year
    @param player_data_df - dataframe containing player statistics
    @param run_data_df - dataframe containing info for each run scored in @year
    @param game_data_df - dataframe containing info on every game played in the season
    @param woba_weights - dataframe containing the woba weight for every batting event
    '''
    team_dict = {}
    
    pa_year = (pa_data_df.pitcher_team == team) & (pa_data_df.year == int(year))
    pa_pos = (pa_data_df.field_pos == position_code)
    position_pa = pa_data_df[pa_pos & (pa_data_df.batter_team == team) & (pa_data_df.year == int(year))]
    player_counts = position_pa['batter_id'].value_counts(normalize=True)
    players = position_pa['batter_id'].value_counts().index.to_list()
    team_dict['PA_first'] = players[0]
    team_dict['PA_first_PA'] = player_counts[0]
    team_dict['PA_first_wRAA'] = get_player_position_wRAA(position_pa, team_dict['PA_first'], woba_weights, pf_weights)
    if len(players) > 1:
        team_dict['PA_second'] = players[1]
        team_dict['PA_second_PA'] = player_counts[1]
        team_dict['PA_second_wRAA'] = get_player_position_wRAA(position_pa, team_dict['PA_second'], woba_weights, pf_weights)
    if len(players) > 2:
        team_dict['PA_third'] = players[2]
        team_dict['PA_third_PA'] = player_counts[2]
        team_dict['PA_third_wRAA'] = get_player_position_wRAA(position_pa, team_dict['PA_third'], woba_weights, pf_weights)
    team_dict['team'] = team
    team_dict['year'] = year
    team_dict['position_code'] = position_code
    team_dict['position'] = positions[position_code]
    team_dict['PA'] = position_pa[position_pa.pa_flag == True].pa_flag.count()
    team_dict['AB'] = position_pa[position_pa.ab_flag == True].ab_flag.count()
    team_dict['S'] = position_pa[position_pa.hit_val == 1].hit_val.count()
    team_dict['D'] = position_pa[position_pa.hit_val == 2].hit_val.count()
    team_dict['T'] = position_pa[position_pa.hit_val == 3].hit_val.count()
    team_dict['HR'] = position_pa[position_pa.hit_val == 4].hit_val.count()
    team_dict['BB'] = position_pa[position_pa.event_type == 14].event_type.count()
    team_dict['IBB'] = position_pa[position_pa.event_type == 15].event_type.count()
    team_dict['HBP'] = position_pa[position_pa.event_type == 16].event_type.count()
    team_dict['SF'] = position_pa[position_pa.sac_fly == True].event_type.count()
    team_dict['SH'] = position_pa[position_pa.sac_bunt == True].event_type.count()
    bb = woba_weights.wBB * (team_dict['BB'] - team_dict['IBB'])
    hbp = (woba_weights.wHBP * team_dict['HBP'])
    hits = (woba_weights.w1B * team_dict['S']) + (woba_weights.w2B * team_dict['D']) + (woba_weights.w3B + team_dict['T']) + (woba_weights.wHR * team_dict['HR'])
    sabr_PA = (team_dict['AB'] + team_dict['BB'] + team_dict['HBP'] + team_dict['SF'])
    sabr_PA_no_IBB = sabr_PA - team_dict['IBB']
    team_dict['PPFp'] = 1/((pf_weights['Basic (5yr)'] / 100).item())
    team_dict['wOBA'] = ((bb + hbp + hits)/sabr_PA_no_IBB)
    team_dict['wOBAadj'] = team_dict['wOBA'].item() * team_dict['PPFp']
    team_dict['wRAA'] = (((team_dict['wOBAadj'] - woba_weights.wOBA)/(woba_weights.wOBAScale)) * sabr_PA) 
    return team_dict

def get_relief_team_data(team, year, pa_data_df):
    relief_dict = {}
    relief_dict['team'] = team
    relief_dict['year'] = year
    closer_df = pd.read_sql_query(
        '''SELECT player_id,WAR FROM retrosheet.Player where team = \'''' + team + '''\' and year = ''' + year + ''' order by SV desc limit 1''',
        con=ENGINE
    )
    relief_dict['closer'] = closer_df.iloc[0]['player_id']
    relief_dict['closer_WAR'] = closer_df.iloc[0]['WAR']
    relief_pa = pa_data_df[(pa_data_df.sp_flag == False) & (pa_data_df.pitcher_team == team) & (pa_data_df.year == int(year))]
    player_counts = relief_pa['pitcher_id'].value_counts(normalize=True)
    players = relief_pa['pitcher_id'].value_counts().index.to_list()
    i = 1
    reliever = 1
    while(i < len(player_counts) and reliever < 5):
        query = '''select player_id,WAR from Player where team = \'''' + team + '''\' and year = ''' + year + ''' and player_id = \'''' + players[i] + '''\''''
        player_df = pd.read_sql_query(
            query,
            con=ENGINE
        )
        if players[i] != relief_dict['closer']:

            relief_dict['relief_' + str(reliever)] = player_df.iloc[0]['player_id']
            relief_dict['relief_' + str(reliever) + '_WAR'] = player_df.iloc[0]['WAR']
            reliever+=1
        i+=1
        # print[player_counts[i]]
    return relief_dict

def get_relief_teams_data(year, pa_data_df):
    relief_dicts = []

    #create series of the teams 
    teams = pa_data_df[pa_data_df.year == int(year)].batter_team.unique()

    for team in teams:
        relief_dict = get_relief_team_data(team, year, pa_data_df)
        new_reliever = r().dump(relief_dict)
        relief_dicts.append(new_reliever)
    return relief_dicts

def get_teams_data(year, pa_data_df):
    '''
    Given a year collects statistics for every team
    @param year - string of appropriate year
    @param pa_data_df - dataframe containing every plate appearance from @year
    @param player_data_df - dataframe containing player statistics
    @param run_data_df - dataframe containing info for each run scored in @year
    @param game_data_df - dataframe containing info on every game played in the season
    '''
    team_dicts = []

    #create series of the teams 
    teams = pa_data_df[pa_data_df.year == int(year)].batter_team.unique()
    print(teams)

    #using fangraphs data retrieve appropriate wOBA weights
    woba_df = extract_fangraphs()
    woba_weights = woba_df[woba_df.Season == int(year)]

    pf_df = extract_park_factors(year)
    #for each team build and serialize data
    for team in teams:
        if team == 'TBA' and int(year) <= 2007:
            team = 'TBD'
        pf_weights = pf_df[pf_df.Team == expand_team[team]]
        if team == 'TBD':
            team = 'TBA'
        for position_code in positions.keys():
            team_dict = get_team_data(team, year, position_code, pa_data_df, woba_weights, pf_weights)
            new_team = t().dump(team_dict)
            team_dicts.append(new_team)
    return team_dicts

def load(results):
    '''
    @param results - dictionary of a list of teams to be loaded into the SQL database
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
            session.merge(model(**row))
    
    session.commit()

def relief_load(results):
    BASE.metadata.create_all(tables=[x.__table__ for x in RELIEFMODELS], checkfirst=True)
    session = get_session()
    for model in RELIEFMODELS:
        data = results[model.__tablename__]
        i = 0
        # Here is where we convert directly the dictionary output of our marshmallow schema into sqlalchemy
        objs = []
        for row in data:
            if i % 1000 == 0:
                print('loading...', i)
            i+=1
            session.merge(model(**row))
    
    session.commit()

def etl_team_data(year):
    '''
    @param year - string with appropriate year
    '''
    print(year)

    # Query the SQL database for every plate apperance
    pa_query = '''
        select * from PlateAppearance where year =
    ''' + year
    pa_data_df = pd.concat(list(pd.read_sql_query(
        pa_query,
        con = ENGINE,
        chunksize = 1000
    )))
    parsed_data = get_teams_data(year, pa_data_df)
    rows = {table: [] for table in ['TeamPosition']}
    rows['TeamPosition'].extend(parsed_data)
    load(rows)

def etl_relief_data(year):
    pa_query = '''
        select * from PlateAppearance where year =
    ''' + year
    pa_data_df = pd.concat(list(pd.read_sql_query(
        pa_query,
        con = ENGINE,
        chunksize = 1000
    )))
    parsed_data = get_relief_teams_data(year, pa_data_df)
    rows = {table: [] for table in ['ReliefPosition']}
    rows['ReliefPosition'].extend(parsed_data)
    relief_load(rows)

#etl_relief_data('2019')
#etl_team_data('2018')
for i in range(2002, 2019):
    etl_relief_data(str(i))
