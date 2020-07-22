import pandas as pd
import pymysql
from os import walk
import time
import shutil
from models.sqla_utils import ENGINE, BASE, get_session
from models.team import Team
from parsed_schemas.team import Team as t
from extract import extract_roster_team, extract_game_data_by_year
from extract_fangraphs import extract_fangraphs, extract_park_factors

MODELS = [Team]

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



def get_team_data(team, year, pa_data_df, player_data_df, game_data_df, run_data_df, woba_weights, pf_weights, nl_teams, al_teams):
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
    game_year = (game_data_df.year == int(year))
    player_year = (player_data_df.year == int(year)) & (player_data_df.team == team)
    pa_year = (pa_data_df.pitcher_team == team) & (pa_data_df.year == int(year))
    run_year = (run_data_df.year == int(year))
    team_dict['team'] = team
    team_dict['year'] = year
    if team in nl_teams:
        team_dict['league'] = 'NL'
    elif team in al_teams:
        team_dict['league'] = 'AL'
    team_dict['W'] = game_data_df[(game_data_df.winning_team == team) & game_year].winning_team.count()
    team_dict['L'] = game_data_df[(game_data_df.losing_team == team) & game_year].losing_team.count()
    team_dict['win_pct'] = (team_dict['W']/(team_dict['W'] + team_dict['L']))
    team_dict['homeW'] = game_data_df[(game_data_df.winning_team == team) & (game_data_df.home_team == team) & game_year].home_team.count()
    team_dict['homeL'] = game_data_df[(game_data_df.losing_team == team) & (game_data_df.home_team == team) & game_year].home_team.count()
    team_dict['awayW'] = game_data_df[(game_data_df.winning_team == team) & (game_data_df.away_team == team) & game_year].away_team.count()
    team_dict['awayL'] = game_data_df[(game_data_df.losing_team == team) & (game_data_df.away_team == team) & game_year].away_team.count()
    team_dict['RS'] = game_data_df[(game_data_df.home_team == team) & game_year].home_team_runs.sum() + game_data_df[(game_data_df.away_team == team) & game_year].away_team_runs.sum()
    team_dict['RA'] = game_data_df[(game_data_df.home_team == team) & game_year].away_team_runs.sum() + game_data_df[(game_data_df.away_team == team) & game_year].home_team_runs.sum()
    team_dict['DIFF'] = team_dict['RS'] - team_dict['RA']
    team_dict['exp_win_pct'] = (1/(1 + ((team_dict['RA']/team_dict['RS'])**1.83)))
    team_dict['PA'] = player_data_df[player_year].PA.sum()
    team_dict['AB'] = player_data_df[player_year].AB.sum()
    team_dict['S'] = player_data_df[player_year].S.sum()
    team_dict['D'] = player_data_df[player_year].D.sum()
    team_dict['T'] = player_data_df[player_year & (player_data_df['T'] > 0)]['T'].sum()
    team_dict['HR'] = player_data_df[player_year].HR.sum()
    team_dict['TB'] = player_data_df[player_year].TB.sum()
    team_dict['H'] = player_data_df[player_year].H.sum()
    team_dict['R'] = player_data_df[player_year].R.sum()
    team_dict['RBI'] = player_data_df[player_year].RBI.sum()
    team_dict['SB'] = player_data_df[player_year].SB.sum()
    team_dict['CS'] = player_data_df[player_year].CS.sum()
    team_dict['BB'] = player_data_df[player_year].BB.sum()
    team_dict['IBB'] = player_data_df[player_year].IBB.sum()
    team_dict['SO'] = player_data_df[player_year].SO.sum()
    team_dict['HBP'] = player_data_df[player_year].HBP.sum()
    team_dict['SF'] = player_data_df[player_year].SF.sum()
    team_dict['SH'] = player_data_df[player_year].SH.sum()
    team_dict['AVG'] = team_dict['H'] / team_dict['AB'] + 0.0
    team_dict['OBP'] = (team_dict['H'] + team_dict['BB'] + team_dict['HBP'] + 0.0) / (team_dict['AB'] + team_dict['BB'] + team_dict['HBP'] + team_dict['SF'])
    team_dict['SLG'] = team_dict['TB'] / team_dict ['AB']
    team_dict['OPS'] = team_dict['OBP'] + team_dict['SLG']
    bb = woba_weights.wBB * (team_dict['BB'] - team_dict['IBB'])
    hbp = (woba_weights.wHBP * team_dict['HBP'])
    hits = (woba_weights.w1B * team_dict['S']) + (woba_weights.w2B * team_dict['D']) + (woba_weights.w3B + team_dict['T']) + (woba_weights.wHR * team_dict['HR'])
    baserunning = (woba_weights.runSB * team_dict['SB']) + (woba_weights.runCS * team_dict['CS'])
    sabr_PA = (team_dict['AB'] + team_dict['BB'] + team_dict['HBP'] + team_dict['SF'])
    sabr_PA_no_IBB = sabr_PA - team_dict['IBB']
    team_dict['PPFp'] = (pf_weights['Basic (5yr)'] / 100).item()
    team_dict['wOBA'] = ((bb + hbp + hits + baserunning)/sabr_PA_no_IBB) * team_dict['PPFp']
    team_dict['wRAA'] = (((team_dict['wOBA'] - woba_weights.wOBA)/(woba_weights.wOBAScale)) * sabr_PA) 
    team_dict['BF'] = player_data_df[player_year].BF.sum()
    team_dict['IP'] = player_data_df[player_year].IP.sum()
    team_dict['Ha'] = player_data_df[player_year].Ha.sum()
    team_dict['HRa'] = player_data_df[player_year].HRa.sum()
    team_dict['TBa'] = player_data_df[player_year].TBa.sum()
    team_dict['BBa'] = player_data_df[player_year].BBa.sum()
    team_dict['IBBa'] = player_data_df[player_year].IBBa.sum()
    team_dict['IFFB'] = player_data_df[player_year].IFFB.sum()
    team_dict['K'] = player_data_df[player_year].K.sum()
    team_dict['HBPa'] = player_data_df[player_year].HBPa.sum()
    team_dict['BK'] = player_data_df[player_year].BK.sum()
    team_dict['SV'] = player_data_df[player_year].SV.sum()
    team_dict['TR'] = run_data_df[(run_data_df.conceding_team == team)  & run_year].conceding_team.count()
    team_dict['ER'] = run_data_df[(run_data_df.conceding_team == team) 
                                 & (run_data_df.is_team_earned) & run_year].conceding_team.count()
    team_dict['RAA'] = (team_dict['TR'] / team_dict['IP']) * 9
    team_dict['ERA'] = (team_dict['ER'] / team_dict['IP']) * 9
    team_dict['FIP'] = (((13 * team_dict['HRa']) + (3 * (team_dict['BBa'] + team_dict['HBPa'])) - (2 * team_dict['K']))/team_dict['IP'])
    team_dict['SpIP'] = pa_data_df[(pa_data_df.sp_flag) & pa_year].outs_on_play.sum()/3
    team_dict['RpIP'] = pa_data_df[~(pa_data_df.sp_flag) & pa_year].outs_on_play.sum()/3
    # print(type((run_data_df.is_sp)), type(run_data_df.conceding_team == team), type(run_data_df.is_earned), type(run_year))
    team_dict['SpER'] = run_data_df[((run_data_df.is_sp == True) & (run_data_df.conceding_team == team)) & (run_data_df.is_earned == True) & run_year].is_sp.count()
    team_dict['RpER'] = run_data_df[(run_data_df.is_sp == False) & (run_data_df.conceding_team == team) 
                                    & (run_data_df.is_earned == True) & run_year].is_sp.count()
    team_dict['SpTR'] = run_data_df[(run_data_df.is_sp == True) & (run_data_df.conceding_team == team) 
                                    & run_year].is_sp.count()
    team_dict['RpTR'] = run_data_df[(run_data_df.is_sp == False) & (run_data_df.conceding_team == team) 
                                     & run_year].is_sp.count()
    relief_hr = pa_data_df[(pa_data_df.sp_flag == False) & (pa_data_df.pitcher_team == team) & (pa_data_df.hit_val == 4) & pa_year].pa_flag.count()
    relief_k = pa_data_df[(pa_data_df.sp_flag == False) & (pa_data_df.pitcher_team == team) & (pa_data_df.event_type == 3) & pa_year].pa_flag.count()
    relief_bb = pa_data_df[(pa_data_df.sp_flag == False) & (pa_data_df.pitcher_team == team) & ((pa_data_df.event_type == 14) | (pa_data_df.event_type == 15)) & pa_year].pa_flag.count()
    start_hr = team_dict['HRa'] - relief_hr
    start_k = team_dict['K'] - relief_k
    start_bb = (team_dict['BBa'] - team_dict['HBPa']) - relief_bb
    
    team_dict['SpERA'] = (team_dict['SpER'] / team_dict['SpIP']) * 9
    team_dict['RpERA'] = (team_dict['RpER'] / team_dict['RpIP']) * 9
    team_dict['SpFIP'] = ((13 * start_hr) + (3 * start_bb) - (2 * start_k)) / (team_dict['SpIP'])
    team_dict['RpFIP'] = ((13 * relief_hr) + (3 * relief_bb) - (2 * relief_k))/(team_dict['RpIP'])
    return team_dict

def get_teams_data(year, pa_data_df, player_data_df, game_data_df, run_data_df, nl_teams, al_teams):
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
    teams = player_data_df[player_data_df.year == int(year)].team.unique()
    print(teams)

    #using fangraphs data retrieve appropriate wOBA weights
    woba_df = extract_fangraphs()
    woba_weights = woba_df[woba_df.Season == int(year)]

    pf_df = extract_park_factors(year)
    print(pf_df)
    #for each team build and serialize data
    for team in teams:
        if team == 'TBA' and int(year) <= 2007:
            team = 'TBD'
        pf_weights = pf_df[pf_df.Team == expand_team[team]]
        if team == 'TBD':
            team = 'TBA'
        print(pf_weights)
        team_dict = get_team_data(team, year, pa_data_df, player_data_df, game_data_df, run_data_df, woba_weights, pf_weights, nl_teams, al_teams)
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

def etl_team_data(year):
    '''
    @param year - string with appropriate year
    '''
    print(year)
    data_zip, data_td = extract_game_data_by_year(year)
    nl_teams = set([])
    al_teams = set([])
    f = []
    for (dirpath, dirnames, filenames) in walk(data_td):
        f.extend(filenames)
        break
    shutil.rmtree(data_td)
    print(f)
    for team_file in f:
        if team_file[-4:] == '.EVN':
            nl_teams.add(team_file[4:7])
        if team_file[-4:] == '.EVA':
            al_teams.add(team_file[4:7])

    # Query the SQL database for every plate apperance
    pa_query = '''
        select * from plateappearance where year =
    ''' + year
    pa_data_df = pd.concat(list(pd.read_sql_query(
        pa_query,
        con = ENGINE,
        chunksize = 1000
    )))
    #Player Query
    player_data_df = pd.read_sql_table(
        'player',
        con = ENGINE
    )
    #Game Query
    game_data_df = pd.read_sql_table(
        'game',
        con = ENGINE
    )
    #Run Query
    run_data_df = pd.read_sql_table(
        'run',
        con = ENGINE
    )
    parsed_data = get_teams_data(year, pa_data_df, player_data_df, game_data_df, run_data_df, nl_teams, al_teams)
    rows = {table: [] for table in ['team']}
    rows['team'].extend(parsed_data)
    load(rows)

etl_team_data('2019')
# for i in range(2007, 2019):
#     etl_team_data(str(i))
