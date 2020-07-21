import pandas as pd
import pymysql
from os import walk
import time
import shutil
from models.sqla_utils import ENGINE, BASE, get_session
from models.league import League
from parsed_schemas.league import League as l
from extract import extract_roster_team, extract_game_data_by_year
from extract_fangraphs import extract_fangraphs, extract_park_factors

MODELS = [League]

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

def build_league_stats(year, league, team_data_df):
    league_dict = {}
    league_dict['year'] = year
    league_dict['league'] = league
    league_dict['H'] = team_data_df.H.sum()
    league_dict['HR'] = team_data_df.HR.sum()
    league_dict['BB'] = team_data_df.BB.sum()
    league_dict['K'] = team_data_df.K.sum()
    league_dict['HBP'] = team_data_df.HBP.sum()
    league_dict['IFFB'] = team_data_df.IFFB.sum()
    league_dict['IP'] = team_data_df.IP.sum()
    league_dict['ER'] = team_data_df.ER.sum()
    league_dict['TR'] = team_data_df.TR.sum()
    league_dict['SpER'] = team_data_df.SpER.sum()
    league_dict['SpTR'] = team_data_df.SpTR.sum()
    league_dict['SpIP'] = team_data_df.SpIP.sum()
    league_dict['RpER'] = team_data_df.RpER.sum()
    league_dict['RpTR'] = team_data_df.RpTR.sum()
    league_dict['RpIP'] = team_data_df.RpIP.sum()
    league_dict['lgRAA'] = (league_dict['TR'] / league_dict['IP']) * 9
    league_dict['lgERA'] = (league_dict['ER'] / league_dict['IP']) * 9
    league_dict['lgFIP'] = ((13 * league_dict['HR']) + (3 * (league_dict['BB'] + league_dict['HBP']) - (2 * league_dict['K']))) / league_dict['IP']
    league_dict['lgiFIP'] = ((13 * league_dict['HR']) + (3 * (league_dict['BB'] + league_dict['HBP']) - (2 * (league_dict['K'] + league_dict['IFFB']))))/ league_dict['IP']
    league_dict['cFIP'] = league_dict['lgERA'] - (league_dict['lgFIP'])
    league_dict['ciFIP'] = (league_dict['lgERA'] - (league_dict['lgiFIP']))
    league_dict['lgFIP'] = league_dict['cFIP'] + league_dict['lgFIP']
    league_dict['lgiFIP'] = league_dict['ciFIP'] + league_dict['lgiFIP']
    league_dict['WARadj'] = (league_dict['lgRAA'] - (league_dict['lgERA']))
    league_dict['FIPR9'] = league_dict['lgiFIP'] + league_dict['WARadj']
    league_dict['lgSpERA'] = (league_dict['SpER'] / league_dict['SpIP']) * 9
    league_dict['lgSpRAA'] = (league_dict['SpTR'] / league_dict['SpIP']) * 9
    league_dict['lgSpFIP'] = (team_data_df.SpFIP * team_data_df.SpIP).sum() / (league_dict['SpIP'])
    league_dict['lgRpERA'] = (league_dict['RpER'] / league_dict['RpIP']) * 9
    league_dict['lgRpRAA'] = (league_dict['RpTR'] / league_dict['RpIP']) * 9
    league_dict['lgRpFIP'] = (team_data_df.RpFIP * team_data_df.RpIP).sum() / (league_dict['RpIP'])
    return league_dict


def get_league_data(year, league, team_data_df):
    '''
    Given a year collects statistics for every team
    @param year - string of appropriate year
    @param pa_data_df - dataframe containing every plate appearance from @year
    @param player_data_df - dataframe containing player statistics
    @param run_data_df - dataframe containing info for each run scored in @year
    @param game_data_df - dataframe containing info on every game played in the season
    '''

    #create series of the teams 
    team_data_df = team_data_df[team_data_df.year == int(year)]

    league_dict = build_league_stats(year, league, team_data_df)
    #for each team build and serialize data
    league = l().dump(league_dict)
    return league

def load(league):
    '''
    @param results - dictionary of a list of teams to be loaded into the SQL database
    '''
    BASE.metadata.create_all(tables=[x.__table__ for x in MODELS], checkfirst=True)
    session = get_session()
    for model in MODELS:
        session.merge(model(**league))
    session.commit()

def etl_league_data(year, league='MLB'):
    '''
    @param year - string with appropriate year
    '''
    print(year)
    # Query the SQL database for every team

    team_data_df = pd.read_sql_table(
        'team',
        con = ENGINE
    )
    if league != 'MLB':
        team_data_df = team_data_df[team_data_df['league'] == league]
    league_data = get_league_data(year, league, team_data_df)
    load(league_data)

def etl_league_data_separated(year):
    etl_league_data(year)
    etl_league_data(year, 'NL')
    etl_league_data(year, 'AL')
# for i in range(1990, 2020):
#     etl_league_data(str(i))