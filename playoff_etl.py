from extract import extract_playoff_data_by_year, extract_team, extract_roster
from transform import transform_game 
from playoff_load import create_tables, load_data
import pandas as pd
from models.sqla_utils import ENGINE
from parsed_schemas.series import Series as s
import collections
# from extract_game_data import etl_player_data
# from etl_team_data import etl_team_data
from os import walk
import concurrent.futures
import time
import sys
import getopt
import shutil

def determine_home_field_advantage(year, series_type, team1, team2, team1_info, team2_info):
    home_field = team1
    if series_type == 'NLCS' and year < 1998:
        if (team1_info['division'] == 'E' and year % 2 == 1) or (team1_info['division'] == 'W' and year % 2 == 0):
            home_field = team1
        else:
            home_field = team2
    elif series_type == 'ALCS' and year < 1998:
        if (team1_info['division'] == 'W' and year % 2 == 1) or (team1_info['division'] == 'E' and year % 2 == 0):
            home_field = team1
        else:
            home_field = team2
    elif series_type != 'WS':
        if team1_info['wild_card'] and not team2_info['wild_card']:
            home_field = team2
        elif not team1_info['wild_card'] and team2_info['wild_card']:
            home_field = team1
        else:
            if team1_info['win_pct'] > team2_info['win_pct']:
                home_field = team1
            else:
                home_field = team2
    else:
        if year <= 1993:
            if (team1_info['league'] == 'NL' and year % 2 == 0) or (team1_info['league'] == 'AL' and year % 2 == 1):
                home_field = team1
            else:
                home_field = team2
        elif year <= 2002 and year >= 1995:
            if (team1_info['league'] == 'AL' and year % 2 == 0) or (team1_info['league'] == 'NL' and year % 2 == 1):
                home_field = team1
            else:
                home_field = team2

        elif year >= 2003 and year < 2017:
            if year < 2010 or year > 2012:
                if team1_info['league'] == 'AL':
                    home_field = team1
                else:
                    home_field = team2
            else:
                if team1_info['league'] == 'NL':
                    home_field = team1
                else:
                    home_field = team2   
        else:
            if team1_info['win_pct'] > team2_info['win_pct']:
                home_field = team1
            else:
                home_field = team2
    return home_field

def main():
    '''
    Extract, Transform, and Load playoff data from Retrosheet.
    '''
    years = [1990, 1991, 1992, 1993, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019]
    #years = [1990]
    years = [str(i) for i in years]
    teams = set([])

    # Get system arguments
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'y:t:')
    except getopt.GetoptError as err:
        print(str(err))
        sys.exit(2)
    print(opts, args)

    # Identify the years and teams to be etl
    for o, a in opts:
        if o == '-y':
            if int(a) < 1920 or int(a) > 2019:
                raise Exception('invalid year')
            else:
                years.append(a)
        if o == '-t':
            teams.add(a)
    results = {'PlateAppearance': [], 'Game': [], 'Run': [], 'BaseRunningEvent': [], 'Series': []}

    # Extract, transform and load each years worth of playoff data
    for year in years:
        games = []
        rosters = {}
        roster_files = set([])
        game_files = set([])
        # Download Retrosheet data for appropriate playoff year
        data_zip, data_td = extract_playoff_data_by_year(year)
        
        # Collect the files from the downloaded data
        f = []
        for (dirpath, dirnames, filenames) in walk(data_td):
            f.extend(filenames)
            break
        shutil.rmtree(data_td)
        print(f)

        # If no team is identified, default is every team
        if len(teams) == 0:
            for team_file in f:
                # Add roster file
                if team_file[-4:] == '.ROS':
                    print('roster', team_file)
                    roster_files.add(team_file)
                
                # Add game file
                elif team_file[-4:] == '.EVE':
                    print('game', team_file)
                    game_files.add(team_file)
                else:
                    print(team_file)
        # Otherwise only collect the teams which were identified
        else:
            for team_file in f:
                if team_file[-4:] == '.ROS':
                    roster_files.add(team_file)
                if team_file[4:7] in teams:
                    game_files.add(team_file)
        
        # Extract rosters from every team no matter what
        for team in roster_files:
            rosters.update(extract_roster(team, data_zip))

        home_team_adv_df = pd.read_sql_query(
            '''SELECT t.team, t.year, t.league, d.division, t.W, t.L, t.win_pct FROM retrosheet.team t
                join retrosheet.divisions d
                on t.team = d.old_id and t.year >= d.start_date and t.year <= end_year and year = ''' + year + ''';''',
            con=ENGINE
        )
        # Extract games from every identified team
        for series in game_files:
            series_dict = {}
            series_games = extract_team(series, data_zip)
            series = series.split('.')[0]
            series_type = series[4:]
            series_dict["year"] = year
            series_dict['series_id'] = series
            series_dict['series'] = series_type

            game_wins = collections.defaultdict(int)
            team_info = collections.defaultdict(dict)
            reg_win_pct = {}
            wild_card = {}
            for series_game in series_games:
                parsed_data = transform_game(series_game, rosters)
                parsed_data['game'][0]['series_id'] = series
                results['PlateAppearance'].extend(parsed_data['plate_appearance'])
                game_wins[parsed_data['game'][0]['winning_team']] += 1
                game_wins[parsed_data['game'][0]['losing_team']] =  game_wins[parsed_data['game'][0]['losing_team']]
                results['Game'].extend(parsed_data['game'])
                results['Run'].extend(parsed_data['run'])
                results['BaseRunningEvent'].extend(parsed_data['base_running_event'])
            for team in game_wins.keys():
                league_id = home_team_adv_df[home_team_adv_df['team'] == team]['league'].iloc[0]
                division_id = home_team_adv_df[home_team_adv_df['team'] == team]['division'].iloc[0]
                team_info[team]['win_pct'] = home_team_adv_df[home_team_adv_df['team'] == team]['win_pct'].item()
                team_info[team]['league'] = league_id
                team_info[team]['division'] = division_id
                division = home_team_adv_df[(home_team_adv_df['division'] == division_id) & (home_team_adv_df['league'] == league_id)]
                div_win_idx = division['win_pct'].idxmax()
                division_winner = division.loc[div_win_idx]['team']
                team_info[team]['wild_card'] = division_winner != team

            
            team1 = list(team_info.keys())[0]
            team2 = list(team_info.keys())[1]
            if game_wins[team1] > game_wins[team2]:
                series_dict['winning_team'] = team1
                series_dict['losing_team'] = team2
            else:
                series_dict['winning_team'] = team2
                series_dict['losing_team'] = team1 
                 
            series_dict['team_with_home_field_advantage'] = determine_home_field_advantage(int(year), series_type, team1, team2, team_info[team1], team_info[team2])
            results['Series'].append(s().dump(series_dict))
            # games.extend(extract_team(series, data_zip))
        
        
        
        # Transform games to useful data
        # for game in games:
        #     parsed_data = transform_game(game, rosters)
        #     results['PlateAppearance'].extend(parsed_data['plate_appearance'])
        #     results['Game'].extend(parsed_data['game'])
        #     results['Run'].extend(parsed_data['run'])
        #     results['BaseRunningEvent'].extend(parsed_data['base_running_event'])
    #print(results['Game'])
    return results, years

results, years = main()
print(years)

# Create SQL Tables
create_tables()

# Load collected data into SQL Database
load_data(results)

    