from extract import extract_playoff_data_by_year, extract_team, extract_roster
from transform import transform_game 
from playoff_load import create_tables, load_data
# from extract_game_data import etl_player_data
# from etl_team_data import etl_team_data
from os import walk
import concurrent.futures
import time
import sys
import getopt
import shutil
def main():
    years = []
    teams = set([])
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'y:t:')
    except getopt.GetoptError as err:
        print(str(err))
        sys.exit(2)
    print(opts, args)
    for o, a in opts:
        if o == '-y':
            if int(a) < 1920 or int(a) > 2019:
                raise Exception('invalid year')
            else:
                years.append(a)
        if o == '-t':
            teams.add(a)
    results = {'PlateAppearance': [], 'Game': [], 'Run': [], 'BaseRunningEvent': []}
    for year in years:
        games = []
        rosters = {}
        roster_files = set([])
        game_files = set([])
        data_zip, data_td = extract_playoff_data_by_year(year)
        
        f = []
        for (dirpath, dirnames, filenames) in walk(data_td):
            f.extend(filenames)
            break
        shutil.rmtree(data_td)
        print(f)
        if len(teams) == 0:
            for team_file in f:
                if team_file[-4:] == '.ROS':
                    print('roster', team_file)
                    roster_files.add(team_file)
                elif team_file[-4:] == '.EVE':
                    print('game', team_file)
                    game_files.add(team_file)
                else:
                    print(team_file)
        else:
            for team_file in f:
                if team_file[-4:] == '.ROS':
                    roster_files.add(team_file)
                if team_file[4:7] in teams:
                    game_files.add(team_file)
    
        for series in game_files:
            games.extend(extract_team(series, data_zip))
        
        for team in roster_files:
            rosters.update(extract_roster(team, data_zip))
        
        for game in games:
            parsed_data = transform_game(game, rosters)
            # parsed_data = parsed_data.result()
            # print(parsed_data['game'][0]['home_team'])
            results['PlateAppearance'].extend(parsed_data['plate_appearance'])
            results['Game'].extend(parsed_data['game'])
            results['Run'].extend(parsed_data['run'])
            results['BaseRunningEvent'].extend(parsed_data['base_running_event'])
    return results, years

results, years = main()
print(years)
create_tables()
load_data(results)
    
    # print('starting_player')
    # etl_player_data(year)
    # print('done player')
    # print('working on team')
    # etl_team_data(year)
    # print('done team')
    