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
    '''
    Extract, Transform, and Load playoff data from Retrosheet.
    '''
    years = []
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
    results = {'PlateAppearance': [], 'Game': [], 'Run': [], 'BaseRunningEvent': []}

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

        # Extract games from every identified team
        for series in game_files:
            games.extend(extract_team(series, data_zip))
        
        # Extract rosters from every team no matter what
        for team in roster_files:
            rosters.update(extract_roster(team, data_zip))
        
        # Transform games to useful data
        for game in games:
            parsed_data = transform_game(game, rosters)
            results['PlateAppearance'].extend(parsed_data['plate_appearance'])
            results['Game'].extend(parsed_data['game'])
            results['Run'].extend(parsed_data['run'])
            results['BaseRunningEvent'].extend(parsed_data['base_running_event'])
    
    return results, years

results, years = main()
print(years)

# Create SQL Tables
create_tables()

# Load collected data into SQL Database
load_data(results)

    