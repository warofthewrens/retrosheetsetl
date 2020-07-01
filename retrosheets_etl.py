# retrosheets_etl.py
from extract import extract_team, extract_roster, extract_rosters, extract_game_data_by_year
from transform import transform_game 
from load import create_tables, load_data
from extract_game_data import etl_player_data
from os import walk
import concurrent.futures
import time
import sys
import getopt
import shutil

team_set = set(['ANA', 'ARI', 'ATL', 'BAL', 'BOS', 'CHA', 'CHN', 'CIN', 'CLE', 'COL', 'DET', 'HOU',
            'KCA', 'LAN', 'MIA', 'MIL', 'MIN', 'NYA', 'NYN', 'OAK', 'PHI', 'PIT', 'SEA',
            'SFN', 'SLN', 'SDN', 'TBA', 'TEX', 'TOR', 'WAS'])

nl_team_set = set(['ARI', 'ATL', 'CHN', 'CIN', 'COL', 'LAN', 'MIA', 
                'MIL', 'NYN', 'PHI', 'PIT', 'SFN', 'SLN', 'SDN', 'WAS'])

al_team_set = set(['ANA', 'BAL', 'BOS', 'CHA', 'CLE', 'DET', 'HOU', 
                'KCA', 'MIN', 'NYA', 'OAK', 'SEA', 'TBA', 'TEX', 'TOR'])

rosters = {}

def main():
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        global rosters
        games = []
        year = '2019'
        teams = set([])
        roster_files = set([])
        game_files = set([])
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
                    year = a
            if o == '-t':
                if a not in team_set:
                    raise Exception('invalid_team')
                else:
                    teams.add(a)
        
        data_zip, data_td = extract_game_data_by_year(year)
        
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
                elif team_file[-4:] == '.EVN' or team_file[-4:] == '.EVA':
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
        
        for team in game_files:
            games.extend(extract_team(team, data_zip))
        
        for team in roster_files:
            rosters.update(extract_roster(team, data_zip))
        # print(rosters)
        # games.extend(extract_team('2019SLN', 'N'), data_zip)
        # get_rosters(year, data_zip)
        results = {'PlateAppearance': [], 'Game': [], 'Run': [], 'BaseRunningEvent': []}
        game_results = []
        new_result = [executor.submit(process_game, game, rosters) for game in games]
        for parsed_data in concurrent.futures.as_completed(new_result):
            parsed_data = parsed_data.result()
            # print(parsed_data['game'][0]['home_team'])
            results['PlateAppearance'].extend(parsed_data['plate_appearance'])
            results['Game'].extend(parsed_data['game'])
            results['Run'].extend(parsed_data['run'])
            results['BaseRunningEvent'].extend(parsed_data['base_running_event'])
        executor.shutdown(wait=True) 
    return results, year
    
        
            #game_results = executor.map(process_game, games)
        
    # for game in games:
    #     parsed_data = transform_game(game, rosters)
        
    

def process_game(game, rosters):
    # print(game['game_id'])
    return transform_game(game, rosters)

def get_rosters(year, data_zip):
    for team in team_set:
        rosters.update(extract_roster(year + team, data_zip))
    return rosters


start = time.time()
# get_rosters()
# for team in rosters.keys():
#     print(rosters[team])

if __name__ == '__main__':
    results,year = main()
    print(year)
    create_tables()
    load_data(results)
    etl_player_data(year)

end = time.time()
print(end - start)
