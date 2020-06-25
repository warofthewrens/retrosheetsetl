# retrosheets_etl.py
from extract import extract_team, extract_roster, extract_rosters, extract_game_data_by_year
from transform import transform_game 
from load import create_tables, load_data
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
    games = []
    year = '2019'
    nl_teams = set([])
    al_teams = set([])
    
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
                if a in nl_team_set:
                    nl_teams.add(a)
                elif a in al_team_set:
                    al_teams.add(a)
    data_zip, data_td = extract_game_data_by_year(year)
    print(year, nl_teams, al_teams)
    if len(nl_teams) == 0 and len(al_teams) == 0:
        nl_teams = nl_team_set
        al_teams = al_team_set
    for nl in nl_teams:
        games.extend(extract_team(year + nl, 'N', data_zip))
    
    for al in al_teams:
        games.extend(extract_team(year + al, 'A', data_zip))
    
    
    # games.extend(extract_team('2019SLN', 'N'), data_zip)
    get_rosters(year, data_zip)
    results = {'PlateAppearance': [], 'Game': [], 'Run': [], 'BaseRunningEvent': []}
    for game in games:
        parsed_data = transform_game(game, rosters)
        results['PlateAppearance'].extend(parsed_data['plate_appearance'])
        results['Game'].extend(parsed_data['game'])
        results['Run'].extend(parsed_data['run'])
        results['BaseRunningEvent'].extend(parsed_data['base_running_event'])
    create_tables()
    load_data(results)
    shutil.rmtree(data_td)

def get_rosters(year, data_zip):
    for team in team_set:
        rosters.update(extract_roster(year + team, data_zip))
    return rosters


start = time.time()
# get_rosters()
# for team in rosters.keys():
#     print(rosters[team])
main()

end = time.time()
print(end - start)
