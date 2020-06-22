# retrosheets_etl.py
from extract import extract_team, extract_roster
from transform import transform_game 
from load import create_tables, load_data
import time

team_set = set(['ANA', 'ARI', 'ATL', 'BAL', 'BOS', 'CHA', 'CHN', 'CIN', 'CLE', 'COL', 'DET', 'HOU',
            'KCA', 'LAN', 'MIA', 'MIL', 'MIN', 'NYA', 'NYN', 'OAK', 'PHI', 'PIT', 'SEA',
            'SFN', 'SLN', 'SDN', 'TBA', 'TEX', 'TOR', 'WAS'])

nl_teams = set(['ARI', 'ATL', 'CHN', 'CIN', 'COL', 'LAN', 'MIA', 
                'MIL', 'NYN', 'PHI', 'PIT', 'SFN', 'SLN', 'SDN', 'WAS'])

al_teams = set(['ANA', 'BAL', 'BOS', 'CHA', 'CLE', 'DET', 'HOU', 
                'KCA', 'MIN', 'NYA', 'OAK', 'SEA', 'TBA', 'TEX', 'TOR'])

rosters = {}

def main():
    games = []
    # for nl in nl_teams:
    #     games.extend(extract_team('2019' + nl, 'N'))
    
    # for al in al_teams:
    #     games.extend(extract_team('2019' + al, 'A'))
    games.extend(extract_team('2019SLN', 'N'))
    results = {'PlateAppearance': [], 'Game': []}
    for game in games:
        parsed_data = transform_game(game, rosters)
        results['PlateAppearance'].extend(parsed_data['plate_appearance'])
        results['Game'].extend(parsed_data['game'])
    create_tables()
    load_data(results)

def get_rosters():
    for team in team_set:
        rosters.update(extract_roster('2019' + team))
    return rosters


start = time.time()
get_rosters()
main()
end = time.time()
print(end - start)
