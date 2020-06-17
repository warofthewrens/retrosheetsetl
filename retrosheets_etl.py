# retrosheets_etl.py
from extract import extract_team
from transform import transform_game 
from load import create_tables, load_data
import time

def main():
    games = extract_team('2019PHI', 'N')
    results = {'PlateAppearance': []}
    for game in games:
        parsed_data = transform_game(game)
        results['PlateAppearance'].extend(parsed_data['plate_appearance'])
    create_tables()
    load_data(results)

start = time.time()
main()
end = time.time()
print(end - start)
