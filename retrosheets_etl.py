# retrosheets_etl.py
from extract import extract_team
from transform import transform_game

def main():
    game = extract_team('2019OAK', 'A')
    transform_game(game)
    
    # print(info)
    # print(starts)
    # for play in plays:
    #     print(play)
    
    # print(data)

main()