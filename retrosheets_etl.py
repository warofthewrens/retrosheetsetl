# retrosheets_etl.py
from extract import extract_team
from transform import transform_game
import time

def main():
    game = extract_team('2019ATL', 'N')
    transform_game(game)
    
    # print(info)
    # print(starts)
    # for play in plays:
    #     print(play)
    
    # print(data)

start = time.time()
main()
end = time.time()
print(end - start)
