import pandas as pd
import math
from raw_schemas.id import GameId
from raw_schemas.info import Info
from raw_schemas.start import Start
from raw_schemas.play import Play
from raw_schemas.data import Data
from raw_schemas.game import Game
import numpy as np

def extract_team(team_id, league):
    file_name = team_id + '.EV' + league
    df = pd.read_table('C:\\Users\warre\\Documents\\retrosheetetl\\game_files\\' + file_name, sep = ',', error_bad_lines=False, header=None, names=list(range(7)), converters={4: lambda x: str(x)})
    
    first_game = True
    starts = []
    subs = []
    plays = []
    data = []

    info = None
    info_dict = {}
    for row in df.itertuples():
        if row[1] == 'id':
            if not first_game == True:
                # info = info_load(info_dict)
                # info_dict = {}
                break
            first_game = False
            game_id = {'game_id': row[2]}
        
        elif row[1] == 'version':
            continue

        elif row[1] == 'info':
            if not row[3] is np.nan:
                info_dict[row[2]] = row[3]

        elif row[1] == 'start':
            keys = ['player_id', 'name', 'is_away', 'bat_pos', 'field_pos']
            start = dict(zip(keys, row[2:7]))
            # start = Start().load(start_dict)
            starts.append(start)

        elif row[1] == 'play':
            keys = ['inning', 'is_away', 'batter_id', 'count', 'pitches', 'play']
            play = dict(zip(keys, row[2:]))
            play['pitches'] = str(play['pitches'])
            # play = Play().load(play_dict)
            plays.append(play)
        
        elif row[1] == 'sub':
            keys = ['player_id', 'name', 'is_away', 'bat_pos', 'field_pos']
            sub = dict(zip(keys, row[2:7]))
            sub['play_idx'] = len(plays)
            subs.append(sub)

        elif row[1] == 'data':
            keys = ['type', 'pitcher_id', 'data']
            datum = dict(zip(keys, row[2:5]))
            # datum = Data().load(data_dict)
            data.append(datum)
    
    keys = ['game_id', 'info', 'lineup', 'plays', 'subs', 'data']
    game_dict = dict(zip(keys, [game_id, info_dict, starts, plays, subs, data]))
    game = Game().load(game_dict)
    return game

def info_load(info):
    return Info().load(info)

# extract_team('sl', 's')