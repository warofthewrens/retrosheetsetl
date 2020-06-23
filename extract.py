import pandas as pd
import math
from raw_schemas.id import GameId
from raw_schemas.info import Info
from raw_schemas.start import Start
from raw_schemas.play import Play
from raw_schemas.data import Data
from raw_schemas.game import Game
import numpy as np

def extract_roster(team_id):
    '''
    extracts roster from team 'team_id' with the index set to the player_id
    @ param - team_id in the form YYYY + 3 letter team code
    @ return - roster dictionary from player_ids to roster info
    '''
    file_name = team_id[4:] + team_id[0:4] + '.ROS'
    df = pd.read_table('C:\\Users\warre\\Documents\\retrosheetetl\\game_files\\' + file_name, sep = ',', 
                        error_bad_lines=False, names=['player_id', 'player_last_name', 'player_first_name', 'bats', 'throws', 'team', 'pos'])
    df = df.set_index('player_id')
    roster = df.to_dict('index')
    return roster

def extract_roster_team(team_id):
    '''
    extracts roster from team 'team_id' with the index set to the player_id
    @ param - team_id in the form YYYY + 3 letter team code
    @ return - roster dictionary from tuple of (player_ids, team_name) to roster info
    different from extract_roster in that the dict key includes the team_name
    '''
    file_name = team_id[4:] + team_id[0:4] + '.ROS'
    df = pd.read_table('C:\\Users\warre\\Documents\\retrosheetetl\\game_files\\' + file_name, sep = ',', 
                        error_bad_lines=False, names=['player_id', 'player_last_name', 'player_first_name', 'bats', 'throws', 'team', 'pos'])
    df = df.set_index(['player_id', 'team'])
    roster = df.to_dict('index')
    return roster


def extract_team(team_id, league):
    '''
    extract all home games from team team_id and load all data into a Game marshmallow schema
    @param team_id - team_id in the form YYYY + 3 letter team code
    @param league - either 'A' representing american league or 'N' representing national league
    @return games - list of loaded raw_schema.Games 
    '''
    file_name = team_id + '.EV' + league
    df = pd.read_table('C:\\Users\warre\\Documents\\retrosheetetl\\game_files\\' + file_name, sep = ',', 
                        error_bad_lines=False, header=None, names=list(range(7)), converters={4: lambda x: str(x)})
    
    first_game = True
    games = []
    starts = []
    subs = []
    plays = []
    data = []
    game_id = ''
    info = None
    info_dict = {}
    for row in df.itertuples():
        if row[1] == 'id':
            if not first_game == True:
                # info = info_load(info_dict)
                # info_dict = {}
                keys = ['game_id', 'info', 'lineup', 'plays', 'subs', 'data']
                game_dict = dict(zip(keys, [game_id, info_dict, starts, plays, subs, data]))
        
                game = Game().load(game_dict)
                games.append(game)
                starts = []
                subs = []
                plays = []
                data = []
                game_id = ''
                info = None
                info_dict = {}
                
            first_game = False
            game_id = {'game_id': row[2]}
        
        elif row[1] == 'version':
            continue

        elif row[1] == 'info':
            handle_info(row, info_dict)

        elif row[1] == 'start':
            handle_start(row, starts)
            
        elif row[1] == 'play':
            handle_play(row, plays)
            
        elif row[1] == 'sub':
            handle_sub(row, subs, len(plays))

        elif row[1] == 'data':
            handle_data(row, data)
    
    keys = ['game_id', 'info', 'lineup', 'plays', 'subs', 'data']
    game_dict = dict(zip(keys, [game_id, info_dict, starts, plays, subs, data]))

    game = Game().load(game_dict)
    games.append(game)
    return games

def handle_info(row, info_dict):
    '''
    handles all rows which has game info
    '''
    if not row[3] is np.nan:
        info_dict[row[2]] = row[3]

def handle_start(row, starts):
    '''
    handles a row which has starting lineup information
    '''
    keys = ['player_id', 'name', 'is_home', 'bat_pos', 'field_pos']
    start = dict(zip(keys, row[2:7]))
    starts.append(start)

def handle_play(row, plays):
    '''
    handles a row which has play-by-play information
    '''
    keys = ['inning', 'is_home', 'batter_id', 'count', 'pitches', 'play']
    play = dict(zip(keys, row[2:]))
    play['pitches'] = str(play['pitches'])
    plays.append(play)

def handle_sub(row, subs, play_idx):
    '''
    handles a row which has substitution information
    '''
    keys = ['player_id', 'name', 'is_home', 'bat_pos', 'field_pos']
    sub = dict(zip(keys, row[2:7]))
    sub['play_idx'] = play_idx
    subs.append(sub)

def handle_data(row, data):
    '''
    handles rows of data which are exclusively earned run data
    '''
    keys = ['type', 'pitcher_id', 'data']
    datum = dict(zip(keys, row[2:5]))
    data.append(datum)

# extract_team('sl', 's')