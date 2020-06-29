from parsed_schemas.plate_appearance import PlateAppearance
from parsed_schemas.game import Game
from collections import defaultdict
from play_parser import play_parser, runner_moves, get_field_id, add_error
import re  
import copy


#TODO: errors in runner movements are ignored
#TODO: dropped third strikes are confused
#TODO: advance error and then outfield assist wrong

runner_dest = {'B' : 'batter_dest',
               '1' : 'first_dest',
               '2' : 'second_dest',
               '3' : 'third_dest'}

prev_base = {'2' : '1',
             '3' : '2',
             'H' : '3'}

error_dict = {1: 'first_error',
            2: 'second_error',
            3: 'third_error'}

po_dict = {1: 'first_po',
         2: 'second_po',
         3: 'third_po'}


ast_dict = {1: 'first_ast', 
            2: 'second_ast', 
            3: 'third_ast', 
            4: 'fourth_ast', 
            5: 'fifth_ast'}

scorer_dict = {1: 'first_scorer',
                2: 'second_scorer',
                3: 'third_scorer',
                4: 'fourth_scorer'}

runner_event = {
    '1': 'first_runner_event',
    '2': 'second_runner_event',
    '3': 'third_runner_event',
}

at_bat = set([2, 3, 18, 19, 20, 21, 22, 23])
non_plate_app = set([4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
pinch_player = set([11, 12])


def get_batter_lineup(batter, state):
    '''
    given a batter return his lineup position
    '''
    return state['lineups']['players'][batter][0]

def get_batter_field(batter, state):
    '''
    given a batter return his field position
    '''
    return state['lineups']['players'][batter][1]
    

def modifier(modifier, play, runners_out, state):
    '''
    extract relevant information from the modifier
    @param modifier - a single string containing a play modifier
    @param play - a dictionary containing play info
    @param runners_out - list of runners who were out after parsing play
    '''
    i = 0

    # Hit location
    location = re.search('[1-9]', modifier)
    if location:
        play['hit_loc'] = location.group(0)
    

    if len(modifier) > 0:
        # Bunt
        if modifier[0] == 'B' and modifier not in set(['BINT', 'BOOT', 'BR']):
            play['bunt_flag'] = True
            play['ball_type'] = modifier[1]

        # Batted ball type
        if modifier[0] in set(['G', 'F', 'L', 'P']) and modifier not in set(['FINT', 'FL', 'FO', 'PASS']):
            play['ball_type'] = modifier[0]
    
    # Foul ball
    if modifier == 'FL':
        play['foul_flag'] = True
    
    # Force out
    if modifier == 'FO':
        play['batter_dest'] = '1'
        if len(runners_out) > 0:
            state['responsible_pitchers']['B'] = state['responsible_pitchers'][str(runners_out[0])]

    # Sac fly
    if modifier == 'SF':
        play['sac_fly'] = True
    
    # Sac bunt
    if modifier == 'SH':
        play['sac_bunt'] = True
    
    # Error
    if modifier == 'E':
        add_error(modifier[1], play, state)
    


#TODO: giant WIP
def expand_play(play, state, rows):
    '''
    takes an extracted play and transforms the shorthand play descriptions into 
    more useful information.
    @param play - dictionary containing play info
    '''

    # separate the runner movements
    moves = play['play'].split('.')

    # split modifiers
    play_arr = moves[0].split('/')
    i = 0

    # iterate through elements in the modifier array
    for item in play_arr:
        # play description
        if (i == 0):
            runners_out, outs_this_play_ed = play_parser(item, play, state, False, rows)
        
        # modifiers
        else:
            modifier(item, play, runners_out, state)
        i += 1
    
    # Runner moves
    if (len(moves) > 1):
        outs_this_play_rm = runner_moves(moves[1], play, runners_out, state, rows)
    # Takes care of updating runners even if there are no explicit runner movements
    else:
        outs_this_play_rm = runner_moves(None, play, runners_out, state, rows)
    

    # calculate outs
    outs_this_play = outs_this_play_ed + outs_this_play_rm
    play['outs_on_play'] = outs_this_play
    return play


def build_lineups(starts):
    '''
    use starts to build the starting lineups and starting field positions 
    @param starts - list of dictionaries containing the starting lineups
    '''

    lineups = {'home_lineup': {},
               'home_field_pos': {},
               'away_lineup': {},
               'away_field_pos': {},
               'players': {}}
    # build home and away lineups
    for start in starts:
        if (start['is_home']):
            lineups['home_lineup'][start['bat_pos']] = start['player_id']
            lineups['home_field_pos'][start['field_pos']] = start['player_id']
            if start['field_pos'] == 1:
                lineups['home_field_pos']['sp'] = start['player_id']
            
        else:
            lineups['away_lineup'][start['bat_pos']] = start['player_id']
            lineups['away_field_pos'][start['field_pos']] = start['player_id']
            if start['field_pos'] == 1:
                lineups['away_field_pos']['sp'] = start['player_id']
        lineups['players'][start['player_id']] = [start['bat_pos'], start['field_pos']]
    # make reversed dictionary for player lookup        
    return lineups

def transform_plays(plays, subs, rows, state):
    '''
    wrapper around all other methods to collate all play data into the game
    @param plays - a list of plays for this game
    @param subs - a list of substitutions for this game
    @param rows - the rows of the table which will be loaded
    @param state - the game state
    '''
    # print('transforming')
    print(state['game_id'])
    pa_id = 0
    
    state['outs'] = 0
    state['home_runs'] = 0
    state['away_runs'] = 0
    state['runners_before'] = defaultdict(str)
    state['po'] = 0
    state['ast'] = 0
    state['run_id'] = 0
    state['br_event_id'] = 0
    state['responsible_pitchers'] = defaultdict(str)
    state['runs_allowed'] = defaultdict(int)

    sub_spot = {}
    for sub in subs:
        sub_spot[int(sub['play_idx'])] = sub
    
    play_idx = 0
    for play in plays:
        sub = sub_spot.get(play_idx)
        if sub:
            if sub['is_home']:
                state['lineups']['home_lineup'][sub['bat_pos']] = sub['player_id']
                state['lineups']['home_field_pos'][sub['field_pos']] = sub['player_id']
            else:
                state['lineups']['away_lineup'][sub['bat_pos']] = sub['player_id']
                state['lineups']['away_field_pos'][sub['field_pos']] = sub['player_id']
            state['lineups']['players'][sub['player_id']] = [sub['bat_pos'], sub['field_pos']]
            
        if not play['play'] == 'NP':
            play['pa_id'] = pa_id
            play['sac_fly'] = False
            play['sac_bunt'] = False
            play['wp'] = False
            play['pb'] = False
            play['bunt_flag'] = False
            play['foul_flag'] = False
            play['batter_dest'] = ''
            play['first_dest'] = ''
            play['second_dest'] = ''
            play['third_dest'] = ''
            play['rbi'] = 0
            play['runs_on_play'] = 0
            play['hit_val'] = 0
            play['num_errors'] = 0
            play['fielder_id'] = ''
            play['field_pos'] = get_batter_field(play['batter_id'], state)
            if (play['field_pos'] not in pinch_player):
                play['lineup_pos'] = get_batter_lineup(play['batter_id'], state)
            play['pitcher_id'] = get_field_id(play['is_home'], 1, state)
            state['responsible_pitchers']['B'] = play['pitcher_id']
            if play['is_home']:
                play['pitcher_hand'] = state['roster'][state['away_team']][play['pitcher_id']]['throws']
                play['batter_hand'] = state['roster'][state['home_team']][play['batter_id']]['bats']
            else:
                play['pitcher_hand'] = state['roster'][state['home_team']][play['pitcher_id']]['throws']
                play['batter_hand'] = state['roster'][state['away_team']][play['batter_id']]['bats']
            if play['batter_hand'] == 'B':
                if play['pitcher_hand'] == 'L':
                    play['batter_hand'] = 'R'
                else:
                    play['batter_hand'] = 'L'
            expand_play(play, state, rows)
            play['ab_flag'] = ((play['event_type'] in at_bat) and (play['sac_fly'] == False) and (play['sac_bunt'] == False))
            play['pa_flag'] = play['event_type'] not in non_plate_app
            p_a = PlateAppearance(context=state).dump(play)
            rows['plate_appearance'].append(p_a)
            state['inning'] = play['inning']
            pa_id += 1
        
        play_idx+=1
    
    return state


def transform_game(game, roster):
    '''
    given a list of games and a roster build the tables to be loaded into the database
    @param game - a list of games to be transformed
    @param roster - rosters for all relevant players
    '''
    rows = {table: [] for table in ['plate_appearance', 'game', 'run', 'base_running_event']}
    lineups = build_lineups(game['lineup'])
    context = {
        'game_id': game['game_id'], 'date': game['info']['date'], 
        'lineups': lineups, 'roster': roster, 'home_team': game['info']['home_team'],
        'away_team': game['info']['visiting_team']
        }
    state = transform_plays(game['plays'], game['subs'], rows, context)
    game = {
        'lineup': game['lineup'],
        'data': game['data'],
        'info': game['info'],
        'game_id': game['game_id']
    }
    new_game = Game(context=state).dump(game)
    rows['game'].append(new_game)
    return rows