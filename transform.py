from parsed_schemas.plate_appearance import PlateAppearance
from collections import defaultdict
import re  

home_lineup = {}
away_lineup = {}

home_lineup_rev = {}
away_lineup_rev = {}

home_field_pos = {}
away_field_pos = {}

home_field_rev = {}
away_field_rev = {}

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

def get_lineup_id(is_home, pos, state):
    if is_home:
        return state['lineups']['home_lineup'][pos]
    else:
        return state['lineups']['away_lineup'][pos]

def get_field_id(is_home, pos, state):
    pos = int(pos)
    if is_home:
        return state['lineups']['away_field_pos'][pos]
    else:
        return state['lineups']['home_field_pos'][pos]

def get_batter_lineup(batter, state):
    return state['lineups']['players'][batter][0]

def get_batter_field(batter, state):
    return state['lineups']['players'][batter][1]

def add_error(error_player, play, state):
    play['num_errors'] += 1
    play[error_dict[play['num_errors']]] = get_field_id(play['is_home'], error_player, state)

def add_po(po_player, play, state):
    state['po'] += 1
    play[po_dict[state['po']]] = get_field_id(play['is_home'], po_player, state)

def add_ast(ast_player, play, state):
    state['ast'] += 1
    play[ast_dict[state['ast']]] = get_field_id(play['is_home'], ast_player, state)


def score_run(move, play, state):
    info_index = 0
    no_rbi = False
    fielding_info = move[move.find('(')+ 1: move.find(')')]
    while info_index != -1:
        info_index = fielding_info.find('(')
        
        # if fielding_info == 'UR':
        #     print('unearned run')
        # elif fielding_info == 'TUR':
        #     print('team unearned run')
             
        if fielding_info == 'NR' or fielding_info == 'NORBI':
            no_rbi = True

        elif fielding_info == 'WP':
            play['wp'] = True

        elif fielding_info == 'PB':
            play['pb'] = True
        fielding_info = fielding_info[fielding_info.find('(')+ 1: fielding_info.find(')')]  

    if play['is_home']:
        state['home_runs'] += 1
    else:
        state['away_runs'] += 1
    if not no_rbi:
        play['rbi'] += 1

def stolen_bases(sbs, play, state):
    not_moved = set([1, 2, 3])

    for sb in sbs:
        base = sb[2]
        if base != 'H':
            runner = state['runners_before'][int(base) - 1]
            play[runner_dest[prev_base[base]]] = base
        else:
            score_run(sb, play, state)
        not_moved.remove(int(prev_base[base]))
    
    for not_move in not_moved:
        if state['runners_before'][not_move] != '':
            play[runner_dest[str(not_move)]] = str(not_move)
    
def runner_moves(moves, play, runners_out, state):
    # global runners_dest
    
    not_moved = set([1, 2, 3])
    outs_this_play = 0
    if moves:
        moves = moves.split(';')
        for move in moves:
            if move[0] != 'B':
                runner = state['runners_before'][int(move[0])]
                not_moved.remove(int(move[0]))
            else:
                runner = play['batter_id']
                
            if move[1] != 'X':
                if move[2] == 'H':
                    score_run(move, play, state)
                play[runner_dest[move[0]]] = move[2]
            else:
                fielding_info = move[move.find('(')+ 1: move.find(')')]
                outs_this_play += event_description(fielding_info, play, state, True)[1]
                if outs_this_play > 0:
                    play[runner_dest[move[0]]] = 'O'
                else:
                    play[runner_dest[move[0]]] = move[2]
            
    
    for not_move in not_moved:
        if not_move not in runners_out:
            if state['runners_before'][not_move] != '' and play[runner_dest[str(not_move)]]== '':
                play[runner_dest[str(not_move)]] = str(not_move)
        else:
            play[runner_dest[str(not_move)]] = 'O'
    
    return outs_this_play


def event_description(item, play, state, second_event):
    '''
    parses the play string into relevant pieces
    @param item - play string
    @param play - dictionary containing play info
    @param second_event - boolean determining if it's the second event this play
    '''
    event_type = 0
    outs_this_play = 0
    runners_out = []

    if item[0].isdigit():
        i = 0
        po_ers = []
        last_fielder = item[i]
        outs_this_play = 0
        po_err = {
            1: False,
            2: False,
            3: False,
        }
        po = {
            1: [],
            2: [],
            3: []
        }

        while len(item) > i:
            if item[i].isdigit():
                po[outs_this_play + 1].append(item[i])
                if(i == 0):
                    play['hit_loc'] = int(item[i])
            elif item[i] == '(':
                if item[i + 1] != 'B':
                    runners_out.append(int(item[i + 1]))
                    play[runner_dest[item[i + 1]]] = 'O'
                    outs_this_play += 1
                    if len(item) > i + 3:
                        po[outs_this_play + 1].append(item[i - 1])
                else:
                    outs_this_play += 1
                    if len(item) > i + 3:
                        po[outs_this_play + 1].append(item[i - 1])
                    play['batter_dest'] = 'O'
                i += 2
            elif item[i] == 'E':
                po_err[outs_this_play + 1] = True
            i+=1
        
        pos = po.items()
        
        for out, fielders in pos:
            field = 0
            for fielder in fielders:
                if field == (len(fielders) - 1):
                    if not po_err[out]:
                        add_po(fielder, play, state)
                        event_type = 2
                    else:
                        add_error(fielder, play, state)
                        outs_this_play -= 1
                        event_type = 18
                else:
                    add_ast(fielder, play, state)
                field+=1
        
        
    
    if re.match('^SB[23H]', item):
        sbs = item.split(';')
        stolen_bases(sbs, play, state)
        event_type = 4
    
    elif item[0:2] == 'DI':
        event_type = 5
    
    elif item[0:3] == 'DGR':
        play['hit_val'] = 2
        event_type = 21
        play['batter_dest'] = '2'

    elif item[0] in set(['S', 'D', 'T']) and not re.match('^SB[23H]', item):
        if item[0] == 'S':
            play['hit_val'] = 1
            event_type = 20
            play['batter_dest'] = '1'

        elif item[0] == 'D':
            play['hit_val'] = 2
            event_type = 21
            play['batter_dest'] = '2'

        elif item[0] == 'T':
            play['hit_val'] = 3
            event_type = 22
            play['batter_dest'] = '3'
        if len(item) > 1:
            play['fielder_id'] = get_field_id(play['is_home'], item[1], state)
            play['hit_loc'] = item[1]
    
    elif item[0] == 'E' or item[0:3] == 'FLE':
        if item[0] == 'E':
            event_type = 18
            play['batter_dest'] = '1'
            play['fielder_id'] = get_field_id(play['is_home'], item[1], state)
            play['hit_loc'] = item[1]
            error_player = item[1]
        else:
            event_type = 13
            play['fielder_id'] = get_field_id(play['is_home'], item[3], state)
            play['hit_loc'] = item[3]
            error_player = item[3]
        
        add_error(error_player, play, state)
        
    
    elif item[0] == 'K':
        second_event_arr = item.split('+')
        if len(second_event_arr) > 1:
            new_runners_out, new_outs_this_play = event_description(second_event_arr[1], play, state, True)
            runners_out.extend(new_runners_out)
            outs_this_play += new_outs_this_play
            if 'B-' in play['play']:
                outs_this_play -= 1
        event_type = 3
        outs_this_play += 1
        play['batter_dest'] = 'O'
    
    elif item[0:2] == 'WP':
        play['wp'] = True
        event_type = 9

    elif item[0:2] == 'PB':
        play['pb'] = True
        event_type = 10

    elif item[0] == 'W':
        second_event_arr = item.split('+')
        if len(second_event_arr) > 1:
            new_runners_out, new_outs_this_play = event_description(second_event_arr[1], play, state, True)
            runners_out.extend(new_runners_out)
            outs_this_play += new_outs_this_play
        event_type = 14
        play['batter_dest'] = '1'

    
    
    elif item[0:2] == 'OA':
        event_type = 12
    
    elif item[0:2] == 'PO':
        fielding_info = item[item.find('(')+ 1: item.find(')')]
        if not 'E' in fielding_info:
            event_type = 8
            if item[2] == 'C':
                runners_out.append(int(prev_base[item[4]]))
            else:
                runners_out.append(int(item[2]))
            outs_this_play += 1
            add_po(fielding_info[1], play, state)
            add_ast(fielding_info[0], play, state)
        else:
            event_type = 7
            error_player = item[item.find('E') + 1]
            add_error(error_player, play, state)
    
    elif item[0] == 'I' or item[0:] == 'IW':
        second_event_arr = item.split('+')
        if len(second_event_arr) > 1:
            new_runners_out, new_outs_this_play = event_description(second_event_arr[1], play, state, True)
            runners_out.extend(new_runners_out)
            outs_this_play += new_outs_this_play

        event_type = 15
        play['batter_dest'] = '1'
    
    elif item == 'HP':
        event_type = 16 
        play['batter_dest'] = '1'

    elif item[0] == 'H' or item[0:] == 'HR':
        event_type = 23
        play['hit_val'] = 4
        play['batter_dest'] = 'H'
        if 'B-' not in play['play']:
            score_run('', play, state)
    
    elif item[0:2] == 'BK':
        event_type = 11 

    elif item[0:2] == 'CS':
        event_type = 6
        fielding_info = item[item.find('(')+ 1: item.find(')')]
        if not 'E' in fielding_info:
            runners_out.append(int(prev_base[item[2]]))
            play[runner_dest[prev_base[item[2]]]] = 'O'
            i = 0
            for c in fielding_info:
                if i == len(fielding_info) - 1:
                    add_po(c, play, state)
                else:
                    add_ast(c, play, state)
                i+=1
            outs_this_play += 1
        else:
            error_player = item[item.find('E') + 1]
            add_error(error_player, play, state)
            print('Caught stealing error')

    elif item[0] == 'C' and item != 'CS':
        event_type = 17
        play['batter_dest'] = '1'

    #TODO: fix fielder's choice logic
    elif len(item) > 1:
        if item[0:2] == 'FC':
            event_type = 19
            play['batter_dest'] = '1'
            
    if not second_event:
        play['event_type'] = event_type
    return runners_out, outs_this_play
    

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
def expand_play(play, state):
    '''
    takes an extracted play and transforms the shorthand play descriptions into 
    more useful information.
    @param play - dictionary containing play info
    '''

    # separate the runner movements
    moves = play['play'].split('.')

    # split modifiers
    play_arr = moves[0].split('/')

    outs_this_play = 0
    outs_this_play_rm = 0
    i = 0

    # iterate through elements in the modifier array
    for item in play_arr:
        # play description
        if (i == 0):
            runners_out, outs_this_play_ed = event_description(item, play, state, False)
        
        # modifiers
        else:
            modifier(item, play, runners_out, state)
        i += 1
    
    # Runner moves
    if (len(moves) > 1):
        outs_this_play_rm = runner_moves(moves[1], play, runners_out, state)
    # Takes care of updating runners even if there are no explicit runner movements
    else:
        outs_this_play_rm = runner_moves(None, play, runners_out, state)
    

    # calculate outs
    outs_this_play = outs_this_play_ed + outs_this_play_rm
    play['outs_on_play'] = outs_this_play
    return play

def print_lineups():
    '''
    prints current lineups
    '''
    print(home_lineup)
    print(away_lineup)
    print(home_field_pos)
    print(away_field_pos)

def build_lineups(starts):
    '''
    use starts to build the starting lineups and starting field positions 
    @param starts - list of dictionaries containing the starting lineups
    '''
    global away_lineup_rev, home_lineup_rev, away_field_rev, home_field_rev

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
            
        else:
            lineups['away_lineup'][start['bat_pos']] = start['player_id']
            lineups['away_field_pos'][start['field_pos']] = start['player_id']
        lineups['players'][start['player_id']] = [start['bat_pos'], start['field_pos']]
    # make reversed dictionary for player lookup        
    return lineups

def transform_plays(plays, subs, rows, state):
    pa_id = 0
    
    state['outs'] = 0
    state['home_runs'] = 0
    state['away_runs'] = 0
    state['runners_before'] = defaultdict(str)
    state['po'] = 0
    state['ast'] = 0

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
            play['field_pos'] = get_batter_field(play['batter_id'], state)
            play['bunt_flag'] = False
            play['foul_flag'] = False
            play['batter_dest'] = ''
            play['first_dest'] = ''
            play['second_dest'] = ''
            play['third_dest'] = ''
            play['rbi'] = 0
            play['hit_val'] = 0
            play['num_errors'] = 0
            play['fielder_id'] = ''
            if (play['field_pos'] not in set([11, 12])):
                play['lineup_pos'] = get_batter_lineup(play['batter_id'], state)
            expand_play(play, state)
            play['pitcher_id'] = get_field_id(play['is_home'], 1, state)
            play['ab_flag'] = ((play['event_type'] in set([2, 3, 18, 19, 20, 21, 22, 23])) and (play['sac_fly'] == False) and (play['sac_bunt'] == False))
            p_a = PlateAppearance(context=state).dump(play)
            rows['plate_appearance'].append(p_a)
            pa_id += 1
        
        play_idx+=1
    return


def transform_game(game):
    rows = {table: [] for table in ['plate_appearance']}
    
    lineups = build_lineups(game['lineup'])
    context = {'game_id': game['game_id'], 'date': game['info']['date'], 'lineups': lineups}
    transform_plays(game['plays'], game['subs'], rows, context)
    return rows