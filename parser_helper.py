import copy
from play_parser import play_parser

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
    '''
    returns the player id for position, pos in lineup of appropriate team
    @param is_home - boolean for if batter is_home
    @param pos - lineup position to lookup
    @param state - game state which includes current lineups
    '''
    if is_home:
        return state['lineups']['home_lineup'][pos]
    else:
        return state['lineups']['away_lineup'][pos]

def get_field_id(is_home, pos, state):
    '''
    returns the player id for position, pos in lineup of appropriate team
    @param is_home - boolean for if batter is_home
    @param pos - lineup position to lookup
    @param state - game state which includes current field positions
    '''
    pos = int(pos)
    if is_home:
        return state['lineups']['away_field_pos'][pos]
    else:
        return state['lineups']['home_field_pos'][pos]

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

def add_error(error_player, play, state):
    '''
    add an error to the current plays state
    @param error_player - player commiting error
    @param play - the play to be loaded into the table
    @param state - the state of the game
    '''
    play['num_errors'] += 1
    play[error_dict[play['num_errors']]] = get_field_id(play['is_home'], error_player, state)

def add_po(po_player, play, state):
    '''
    add a putout to the current plays state
    @param po_player - player making put out
    @param play - the play to be loaded into the table
    @param state - the state of the game
    '''
    state['po'] += 1
    play[po_dict[state['po']]] = get_field_id(play['is_home'], po_player, state)

def add_ast(ast_player, play, state):
    '''
    add an assist to the current plays state
    @param est_player - player making assist
    @param play - the play to be loaded into the table
    @param state - the state of the game
    '''
    state['ast'] += 1
    play[ast_dict[state['ast']]] = get_field_id(play['is_home'], ast_player, state)

def run_modifiers(move, play):
    '''
    Handle all modifiers such as (NR) meaning no rbi attached to a run scored
    '''
    no_rbi = False
    modifiers = copy.copy(move)
    info_index = modifiers.find('(')
    end_index = modifiers.find(')')
    fielding_info = modifiers[info_index + 1: end_index]
    modifiers = modifiers[end_index + 1:]
    while info_index != -1:
        # print(move[])
        if fielding_info == 'NR' or fielding_info == 'NORBI':
            no_rbi = True

        elif fielding_info == 'WP':
            play['wp'] = True

        elif fielding_info == 'PB':
            play['pb'] = True
        
        info_index = modifiers.find('(')
        end_index = modifiers.find(')')
        fielding_info = modifiers[info_index + 1: end_index]
        modifiers = modifiers[end_index + 1:]
    return no_rbi

def score_run(move, play, state):
    '''
    score a run for the appropriate team and handle all required upkeep
    @param move - the specific runner move which scored the run
    @param play - the play to be loaded into the table
    @param state - the game state
    '''
    info_index = 0
    no_rbi = run_modifiers(move, play)
    state['runs_allowed'][state['responsible_pitchers'][move[0]]]+=1
    play['runs_on_play'] += 1
    if move[0] != 'B':
        play[scorer_dict[play['runs_on_play']]] = state['runners_before'][int(move[0])]
    else:
        play[scorer_dict[play['runs_on_play']]] = play['batter_id']
    if not no_rbi:
        play['rbi'] += 1

def stolen_bases(sbs, play, state):
    '''
    handler for stolen bases
    @param sbs - a list of stolen bases
    @param play - the play to be loaded into the table
    @param state - the game state
    '''
    not_moved = set([1, 2, 3])
    for sb in sbs:
        base = sb[2]
        if base != 'H':
            runner = state['runners_before'][int(base) - 1]
            play[runner_dest[prev_base[base]]] = base
        else:
            score_run('3-H(NR)', play, state)
            play['third_dest'] = 'H'
        not_moved.remove(int(prev_base[base]))
        play[runner_event[prev_base[base]]] = 'S'
    for not_move in not_moved:
        if state['runners_before'][not_move] != '':
            play[runner_dest[str(not_move)]] = str(not_move)

def handle_runner_not_thrown_out(move, play, state):
    '''
    handles the case during the handling of runner movement that the runner
    has NOT been marked as being thrown out including handling all errors and
    marking the players destination
    @param move - the runner's move
    @param play - the data to be loaded into the table
    @param state - the game state
    '''
    start_idx = move.find('(')
    end_idx = move.find(')')
    if start_idx != -1:
        fielding_info = move[start_idx+ 1: end_idx]
        fielding_info = fielding_info.split('/')
        for field in fielding_info:
            if field not in set(['TH', 'NR', 'UR', 'TUR', 'WP', 'PB']):
                play_parser(field, play, state, True)
    if move[2] == 'H':
        score_run(move, play, state)
    play[runner_dest[move[0]]] = move[2]

def handle_runner_thrown_out(move, play, state, outs_this_play):
    '''
    handles the case during the handling of runner movement that the runner
    HAS been marked as being thrown out including handling all errors and
    marking the players destination
    @param move - the runner's move
    @param play - the data to be loaded into the table
    @param state - the game state
    '''
    fielding_info = move[move.find('(')+ 1: move.find(')')]
    outs_this_play += play_parser(fielding_info, play, state, True)[1]
    if outs_this_play > 0:
        play[runner_dest[move[0]]] = 'O'
    else:
        play[runner_dest[move[0]]] = move[2]
        if move[2] == 'H':
            score_run(move, play, state)    

def runner_moves(moves, play, runners_out, state):
    '''
    handle all explicit movements for the play
    @param moves - list of runner moves
    @param play - the play which is added to the row of the table
    @param runners_out - list of runners who have already been marked out
    @param state - the game state
    '''
    # global runners_dest
    
    not_moved = set([1, 2, 3])
    outs_this_play = 0
    if moves:
        moves = moves.split(';')
        for move in moves:
            #Runner not batter
            if move[0] != 'B':
                runner = state['runners_before'][int(move[0])]
                not_moved.remove(int(move[0]))
            
            #Batter
            else:
                if play['play'][0] == 'K':
                    outs_this_play -= 1
                runner = play['batter_id']

            #Runner not thrown out    
            if move[1] != 'X':
                handle_runner_not_thrown_out(move, play, state)
            
            #Runner thrown out
            else:
                handle_runner_thrown_out(move, play, state, outs_this_play)

    for not_move in not_moved:
        if not_move not in runners_out:
            if state['runners_before'][not_move] != '' and play[runner_dest[str(not_move)]]== '':
                play[runner_dest[str(not_move)]] = str(not_move)
        else:
            play[runner_dest[str(not_move)]] = 'O'
    
    return outs_this_play

def parse_out(item, play, state, runners_out):
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
                    if play['batter_dest'] == 'O':
                        play['batter_dest'] = '1'
                    event_type = 18
            else:
                add_ast(fielder, play, state)
            field+=1
    return outs_this_play