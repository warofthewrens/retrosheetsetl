#from parser_helper import parse_out, stolen_bases, get_field_id, add_po, add_ast, add_error, score_run, runner_dest, prev_base
import re
from parsed_schemas.run import Run
from parsed_schemas.base_running_event import BaseRunningEvent
import copy
#from play_parser import play_parser

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

def get_lineup_id(is_home, pos, state):
    '''
    returns the player id for position, pos in lineup of appropriate team
    @param is_home - boolean for if batter is_home
    @param pos - lineup position to lookup
    @param state - game state which includes current lineups
    '''
    if isinstance(pos, int):
        if is_home:
            return state['lineups']['home_lineup'][pos]
        else:
            return state['lineups']['away_lineup'][pos]
    else:
        return ''

def get_field_id(is_home, pos, state):
    '''
    returns the player id for position, pos in field of appropriate team
    @param is_home - boolean for if batter is_home
    @param pos - field position to lookup
    @param state - game state which includes current field positions
    '''
    if isinstance(pos, int):
        if is_home:
            return state['lineups']['away_field_pos'][pos]
        else:
            return state['lineups']['home_field_pos'][pos]
    else:
        return ''
    

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
    if (state['ast']) > 5:
        return
    play[ast_dict[state['ast']]] = get_field_id(play['is_home'], ast_player, state)

def add_base_running_event(base, play, state, rows, event):
    if event == 'P':
        runner = state['runners_before'][int(base)]
    else:
        runner = state['runners_before'][int(prev_base[base])]
    
    if play['is_home']:
        new_br_event = {'game_id': state['game_id'], 'date': state['date'], 'running_team': state['home_team'], 
                        'pitching_team': state['away_team'],'event': event, 'base': base, 'runner': runner,
                        'pitcher': play['pitcher_id'], 'catcher': get_field_id(play['is_home'], 2, state), 'inning':state['inning'],
                        'outs': state['outs']}
    else:
        new_br_event = {'game_id': state['game_id'], 'date': state['date'], 'running_team': state['away_team'], 
                        'pitching_team': state['home_team'],'event': event, 'base': base, 'runner': runner,
                        'pitcher': play['pitcher_id'], 'catcher': get_field_id(play['is_home'], 2, state), 'inning':state['inning'],
                        'outs': state['outs']}
    new_br_event['event_id'] = state['br_event_id']
    state['br_event_id'] += 1
    br_event = BaseRunningEvent().dump(new_br_event)
    rows['base_running_event'].append(br_event)

def run_modifiers(move, play, state, rows):
    '''
    Handle all modifiers such as (NR) meaning no rbi attached to a run scored
    '''
    no_rbi = False
    ter = True
    er = True
    modifiers = copy.copy(move)
    info_index = modifiers.find('(')
    end_index = modifiers.find(')')
    fielding_info = modifiers[info_index + 1: end_index]
    modifiers = modifiers[end_index + 1:]
    while info_index != -1:
        # print(move[])
        if fielding_info == 'NR' or fielding_info == 'NORBI':
            no_rbi = True

        elif fielding_info == 'TUR':
            ter = False
        
        elif fielding_info == 'UR':
            er = False
            ter = False

        elif fielding_info == 'WP':
            play['wp'] = True

        elif fielding_info == 'PB':
            play['pb'] = True
        
        info_index = modifiers.find('(')
        end_index = modifiers.find(')')
        fielding_info = modifiers[info_index + 1: end_index]
        modifiers = modifiers[end_index + 1:]
    if move[0] == 'B':
        scorer = play['batter_id']
    else:
        scorer = state['runners_before'][int(move[0])]
    
    if play['is_home']:
        new_run = {'game_id': state['game_id'], 'date': state['date'], 'scoring_team': state['home_team'], 'conceding_team': state['away_team'],
                    'scoring_player': scorer, 'responsible_pitcher': state['responsible_pitchers'][move[0]], 'is_sp': state['lineups']['away_field_pos']['sp'] == play['pitcher_id'],
                    'batter': play['batter_id'], 'is_earned': er, 'is_team_earned': ter, 'is_rbi': not no_rbi, 'inning': play['inning'], 'outs': state['outs']}
    else:
        new_run = {'game_id': state['game_id'], 'date': state['date'], 'scoring_team': state['away_team'], 'conceding_team': state['home_team'],
                    'scoring_player': scorer, 'responsible_pitcher': state['responsible_pitchers'][move[0]], 'is_sp': state['lineups']['home_field_pos']['sp'] == play['pitcher_id'],
                    'batter': play['batter_id'], 'is_earned': er, 'is_team_earned': ter, 'is_rbi': not no_rbi, 'inning': play['inning'], 'outs': state['outs']}
    new_run['run_id'] = state['run_id']
    state['run_id'] += 1
    run = Run().dump(new_run)
    rows['run'].append(run)
    return no_rbi

def score_run(move, play, state, rows):
    '''
    score a run for the appropriate team and handle all required upkeep
    @param move - the specific runner move which scored the run
    @param play - the play to be loaded into the table
    @param state - the game state
    '''
    info_index = 0
    no_rbi = run_modifiers(move, play, state, rows)
    state['runs_allowed'][state['responsible_pitchers'][move[0]]]+=1
    play['runs_on_play'] += 1
    if move[0] != 'B':
        play[scorer_dict[play['runs_on_play']]] = state['runners_before'][int(move[0])]
    else:
        play[scorer_dict[play['runs_on_play']]] = play['batter_id']
    if not no_rbi:
        play['rbi'] += 1

def stolen_bases(sbs, play, state, rows):
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
            score_run('3-H(NR)', play, state, rows)
            play['third_dest'] = 'H'
        not_moved.remove(int(prev_base[base]))
        play[runner_event[prev_base[base]]] = 'S'
        add_base_running_event(base, play, state, rows, 'S')
    for not_move in not_moved:
        if state['runners_before'][not_move] != '':
            play[runner_dest[str(not_move)]] = str(not_move)

def handle_runner_not_thrown_out(move, play, state, rows):
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
                play_parser(field, play, state, True, rows)
    if move[2] == 'H':
        score_run(move, play, state, rows)
    play[runner_dest[move[0]]] = move[2]

def handle_runner_thrown_out(move, play, state, outs_this_play, rows):
    '''
    handles the case during the handling of runner movement that the runner
    HAS been marked as being thrown out including handling all errors and
    marking the players destination
    @param move - the runner's move
    @param play - the data to be loaded into the table
    @param state - the game state
    '''
    fielding_info = move[move.find('(')+ 1: move.find(')')]
    outs_this_play += play_parser(fielding_info, play, state, True, rows)[1]
    if outs_this_play > 0:
        play[runner_dest[move[0]]] = 'O'
    else:
        play[runner_dest[move[0]]] = move[2]
        if move[2] == 'H':
            score_run(move, play, state, rows)
    return outs_this_play   

def runner_moves(moves, play, runners_out, state, rows):
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
            if move[0] not in ('B', '1','2','3'):
                continue
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
                handle_runner_not_thrown_out(move, play, state, rows)
            
            #Runner thrown out
            else:
                outs_this_play = handle_runner_thrown_out(move, play, state, outs_this_play, rows)

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
    event_type = 0
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
    return outs_this_play, event_type

def play_parser(item, play, state, second_event, rows):
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
        outs_this_play, event_type = parse_out(item, play, state, runners_out)
    
    if re.match('^SB[23H]', item):
        sbs = item.split(';')
        stolen_bases(sbs, play, state, rows)
        event_type = 4
        
    elif item[0:2] == 'DI':
        event_type = 5
    
    elif item[0:3] == 'DGR':
        play['hit_val'] = 2
        event_type = 21
        play['batter_dest'] = '2'

    elif item[0] in set(['S', 'D', 'T']) and not item[0:2] == 'TH':
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
            if play['batter_dest'] == '':
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
            new_runners_out, new_outs_this_play = play_parser(second_event_arr[1], play, state, True, rows)
            runners_out.extend(new_runners_out)
            outs_this_play += new_outs_this_play
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
            new_runners_out, new_outs_this_play = play_parser(second_event_arr[1], play, state, True, rows)
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
                play[runner_event[prev_base[item[4]]]] = 'P'
                add_base_running_event(prev_base[item[4]], play, state, rows, 'P')
            else:
                runners_out.append(int(item[2]))
                play[runner_event[item[2]]] = 'P'
                add_base_running_event(item[2], play, state, rows, 'P')
            outs_this_play += 1
            if len(fielding_info) > 1:
                add_po(fielding_info[1], play, state)
            if len(fielding_info) > 0:
                add_ast(fielding_info[0], play, state)
        else:
            event_type = 7
            error_player = item[item.find('E') + 1]
            add_error(error_player, play, state)
    
    elif item[0] == 'I' or item[0:] == 'IW':
        second_event_arr = item.split('+')
        if len(second_event_arr) > 1:
            new_runners_out, new_outs_this_play = play_parser(second_event_arr[1], play, state, True, rows)
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
            score_run('B-H', play, state, rows)
    
    elif item[0:2] == 'BK':
        event_type = 11 

    elif item[0:2] == 'CS':
        event_type = 6
        fielding_info = item[item.find('(')+ 1: item.find(')')]
        if not 'E' in fielding_info:
            runners_out.append(int(prev_base[item[2]]))
            play[runner_dest[prev_base[item[2]]]] = 'O'
            play[runner_event[prev_base[item[2]]]] = 'C'
            add_base_running_event(item[2], play, state, rows, 'C')
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