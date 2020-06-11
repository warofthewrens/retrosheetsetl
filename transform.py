from parsed_schemas.plate_appearance import PlateAppearance
from collections import defaultdict
import re  


home_lineup = {}
away_lineup = {}

home_field_pos = {}
away_field_pos = {}

runner_dest = {'1' : 'first_dest',
               '2' : 'second_dest',
               '3' : 'third_dest'}

home_runs = 0
away_runs = 0

outs = 0

runners_before = defaultdict(str)

def get_lineup_id(is_home, pos):
    if is_home:
        return home_lineup[pos]
    else:
        return away_lineup[pos]

def get_field_id(is_home, pos):
    pos = int(pos)
    if is_home:
        return home_field_pos[pos]
    else:
        return away_field_pos[pos]

def score_run(play):
    global away_runs
    global home_runs
    if play['is_home']:
        home_runs += 1
    else:
        away_runs += 1

def stolen_bases(sbs, play, runners_after):
    not_moved = set([1, 2, 3])
    for sb in sbs:
        base = sb[2]
        if base != 'H':
            runner = runners_before[int(base) - 1]
            runners_after[int(base)] = runner
        else:
            score_run(play)
        not_moved.remove(int(base))
    
    for not_move in not_moved:
        runners_after = runners_before[not_move]
    


def event_description(item, play):
    global outs
    event = item.split('.')
    runners_after = defaultdict(str)
    runners_out = []
    if len(event) > 1:
        runner_moves(event[1], play, runners_after, runners_out)
        
    item = event[0]

    if item[0].isdigit():
        i = 0
        po_ers = []
        last_fielder = item[i]
        #TODO: creat first_PO, second_po, and third_po list which store the appropriate fielders
        while len(item) - 1 > i:
            if item[i].isdigit():
                if(i == 0):
                    play['first_ast'] = get_field_id(play['is_home'], item[i])
                    play['hit_loc'] = int(item[i])
                if(i == 1):
                    play['second_ast'] = get_field_id(play['is_home'], item[i])
                last_fielder = item[i]
            if item[i] == '(':
                po_ers.append(item[i - 1])
                runners_out.append(int(item[i + 1]))
                i += 2
            i+=1
        if len(po_ers) > 0:
            if last_fielder != po_ers[0]:
                po_ers.append(last_fielder)
        else:
            po_ers.append(last_fielder)
        p = 0
        po_dict = {0 : 'first_po', 1: 'second_po', 2: 'third_po'}
        for po in po_ers:
            play[po_dict[p]] = get_field_id(play['is_home'], po)
            play['event_type'] = 2
            p += 1
        outs += p
        
    
    if re.match('^SB[23H]', item):
        sbs = item.split(';')
        stolen_bases(sbs, play, runners_after)
        

    elif item[0] in set(['S', 'D', 'T']):
        if item[0] == 'S':
            play['hit_val'] = 1
            play['event_type'] = 20
            runners_after[1] = play['batter_id']

        elif item[0] == 'D':
            play['hit_val'] = 2
            play['event_type'] = 21
            runners_after[2] = play['batter_id']

        elif item[0] == 'T':
            play['hit_val'] = 3
            play['event_type'] = 22
            runners_after[3] = play['batter_id']
        
        play['fielder_id'] = get_field_id(play['is_home'], item[1])
        play['hit_loc'] = item[1]
    
    elif item[0] == 'E':
        is_error = True
        play['event_type'] = 18
        runners_after[1] = play['batter_id']

        play['fielder_id'] = get_field_id(play['is_home'], item[1])
        play['error_player_id'] = play['fielder_id']
        play['hit_loc'] = item[1]
    
    elif item[0] == 'K':
        play['event_type'] = 3
        outs += 1
    
    elif item[0:2] == 'WP':
        play['wp'] = True

    elif item[0] == 'W':
        play['event_type'] = 14
        runners_after[1] = play['batter_id']
    
    elif item[0] == 'I' or item[0:] == 'IW':
        play['event_type'] = 15
        runners_after[1] = play['batter_id']
    
    elif item == 'HP':
        play['event_type'] = 16
        runners_after[1] = play['batter_id']

    elif item[0] == 'H' or item[0:] == 'HR':
        play['event_type'] = 23
        play['hit_val'] = 4
        score_run(play)
        runners_after[1] = play['batter_id']

    elif item[0] == 'C' and item != 'CS':
        play['event_type'] = 17
        runners_after[1] = play['batter_id']

    elif len(item) > 1:
        if item[0:2] == 'FC':
            new_item = play['play'].split('.')
            new_item.insert(1, '.B-1;')
            play['play'] = ''.join(new_item)
            

    return runners_after, runners_out
    

def runner_moves(moves, play, runners_after, runners_out):
    # global runners_dest
    moves = moves.split(';')
    not_moved = set([1, 2, 3])
    for move in moves:
        if move[0] != 'B':
            runner = runners_before[int(move[0])]
            if move[1] != 'X':
                if move[2] == 'H':
                    score_run(play)
                else:
                    runners_after[int(move[2])] = runner
                play[runner_dest[move[0]]] = move[2]
            else:
                fielding_info = move[move.find('(')+ 1: move.find(')')]
                event_description(fielding_info)
            not_moved.remove(int(move[0]))
    
    for not_move in not_moved:
        if not_move not in runners_out:
            runners_after = runners_before[not_move]
    
    return runners_after

def modifier(item, play, runners_after, runners_out):
    modifiers = item.split('.')
    i = 0
    if len(modifiers) > 2:
        raise Exception
    
    if modifiers[0][0] == 'B' and modifiers[0] not in set(['BINT', 'BOOT', 'BR']):
        play['bunt_flag'] = True
        play['ball_type'] = modifiers[0][1]

    if modifiers[0][0] in set(['G', 'F', 'L', 'P']) and modifiers[0] not in set(['FINT', 'FL', 'FO', 'PASS']):
        play['ball_type'] = modifiers[0][0]
    
    if modifiers[0] == 'SF':
        play['sac_fly'] = True
    
    if modifiers[0] == 'SH':
        play['sac_bunt'] = True
    
    if len(modifiers) > 1:
        runner_moves(modifiers[1], play, runners_after, runners_out)


#TODO: giant WIP
def expand_play(play):
    global outs
    play_arr = play['play'].split('/')
    
    i = 0
    for item in play_arr:
        if (i == 0):
            runners_after, runners_out = event_description(item, play)
        else:
            modifier(item, play, runners_after, runners_out)
        i += 1
    runners_before[1] = runners_after[1]
    runners_before[2] = runners_after[2]
    runners_before[3] = runners_after[3]
    return play

def print_lineups():
    print(home_lineup)
    print(away_lineup)
    print(home_field_pos)
    print(away_field_pos)

def build_lineups(starts):
    for start in starts:
        if (start['is_home']):
            away_lineup[start['bat_pos']] = start['player_id']
            away_field_pos[start['field_pos']] = start['player_id']
        else:
            home_lineup[start['bat_pos']] = start['player_id']
            home_field_pos[start['field_pos']] = start['player_id']
    # print_lineups()
    return

def transform_plays(plays, subs, rows, context):
    global outs, home_runs, away_runs, runners_before
    sub_spot = {}
    for sub in subs:
        sub_spot[int(sub['play_idx'])] = sub
    
    play_idx = 0
    for play in plays:
        sub = sub_spot.get(play_idx)
        # print(sub)
        if sub:
            if sub['is_home']:
                home_lineup[sub['bat_pos']] = sub['player_id']
                home_field_pos[sub['field_pos']] = sub['player_id']
            else:
                away_lineup[sub['bat_pos']] = sub['player_id']
                away_field_pos[sub['field_pos']] = sub['player_id']
        if not play['play'] == 'NP':
            expand_play(play)
            play['pitcher_id'] = get_field_id(not play['is_home'], 1)
            p_a = PlateAppearance(context=context).dump(play)
        if outs == 3:
            runners_before = defaultdict(str)
            outs = 0
        
        play_idx+=1
    return


def transform_game(game):
    rows = {table: [] for table in ['plate_appearance', 'game']}
    context = {'game_id': game['game_id'], 'date': game['info']['date']}
    build_lineups(game['lineup'])
    transform_plays(game['plays'], game['subs'], rows, context)
    return