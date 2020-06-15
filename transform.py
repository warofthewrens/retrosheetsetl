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
        return away_field_pos[pos]
    else:
        return home_field_pos[pos]

def get_batter_lineup(is_home, batter):
    if is_home:
        return home_lineup_rev[batter]
    else:
        return away_lineup_rev[batter]

def get_batter_field(is_home, batter):
    if is_home:
        return home_field_rev[batter]
    else:
        return away_field_rev[batter]


def score_run(move, play):
    global away_runs
    global home_runs
    info_index = 0
    no_rbi = False
    fielding_info = move[move.find('(')+ 1: move.find(')')]
    while info_index != -1:
        info_index = fielding_info.find('(')
        
        if fielding_info == 'UR':
            print('unearned run')
        elif fielding_info == 'TUR':
            print('team unearned run')
             
        elif fielding_info == 'NR' or fielding_info == 'NORBI':
            no_rbi = True

        elif fielding_info == 'WP':
            play['wp'] = True

        elif fielding_info == 'PB':
            play['pb'] = True
        fielding_info = fielding_info[fielding_info.find('(')+ 1: fielding_info.find(')')]  

    if play['is_home']:
        home_runs += 1
    else:
        away_runs += 1
    if not no_rbi:
        play['rbi'] += 1

def stolen_bases(sbs, play, runners_after):
    not_moved = set([1, 2, 3])
    for sb in sbs:
        base = sb[2]
        if base != 'H':
            runner = runners_before[int(base) - 1]
            runners_after[int(base)] = runner
            play[runner_dest[prev_base[base]]] = base
        else:
            score_run(sb, play)

        not_moved.remove(int(base))
    
    for not_move in not_moved:
        runners_after = runners_before[not_move]
    
def runner_moves(moves, play, runners_after, runners_out):
    # global runners_dest
    
    not_moved = set([1, 2, 3])
    outs_this_play = 0
    if moves:
        moves = moves.split(';')
        for move in moves:
            if move[0] != 'B':
                runner = runners_before[int(move[0])]
                not_moved.remove(int(move[0]))
            else:
                runner = play['batter_id']
                for base, runner_aft in runners_after.items():
                    if runner_aft == runner:
                        runners_after[base] = ''
                        break
            if move[1] != 'X':
                if move[2] == 'H':
                    score_run(move, play)
                else:
                    runners_after[int(move[2])] = runner
                play[runner_dest[move[0]]] = move[2]
            else:
                fielding_info = move[move.find('(')+ 1: move.find(')')]
                outs_this_play = event_description(fielding_info, play, True)[2]
            
    return outs_this_play
    
    for not_move in not_moved:
        if not_move not in runners_out and runners_after[not_move] == '':
            runners_after[not_move] = runners_before[not_move]
    
    return runners_after


def event_description(item, play, second_event):
    global outs
    event = item.split('.')
    runners_after = defaultdict(str)
    event_type = 0
    outs_this_play = 0
    runners_out = []
    if len(event) > 1:
        runner_moves(event[1], play, runners_after, runners_out)
    
    # runners_after[1] = runners_before[1]
    # runners_after[2] = runners_before[2]
    # runners_after[3] = runners_before[3]
    item = event[0]

    if item[0].isdigit():
        i = 0
        po_ers = []
        last_fielder = item[i]
        outs_this_play = 1
        #TODO: creat first_PO, second_po, and third_po list which store the appropriate fielders
        po = {
            1: [],
            2: [],
            3: []
        }

        while len(item) > i:
            if item[i].isdigit():
                po[outs_this_play].append(get_field_id(play['is_home'], item[i]))
                if(i == 0):
                    play['hit_loc'] = int(item[i])
            if item[i] == '(':
                runners_out.append(int(item[i + 1]))
                if len(item) > i + 3:
                    outs_this_play += 1
                    po[outs_this_play].append(get_field_id(play['is_home'], item[i - 1]))
                i += 2
            i+=1
        # if len(po_ers) > 0:
        #     if last_fielder != po_ers[0]:
        #         po_ers.append(last_fielder)
        # else:
        #     po_ers.append(last_fielder)
        
        pos = po.items()
        po_dict = {1 : 'first_po', 2: 'second_po', 3: 'third_po'}
        ast_dict = {1: 'first_ast', 2: 'second_ast', 3: 'third_ast', 4: 'fourth_ast', 5: 'fifth_ast'}
        ast = 1
        for out, fielders in pos:
            field = 0
            for fielder in fielders:
                if field == (len(fielders) - 1):
                    play[po_dict[out]] = fielder
                else:
                    play[ast_dict[ast]] = fielder
                    ast+=1
                field+=1
                if ast > 5:
                    break
        
        # for po in po_ers:
        #     play[po_dict[outs_this_play]] = get_field_id(play['is_home'], po)
        event_type = 2
        
    
    if re.match('^SB[23H]', item):
        sbs = item.split(';')
        stolen_bases(sbs, play, runners_after)
        

    elif item[0] in set(['S', 'D', 'T']):
        if item[0] == 'S':
            play['hit_val'] = 1
            event_type = 20
            runners_after[1] = play['batter_id']
            play['batter_dest'] = '1'

        elif item[0] == 'D':
            play['hit_val'] = 2
            event_type = 21
            runners_after[2] = play['batter_id']
            play['batter_dest'] = '2'

        elif item[0] == 'T':
            play['hit_val'] = 3
            event_type = 22
            runners_after[3] = play['batter_id']
            play['batter_dest'] = '3'
        
        play['fielder_id'] = get_field_id(play['is_home'], item[1])
        play['hit_loc'] = item[1]
    
    elif item[0] == 'E' or item[0:3] == 'FLE':
        if item[0] == 'E':
            event_type = 18
            runners_after[1] = play['batter_id']
            play['batter_dest'] = '1'
            play['fielder_id'] = get_field_id(play['is_home'], item[1])
            play['hit_loc'] = item[1]
        else:
            event_type = 13
            play['fielder_id'] = get_field_id(play['is_home'], item[3])
            play['hit_loc'] = item[3]
        
        play['error_player_id'] = play['fielder_id']
        
    
    elif item[0] == 'K':
        second_event_arr = item.split('+')
        if len(second_event_arr) > 1:
            event_description(second_event_arr[1], play, True)
            if 'B-' in play['play']:
                outs_this_play -= 1
        event_type = 3
        outs_this_play += 1
    
    elif item[0:2] == 'WP':
        play['wp'] = True
        event_type = 9

    elif item[0:2] == 'PB':
        play['pb'] = True
        event_type = 10

    elif item[0] == 'W':
        second_event_arr = item.split('+')
        if len(second_event_arr) > 1:
            event_description(second_event_arr[1], play, True)
        event_type = 14
        runners_after[1] = play['batter_id']
        play['batter_dest'] = '1'

    elif item[0:2] == 'DI':
        event_type = 5
    
    elif item[0:2] == 'OA':
        event_type = 12
    
    elif item[0:2] == 'PO':
        fielding_info = item[item.find('(')+ 1: item.find(')')]
        if not 'E' in fielding_info:
            event_type = 8
            if item[2] == 'C':
                runners_out.append(int(item[4]))
            else:
                runners_out.append(int(item[2]))
            outs_this_play += 1
            play['first_po'] = fielding_info[0]
            play['first_ast'] = fielding_info[1]
        else:
            event_type = 7
    
    elif item[0] == 'I' or item[0:] == 'IW':
        second_event_arr = item.split('+')
        if len(second_event_arr) > 1:
            event_description(second_event_arr[1], play, True)

        event_type = 15
        runners_after[1] = play['batter_id']
        play['batter_dest'] = '1'
    
    elif item == 'HP':
        event_type = 16
        runners_after[1] = play['batter_id']
        play['batter_dest'] = '1'

    elif item[0] == 'H' or item[0:] == 'HR':
        event_type = 23
        play['hit_val'] = 4
        play['batter_dest'] = 'H'
        if 'B-' not in play['play']:
            score_run('', play)
    
    elif item[0:2] == 'BK':
        event_type = 11 

    elif item[0:2] == 'CS':
        event_type = 6
        fielding_info = item[item.find('(')+ 1: item.find(')')]
        if not 'E' in fielding_info:
            runners_out.append(int(prev_base[item[2]]))
            play['first_ast'] = fielding_info[0]
            play['first_po'] = fielding_info[1]
            outs_this_play += 1
        else:
            print('Caught stealing error')

    elif item[0] == 'C' and item != 'CS':
        event_type = 17
        runners_after[1] = play['batter_id']
        play['batter_dest'] = '1'

    elif len(item) > 1:
        if item[0:2] == 'FC':
            print("fielder's choice")
            new_item = play['play'].split('.')
            new_item.insert(1, '.B-1;')
            play['play'] = ''.join(new_item)
            
    if not second_event:
        play['event_type'] = event_type
    return runners_after, runners_out, outs_this_play
    

def modifier(modifier, play, runners_after, runners_out):
    i = 0

    # print(play['play'])
    # print(modifier)
    location = re.search('[1-9]', modifier)
    if location:
        play['hit_loc'] = location.group(0)
    # print(location)
    if modifier[0] == 'B' and modifier not in set(['BINT', 'BOOT', 'BR']):
        play['bunt_flag'] = True
        play['ball_type'] = modifier[1]

    if modifier[0] in set(['G', 'F', 'L', 'P']) and modifier not in set(['FINT', 'FL', 'FO', 'PASS']):
        play['ball_type'] = modifier[0]
    
    if modifier == 'FL':
        play['foul_flag'] = True

    if modifier == 'SF':
        play['sac_fly'] = True
    
    if modifier == 'SH':
        play['sac_bunt'] = True
    
    if modifier == 'E':
        play['error_player_id'] = get_field_id(play['is_home'], modifier[1])
    


#TODO: giant WIP
def expand_play(play):
    global outs
    moves = play['play'].split('.')
    play_arr = moves[0].split('/')
    outs_this_play = 0
    outs_this_play_rm = 0
    i = 0
    for item in play_arr:
        if (i == 0):
            runners_after, runners_out, outs_this_play_ed = event_description(item, play, False)
        else:
            modifier(item, play, runners_after, runners_out)
        i += 1
    if (len(moves) > 1):
        outs_this_play_rm = runner_moves(moves[1], play, runners_after, runners_out)
    else:
        runner_moves(None, play, runners_after, runners_out)
    runners_before[1] = runners_after[1]
    runners_before[2] = runners_after[2]
    runners_before[3] = runners_after[3]
    outs_this_play = outs_this_play_ed + outs_this_play_rm
    outs += outs_this_play
    play['outs_on_play'] = outs_this_play
    return play

def print_lineups():
    print(home_lineup)
    print(away_lineup)
    print(home_field_pos)
    print(away_field_pos)

def build_lineups(starts):
    global away_lineup_rev, home_lineup_rev, away_field_rev, home_field_rev
    for start in starts:
        if (start['is_home']):
            home_lineup[start['bat_pos']] = start['player_id']
            home_field_pos[start['field_pos']] = start['player_id']
            
        else:
            away_lineup[start['bat_pos']] = start['player_id']
            away_field_pos[start['field_pos']] = start['player_id']
            
    # print_lineups()
    away_lineup_rev = {v: k for k, v in away_lineup.items()}
    home_lineup_rev = {v: k for k, v in home_lineup.items()}
    away_field_rev = {v: k for k, v in away_field_pos.items()}
    home_field_rev = {v: k for k, v in home_field_pos.items()}
    return

def transform_plays(plays, subs, rows, context):
    global outs, home_runs, away_runs, runners_before
    sub_spot = {}
    for sub in subs:
        sub_spot[int(sub['play_idx'])] = sub
    
    play_idx = 0
    for play in plays:
        sub = sub_spot.get(play_idx)

        # If there is 
        if sub:
            if sub['is_home']:
                home_lineup[sub['bat_pos']] = sub['player_id']
                home_lineup_rev[sub['player_id']] = sub['bat_pos']
                home_field_pos[sub['field_pos']] = sub['player_id']
                home_field_rev[sub['player_id']] = sub['field_pos']
            else:
                away_lineup[sub['bat_pos']] = sub['player_id']
                away_field_pos[sub['field_pos']] = sub['player_id']
                away_field_pos[sub['field_pos']] = sub['player_id']
                away_field_rev[sub['player_id']] = sub['field_pos']
        if not play['play'] == 'NP':
            play['outs'] = outs
            play['sac_fly'] = False
            play['sac_bunt'] = False
            play['wp'] = False
            play['pb'] = False
            play['away_runs'] = away_runs
            play['home_runs'] = home_runs
            play['first_runner_id'] = runners_before[1]
            play['second_runner_id'] = runners_before[2]
            play['third_runner_id'] = runners_before[3]
            play['field_pos'] = get_batter_field(play['is_home'], play['batter_id'])
            play['bunt_flag'] = False
            play['foul_flag'] = False
            play['batter_dest'] = ''
            play['first_dest'] = ''
            play['second_dest'] = ''
            play['third_dest'] = ''
            play['rbi'] = 0
            play['hit_val'] = 0
            play['fielder_id'] = ''
            if (play['field_pos'] not in set([11, 12])):
                play['lineup_pos'] = get_batter_lineup(play['is_home'], play['batter_id'])
            expand_play(play)
            play['pitcher_id'] = get_field_id(play['is_home'], 1)
            play['ab_flag'] = ((play['event_type'] in set([2, 3, 18, 19, 20, 21, 22, 23])) and (play['sac_fly'] == False) and (play['sac_bunt'] == False))

            p_a = PlateAppearance(context=context).dump(play)
            print(p_a)

        # inning ended
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
    print('Home score: ', home_runs)
    print('Away score: ', away_runs)
    return