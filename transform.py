from parsed_schemas.plate_appearance import PlateAppearance


home_lineup = {}
away_lineup = {}

home_field_pos = {}
away_field_pos = {}

outs = 0

runners = {1: None, 2: None, 3:None}

def get_lineup_id(is_away, pos):
    if is_away:
        return away_lineup[pos]
    else:
        return home_lineup[pos]

def get_field_id(is_away, pos):
    if is_away:
        return away_field_pos[pos]
    else:
        return home_field_pos[pos]

#TODO: giant WIP
def expand_play(play):
    print(play)
    play_arr = play['play'].split('/')
    hits = ['S', 'D', 'T']
    i = 0
    for item in play_arr:
        i = 0
        for c in item:
            if c.isdigit():
                new_i = i
                if len(item) > i:
                    
                    while item[new_i+1].isdigit():
                        if(new_i - i == 0):
                            play['first_ast'] = get_field_id(play['is_away'], item[new_i])
                        if(new_i - i == 1):
                            play['second_ast'] = get_field_id(play['is_away'], item[new_i])
                        new_i+=1
                play['first_po'] = get_field_id(play['is_away'], item[new_i])
                play['event_type'] = 2
            if c in set(['S', 'D', 'T']):
                is_hit = True
                if c == 'S':
                    play['hit_val'] = 1
                    play['event_type'] = 20

            i+=1
            print(c)


    return play

def print_lineups():
    print(home_lineup)
    print(away_lineup)
    print(home_field_pos)
    print(away_field_pos)

def build_lineups(starts):
    for start in starts:
        if (start['is_away']):
            away_lineup[start['bat_pos']] = start['player_id']
            away_field_pos[start['field_pos']] = start['player_id']
        else:
            home_lineup[start['bat_pos']] = start['player_id']
            home_field_pos[start['field_pos']] = start['player_id']
    print_lineups()
    return

def transform_plays(plays, subs, rows, context):
    sub_spot = {}
    for sub in subs:
        sub_spot['play_idx'] = sub
    
    play_idx = 0
    for play in plays:
        sub = sub_spot.get(play_idx)
        if sub:
            if sub['is_away']:
                away_lineup[sub['bat_pos']] = sub['player_id']
                away_field_pos[sub['field_pos']] = sub['player_id']
            else:
                home_lineup[start['bat_pos']] = sub['player_id']
                home_field_pos[start['field_pos']] = sub['player_id']
        if not play['play'] == 'NP':
            expand_play(play)
            p_a = PlateAppearance(context=context).dump(play)
            print(p_a)
        play_idx+=1
    return


def transform_game(game):
    rows = {table: [] for table in ['plate_appearance', 'game']}
    context = {'game_id': game['game_id'], 'date': game['info']['date']}
    build_lineups(game['lineup'])
    transform_plays(game['plays'], game['subs'], rows, context)
    return