from marshmallow import Schema, fields, validate, pre_load, post_load

def validate_count(count):
    # print(count)
    return
    

def validate_pitches(pitches):
    # print(pitches)
    return

def validate_play(play):
    # print(play)
    return
class Play(Schema):
    inning = fields.Integer(validate=validate.Range(1, 30))
    is_home = fields.Boolean()
    batter_id = fields.String()
    count = fields.String(validate=validate_count)
    pitches = fields.String(validate=validate_pitches)
    play = fields.String(validate=validate_play)

    @post_load
    def improve_play(self, data, **kwargs):
        data['play'] = data['play'].replace('!', '')
        play = data['play'].split('.')
        # print(play)
        event = play[0].split('/')
        outs = event[0]
        flag=False
        if len(play) > 1:
            runner_moves(play)
                
        if event[0][0].isdigit() or event[0][0] == 'E':
            fielded(event, play, data)
            
        if event[0][0:2] in set(['WP', 'PB', 'BK', 'CS', 'OA', 'DI', 'PO']) and len(play) > 1:
            runner_events(play)
        
        data['play'] = '.'.join(play)
        return data

def runner_moves(play):
    moves = play[1].split(';')
    i = 0
    for move in moves:
        move = move.replace('/TH', '')
        move_action = move.find('X')
        if move_action == -1:
            move_action = move.find('-')
        runner = move[move_action - 1]
        info_index = move.find('(')
        end_index = move.find(')')
        fielding_info = move[info_index + 1: end_index]
        cur_info = info_index
        cur_end = end_index + 1
    # if it's an error or an out include the player getting thrown out/moving on an error
        while info_index != -1:
            mutated = False
            new_move = move[cur_end:]
            placement = fielding_info.find('/')
            if placement == -1:
                placement = end_index
            if len(fielding_info) > 0:
                if (fielding_info[0].isdigit() or fielding_info[0] == 'E'):
                    # print(moves[i][0:cur_index])
                    mutated = True
                    if len(move) > cur_end + 1:
                        move = move[0:cur_info] + '(' + fielding_info[0:placement] + '(' + runner + ')' + fielding_info[placement:] + ')'  + '(' + move[cur_end + 1:]
                    else:
                        move = move[0:cur_info] + '(' + fielding_info[0:placement] + '(' + runner + ')' + fielding_info[placement:] + ')' + move[cur_end + 1:]
                if (fielding_info == 'NR'):
                    if '-H' in move[cur_end:] or 'XH' in move[cur_end:]:
                        runner = move[cur_info - 1]
                        move = move[0:cur_info] + move[cur_end:] + '(NR)'
                        cur_end -= 4
                        new_move = move[cur_end:]
                        
            # new_move = new_move[3:]
            # print('new_move', new_move)
            
            info_index = new_move.find('(')
            end_index = new_move.find(')')
            if mutated:
                cur_info = cur_end + 3 + info_index
                cur_end += end_index + 4
            else:
                cur_info = cur_end + info_index
                cur_end += end_index
            fielding_info = new_move[info_index + 1: end_index]
            j = 1
            while fielding_info == '' and end_index != -1:
                end_index = new_move[j:].find(')')
                fielding_info = new_move[info_index + 1: end_index]
                new_move = new_move[0] + new_move[2:]
                j += 1
            cur_end+=(j - 1)
        move = move.replace('((', '(')
        moves[i] = move
        i+=1
    play[1] = ';'.join(moves)

def fielded(event, play, data):
    outs = event[0].split(')')
    if 'B' not in data['play'][0]:
        if outs[-1] != '':
            outs[-1] = outs[-1] + '(B)'
    outs = ')'.join(outs)
    event[0] = outs
    event = '/'.join(event)
    play[0] = event

def runner_events(play):
    if '-H' in play[1]:
        moves = play[1].split(';')
        i = 0
        for move in moves:
            scored = move.find('-H')
            if scored != -1 and '(NR)' not in move:
                moves[i] = move[:scored] + move[scored:] + '(NR)'
            i+=1
        play[1] = ';'.join(moves)