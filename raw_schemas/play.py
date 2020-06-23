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
        play = data['play'].split('.')
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
        move_action = move.find('X')
        if move_action == -1:
            move_action = move.find('-')
        runner = move[move_action - 1]
        info_index = move.find('(')
        end_index = move.find(')')
        fielding_info = move[info_index + 1: end_index]
        new_move = move[end_index + 1:]
    # if it's an error or an out include the player getting thrown out/moving on an error
        while info_index != -1:
            
            placement = fielding_info.find('/')
            if placement == -1:
                placement = end_index
            if (fielding_info[0].isdigit() or fielding_info[0] == 'E'):
                move = move[0:info_index] + '(' + fielding_info[0:placement] + '(' + runner + ')' + fielding_info[placement:] + move[end_index:]
            info_index = new_move.find('(')
            end_index = new_move.find(')')
            fielding_info = new_move[info_index + 1: end_index]
            new_move = new_move[end_index + 1:]
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
                moves[i] = move[:scored] + '(NR)' + move[scored:]
            i+=1
        play[1] = ';'.join(moves)