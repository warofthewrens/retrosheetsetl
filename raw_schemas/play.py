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
            if 'X' in play[1]:
                moves = play[1].split(';')
                i = 0
                for move in moves:
                    runner = move[move.find('X') - 1]
                    if runner != '-':
                        fielding_info = move.find('/')
                        if fielding_info == -1:
                            fielding_info = move.find(')')
                        moves[i] = move[0:fielding_info] + '(' + runner + ')' + move[fielding_info:]
                    i+=1
                
                play[1] = ';'.join(moves)
        if event[0][0].isdigit():
            outs = event[0].split(')')
            if 'B' not in data['play'][0]:
                if outs[-1] != '':
                    outs[-1] = outs[-1] + '(B)'
            outs = ')'.join(outs)
            event[0] = outs
            event = '/'.join(event)
            play[0] = event
        data['play'] = '.'.join(play)
        return data

