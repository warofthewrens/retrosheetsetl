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
            moves = play[1].split(';')
            # for move in moves:
            #     info_index = move.find('(')
            #     end_index = move.find(')')
            #     fielding_info = move[info_index + 1: end_index]
            #     move = move[end_index + 1:]
            # # if it's an error or an out include the player getting thrown out/moving on an error
            #     while info_index != -1:
            #         if (fielding_info[0].isdigit() or fielding_info[0] == 'E'):
            #             moves[i] = move[0:fielding_info] + '(' + runner + ')' + move[fielding_info:]

            #         info_index = moves.find('(')
            #         end_index = moves.find(')')
            #         fielding_info = moves[info_index + 1: end_index]
            #         moves = moves[end_index + 1:]
            if 'X' in play[1]:
                i = 0
                for move in moves:
                    thrown_out = move.find('X')
                    if thrown_out != -1:
                        runner = move[thrown_out - 1]
                        fielding_info = move.find('/')
                        if fielding_info == -1:
                            fielding_info = move.find(')')
                        if move[move.find('(')+ 1: move.find(')')] != 'TH':
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
        
        if event[0][0:2] in set(['WP', 'PB', 'BK', 'CS', 'OA', 'DI', 'PO']) and len(play) > 1:
            if '-H' in play[1]:
                moves = play[1].split(';')
                i = 0
                for move in moves:
                    scored = move.find('-H')
                    if scored != -1 and '(NR)' not in move:
                        moves[i] = move[:scored] + '(NR)' + move[scored:]
                    i+=1
                play[1] = ';'.join(moves)
        
        # if event[0][0:3] == 'SBH':
        #     if '(NR)' not in event[0]:
        #         print(data['play'])
        #         event[0] = event[0][0:3] + '(NR)' + event[0][3:]
        #         play[0] = '/'.join(event)
        data['play'] = '.'.join(play)
        return data

