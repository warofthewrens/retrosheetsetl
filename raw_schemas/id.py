from marshmallow import Schema, fields, ValidationError

# TODO: fill in all teams
team_set = ['ANA', 'ARI', 'ATL', 'BAL', 'BOS', 'CHA', 'CHN', 'CIN', 'CLE', 'COL', 'DET', 'HOU',
            'HOU', 'KCA', 'LAN', 'MIA', 'MIL', 'MIN', 'NYA', 'NYN', 'OAK', 'PHI', 'PIT', 'SEA',
            'SFN', 'SLN', 'TBA', 'TEX', 'TOR', 'WAS']

def validate_id(game_id):
    print('validating...')
    if game_id[:3] not in team_set:
        raise ValidationError('Not a valid team in id')
    if not game_id[3:12].isdigit():
        raise ValidationError('Not a valid game id')
    if int(game_id[3:7]) not in range(1900, 2050) or int(game_id[7:9]) not in range(3, 11) or int(game_id[9:11]) not in range(1, 32):
        raise ValidationError('Not a valid date in id')
    if int(game_id[11]) not in range(0,3):
        raise ValidationError('Not a valid game number in id')

class GameId(Schema):
    game_id = fields.String(validate=validate_id)

