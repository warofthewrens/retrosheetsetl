from marshmallow import Schema, fields, ValidationError
from raw_schemas.data import Data
from raw_schemas.id import GameId
from raw_schemas.info import Info
from raw_schemas.play import Play
from raw_schemas.sub import Sub
from raw_schemas.start import Start

team_set = set(['ANA', 'ARI', 'ATL', 'BAL', 'BOS', 'CHA', 'CHN', 'CIN', 'CLE', 'COL', 'DET', 'HOU',
            'KCA', 'LAN', 'MIA', 'MIL', 'MIN', 'NYA', 'NYN', 'OAK', 'PHI', 'PIT', 'SEA',
            'SFN', 'SLN', 'SDN', 'TBA', 'TEX', 'TOR', 'WAS'])

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
    print(game_id)

class Game(Schema):
    game_id = fields.String(validate=validate_id)
    info = fields.Nested(Info)
    lineup = fields.List(fields.Nested(Start))
    plays = fields.List(fields.Nested(Play))
    subs = fields.List(fields.Nested(Sub))
    data = fields.List(fields.Nested(Data))
