from marshmallow import Schema, fields
from raw_schemas.data import Data
from raw_schemas.id import GameId
from raw_schemas.info import Info
from raw_schemas.play import Play
from raw_schemas.sub import Sub
from raw_schemas.start import Start


class Game(Schema):
    game_id = fields.Nested(GameId)
    info = fields.Nested(Info)
    lineup = fields.List(fields.Nested(Start))
    plays = fields.List(fields.Nested(Play))
    subs = fields.List(fields.Nested(Sub))
    data = fields.List(fields.Nested(Data))
