
def transform_plays(plays, subs, rows, context):
    return


def transform_game(game):
    rows = {table: [] for table in ['plate_appearance', 'game']}
    context = {'game_id': game['game_id'], 'date': game['info']['date']}
    transform_plays(game['plays'], game['subs'], rows, context)
    return