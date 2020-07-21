import pandas as pd
import pymysql
from os import walk
import time
import shutil
from models.sqla_utils import ENGINE, BASE, get_session
from models.player import Player
from models.team import Team
from parsed_schemas.player import Player as p
from extract import extract_roster_team, extract_game_data_by_year

def etl_league_adjusted_stats(year):
    session = get_session()
    league_data_df = pd.read_sql_table(
        'league',
        con=ENGINE
    )
    team_data_df = pd.read_sql_table(
        'team',
        con=ENGINE
    )
    mlb_data_df = league_data_df[(league_data_df['year'] == int(year)) & (league_data_df['league'] == 'MLB')]
    nl_data_df = league_data_df[(league_data_df['year'] == int(year)) & (league_data_df['league'] == 'NL')]
    al_data_df = league_data_df[(league_data_df['year'] == int(year)) & (league_data_df['league'] == 'AL')]
    team_data_df = team_data_df[team_data_df['year'] == int(year)]
    players = session.query(Player).filter(Player.year == int(year))

    players.update({Player.FIPR9: Player.iFIP + mlb_data_df.get('ciFIP').item()})
    for player in players:
        team = team_data_df[team_data_df.team == player.team]
        player.pFIPR9 = (player.FIPR9 / team['PPFp'].item())
        if team['league'].item() == 'NL':
            league_FIPR9 = nl_data_df['FIPR9'].item()
            player.RAAP9 = league_FIPR9 - player.pFIPR9
            
        elif team['league'].item() == 'AL':
            league_FIPR9 = al_data_df['FIPR9'].item()
            player.RAAP9 = league_FIPR9 - player.pFIPR9

        player.dRPW = (((((18 - (player.IP / player.GP)) * league_FIPR9) + ((player.IP/player.GP) * player.pFIPR9)) / 18) + 2) * 1.5
        player.WPGAA = player.RAAP9 / player.dRPW

        repl = 0.03 * (1 - (player.GS / player.GP)) + 0.12 * (player.GS / player.GP)

        player.WPGAR = player.WPGAA + repl
        player.WAR = player.WPGAR * (player.IP / 9)

    #     pf = team_data_df[team_data_df.team == player.team]['PPFp']

    session.commit()


