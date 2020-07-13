import pandas as pd
import matplotlib.pyplot as plt
from models.sqla_utils import ENGINE

player_data_df = pd.read_sql_table(
    'player',
    con = ENGINE
)


def plot_wraa():
    '''
    plots wraa for all players since 1990
    '''
    plt.xkcd()
    year = player_data_df['year']
    wraa = player_data_df['wRAA']
    player = player_data_df['player_id']

    plt.figure(figsize = (14,7))
    mortals = plt.scatter(year, wraa, facecolors='#002D72', alpha=.75, s=75)
    bonds = plt.scatter(player_data_df['year'][player_data_df['player_id'] == 'bondb001'], player_data_df['wRAA'][player_data_df['player_id'] == 'bondb001'], color='#fd5a1e', s=75)
    trout = plt.scatter(player_data_df['year'][player_data_df['player_id'] == 'troum001'], player_data_df['wRAA'][player_data_df['player_id'] == 'troum001'], color='#ba0021', s=75)
    plt.legend([bonds, trout, mortals], ['Barry Bonds', 'Mike Trout', 'Mortals'])
    # player_data_df.plot(x='year', y = 'wRAA', kind='scatter')
    plt.xticks = ([1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 
                   2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 
                   2010], 
                   ['1990', '', '', '', '', '1995', '', '', '', '', 
                    '2000', '', '', '', '', '2005', '', '', '', '',
                    '2010'])
    plt.title('wRAA since 1990')
    plt.show()

plot_wraa()