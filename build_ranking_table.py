from models.sqla_utils import ENGINE
import pandas as pd

positions = {
    2: 'Catcher',
    3: 'First Baseman',
    4: 'Second Baseman',
    5: 'Third Baseman',
    6: 'Shortstop',
    7: 'Left Field',
    8: 'Center Field',
    9: 'Right Field',
    10: 'Designate Hitter'
}


def query_position(position, year):
    query = '''SELECT team, wRAA, @curRank := @curRank + 1 AS rank
            FROM retrosheet.TeamPosition, (select @curRank := 0) r 
            where year ='''+ str(year) + ''' and position_code='''+ str(position) + '''
            order by wRAA desc;
            '''
    
    position_rank_df = pd.read_sql_query(
        query,
        con=ENGINE
    )
    position_rank_df.columns = ['team', 'wRAA', '_'.join(positions[position].split()) + '_Rank']
    position_rank_df = position_rank_df.drop('wRAA', axis=1)
    position_rank_df['year'] = year
    return position_rank_df

def main():
    columns = ['team', 'year']
    #position_rank_df = pd.DataFrame(columns=['team', 'Catcher_Rank', 'position_x', 'year', 'First Baseman_Rank_x', 'position_y', 'Second Baseman_Rank_x', 'Third Baseman_Rank_x', 'Shortstop_Rank_x', 'Left Field_Rank_x', 'Center Field_Rank_x', 'Right Field_Rank_x', 'Catcher_Rank_y', 'First Baseman_Rank_y', 'Second Baseman_Rank_y', 'Third Baseman_Rank_y', 'Shortstop_Rank_y', 'Left Field_Rank_y', 'Center Field_Rank_y', 'Right Field_Rank_y'])
    position_rank_df = pd.DataFrame(columns=['team', 'year'])
    for year in range(1990, 2020):
        if year == 1994:
            continue
        year_rank_df = pd.DataFrame(columns=columns)
        for position in range(2, 10):
            year_rank_df = pd.merge(year_rank_df, query_position(position, year), how='right', left_on = ['team', 'year'], right_on = ['team', 'year'])
        position_rank_df = position_rank_df.append(year_rank_df)
            # position_rank_df = position_rank_df.join(query_position(position, year))
    print(position_rank_df)    
    position_rank_df.to_sql(
        'TeamPositionRank', 
        con=ENGINE
    )

main()