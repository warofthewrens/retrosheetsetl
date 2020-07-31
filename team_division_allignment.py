import pandas as pd
from models.sqla_utils import ENGINE

def read_division_allignment():
    columns = ['current_id', 'old_id', 'league', 'division', 'Location', 'Nickname', 'alt', 'starting_date', 'end_date', 'City', 'State']
    div_df = pd.read_csv('divisionallignment.csv', header=None)
    div_df.columns = columns
    div_df['end_date'].fillna('12/31/2030', inplace=True)
    div_df['starting_date'] = pd.to_datetime(div_df['starting_date'], format='%m/%d/%Y')
    div_df['start_date'] = pd.DatetimeIndex(div_df['starting_date']).year
    div_df['end_date'] = pd.to_datetime(div_df['end_date'], format='%m/%d/%Y')
    div_df['end_year'] = pd.DatetimeIndex(div_df['end_date']).year
    div_df = div_df.drop('Location', axis=1).drop('Nickname', axis=1).drop('alt', axis=1).drop('City', axis=1).drop('State', axis=1).drop('starting_date', axis=1).drop('end_date', axis=1)
    
    div_df.to_sql(
        'divisions',
        con=ENGINE
    )
    print(div_df)

read_division_allignment()