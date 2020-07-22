from bs4 import BeautifulSoup

import requests
import pandas as pd

def extract_fangraphs():
    url = 'https://www.fangraphs.com/guts.aspx?type=cn'

    html = requests.get(url).text
    bs = BeautifulSoup(html, 'lxml')
    # tables = bs.find_all('table')
    table = bs.find('table', 'rgMasterTable')
    woba_df = pd.read_html(table.prettify())[0]
    return woba_df 

def extract_park_factors(year):
    '''
    extract the park factors for a given year
    '''
    url = 'https://www.fangraphs.com/guts.aspx?type=pf&teamid=0&season='+year

    html = requests.get(url).text
    bs = BeautifulSoup(html, 'lxml')
    table = bs.find('table', 'rgMasterTable')
    pf_df = pd.read_html(table.prettify())[0]
    return pf_df

