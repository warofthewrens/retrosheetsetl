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

extract_fangraphs()
