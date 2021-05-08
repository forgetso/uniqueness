import pandas as pd
from pathlib import Path
import re
import requests
from hyper.contrib import HTTP20Adapter


def setup_session(user_agent, base_url, headers=None, cookies=None):
    if not cookies:
        cookies = {}
    session = requests.Session()
    if not headers:
        headers = {'User-Agent': user_agent}
    initial_request = session.get(url=base_url, headers=headers)
    cookies = {**session.cookies.get_dict(), **cookies}
    adapter = HTTP20Adapter(headers=headers, cookies=cookies)
    session.mount(prefix='', adapter=adapter)
    return session, initial_request


def batch(iterable, batch_size=1):
    l = len(iterable)
    for ndx in range(0, l, batch_size):
        yield iterable[ndx:min(ndx + batch_size, l)]


def load_words_from_file(file, filter=None):
    # check if the file is json otherwise assume csv
    words_col = 'words'
    with open(file) as f:
        line = f.readline()
        if line.startswith('{'):
            df = pd.DataFrame(pd.read_json(file, typ='series'))
            df.index.name = words_col
            df.reset_index(inplace=True)
        elif '\t' in line:
            df = pd.read_csv(file, delimiter='\t')
        else:
            df = pd.read_csv(file)
        for col in df.columns:
            # assume that the column with the words is called word...
            if re.match('^word.*', col, flags=re.IGNORECASE):
                df = df.rename({col: words_col}, axis=1)
        if words_col not in df.columns:
            # otherwise assume it is the first column in the file
            df = df.rename({0: words_col})
        if filter:
            df = df.query(filter)
    return df['words'].astype(str)


def save_data_to_file(output_file, existing_data, df):
    try:
        df = pd.concat((existing_data, df))
        df = df.drop_duplicates()
    except ValueError:
        pass
    df.to_csv(output_file)


def get_progress(output_file):
    existing_data = pd.DataFrame([])
    output_path = Path(output_file)
    if output_path.exists():
        existing_data = pd.read_csv(output_file)
        for col in existing_data.columns:
            if col.startswith('Unnamed:'):
                existing_data.drop(col, axis=1, inplace=True)
    return existing_data


def remove_downloaded_words_from_list(existing_data, words_to_scrape, col):
    if len(existing_data):
        words_to_scrape = words_to_scrape[~words_to_scrape.isin(existing_data[col])]
    return words_to_scrape
