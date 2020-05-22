from backend.lib.multiprocessor import job_pool
import time
import pandas as pd
from backend.settings import JOB_NUMBER, LOG_LEVEL
import numpy as np
import logging

logging.basicConfig(level=LOG_LEVEL)


def create_dataframe_in_parallel(item_list, index=None, columns_types=None):
    start = time.perf_counter()
    jobs = job_pool(item_list, JOB_NUMBER, create_dataframe, columns_types=columns_types)
    end = time.perf_counter()
    logging.debug('time taken {}'.format(end - start))
    logging.debug('concatenating dataframes...')
    result = pd.concat(jobs)
    column_names = list(result.columns.values)
    result['word'] = result['word'].str.decode('utf8')
    result['word'] = result['word'].str.lower()
    result['word'] = result['word'].str.encode('utf8')
    # TODO use columns_types
    non_numeric = ['word', 'part_of_speech']
    for column in result.columns:
        if column not in non_numeric:
            result[column] = result[column].astype(np.float)
    result = result.groupby(['word', 'part_of_speech'], as_index=True).agg(['mean'])
    # drop the part of speech index
    result = result.reset_index()
    # Order getting messed up with part_of_speech and ages being mixed because of group by
    for idn, nn in enumerate(non_numeric):
        if column_names[idn] != nn:
            column_names[column_names.index(nn)], column_names[idn] = column_names[idn], nn
    # re-index on word (or whatever is passed in)
    if index:
        logging.debug('setting index to {}'.format(index))
        result = result.set_index([index])

    result.columns = [x for x in column_names if x not in result.index.names]
    # an example of a word that should always be present
    # assert (len(result.loc[result.index.get_level_values(level='word') == b'enemy']) == 1)
    return result


def create_dataframe(item_list, columns_types=None):
    assert (isinstance(item_list, list))
    assert (isinstance(item_list[0], dict))
    logging.debug('creating df')
    df = pd.DataFrame(item_list)
    logging.debug('filling Nones')
    df.fillna(value=np.nan, inplace=True)
    if columns_types:
        logging.debug('setting types of columns...')
        for c, d in columns_types:
            df[c] = df[c].astype(d)
    return df
