from backend.lib.multiprocessor import job_pool
import time
import pandas as pd
from backend.settings import JOB_NUMBER, LOG_LEVEL
from numpy import nan
import logging

logging.basicConfig(level=LOG_LEVEL)

def create_dataframe_in_parallel(item_list, index=None, columns_types=None):
    start = time.perf_counter()
    jobs = job_pool(item_list, JOB_NUMBER, create_dataframe, index=index, columns_types=columns_types)
    end = time.perf_counter()
    logging.debug('time taken {}'.format(end - start))
    logging.debug('concatenating dataframes...')
    result = pd.concat(jobs)
    result['word'].str.encode('utf8')
    if index:
        logging.debug('setting index to {}'.format(index))
        result = result.set_index(index)
    return result


def create_dataframe(item_list, columns_types=None, index=None):
    assert (isinstance(item_list, list))
    assert (isinstance(item_list[0], dict))
    logging.debug('creating df')
    if index:
        df = pd.DataFrame(item_list, index=index)
    else:
        df = pd.DataFrame(item_list)

    logging.debug('filling Nones')
    df.fillna(value=nan, inplace=True)

    if columns_types:
        logging.debug('setting types of columns...')
        for c, d in columns_types:
            df[c] = df[c].astype(d)

    return df
