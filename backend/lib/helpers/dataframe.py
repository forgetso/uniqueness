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
    result['word'] = lower_and_decode(result['word'])
    # TODO use columns_types to figure out which columns are not numeric
    non_numeric = ['word', 'part_of_speech']
    for column in result.columns:
        if column not in non_numeric:
            result[column] = result[column].astype(np.float)
    result = result.groupby(['word', 'part_of_speech'], as_index=True).agg(['mean']).dropna(axis=0, how='all')
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

    return result


def lower_and_decode(col):
    col_copy = col.copy()
    try:
        col = col.str.decode('utf8')
        col = col.str.lower()
    except Exception as e:
        logging.debug('Error decoding word column :{}'.format(e))
        col = col_copy.astype(str).str.lower()
    return col


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


def unpack_splits(splits, length=None):
    # return a list of individual words from columns of word lists and include the original column index of the list
    unpacked = None
    if not length:
        length = 2
    try:
        unpacked = np.vstack([np.c_[x, [idx] * len(x)] for idx, x in enumerate(splits) if len(x) >= length])
    except ValueError:
        # the array may be empty if there are no multi word strings, causing concatenate to fail with
        # "ValueError: need at least one array to concatenate"
        pass
    return unpacked


def get_unpacked_words(df):
    cue_clean = df['cue_clean']
    response_clean = df['response_clean']
    # get the words from the columns as big lists of individual words (instead of multi word cues/responses)
    unpacked_words = pd.DataFrame(np.unique(np.vstack((cue_clean, response_clean))), columns=['word']).sort_values(
        by=['word'])
    unpacked_words = unpacked_words[~unpacked_words['word'].isnull()]
    unpacked_words.replace("", np.nan, inplace=True)
    unpacked_words.dropna(inplace=True)
    return unpacked_words
