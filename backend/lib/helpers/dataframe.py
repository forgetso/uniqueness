from backend.lib.multiprocessor import job_pool
import time
import pandas as pd
from backend.settings import JOB_NUMBER, LOG_LEVEL
import numpy as np
import logging
from backend.lib.helpers.dfdebug import dfdebug, get_linenumber

logging.basicConfig(level=LOG_LEVEL)
UNKNOWN_CATEGORY = 'unknown'


def format_word_data(cur, item_dtype):
    words = []
    for item in cur:
        row_dict = dict(item)
        word_set = {row_dict['c_word'], row_dict['a_word'], row_dict['o_word'], row_dict['e_word'],
                    row_dict['phon_word']}
        word_set.discard(None)
        for w in ['c_word', 'a_word', 'o_word', 'e_word', 'phon_word']:
            row_dict.pop(w)
        for word in word_set:
            break
        assert (word is not None)
        row_dict['word'] = word
        words.append(row_dict)
    words = sorted(words, key=lambda k: k['word'])
    # split the job into multiprocesses, making sure that words starting with the same letter remain in the same group
    jobs = job_pool(words, JOB_NUMBER * 2, format_word_data_multiprocess, groupby=['word', 0], item_dtype=item_dtype)
    total_lines = sum([len(x) for x in jobs])
    logging.debug('jobs returned {} lines'.format(total_lines))
    result = pd.concat(jobs)
    logging.debug('concat returned {} lines'.format(len(result)))
    dfdebug(result, 'word', b'i', get_linenumber())
    return result


def format_word_data_multiprocess(data, item_dtype):
    words = {}
    to_merge = {}
    # get a giant dict from the cursor
    for row_dict in data:
        word = row_dict['word']
        if word not in words:
            row_dict['word'] = word
            words[word] = row_dict
        else:
            to_merge.setdefault(word, [])
            to_merge[word].extend([row_dict, words[word]])
            words.pop(word)

    item_dtype_dict = {k: v for k, v in item_dtype}
    for word in to_merge:
        words[word] = merge_word_records(to_merge[word], item_dtype_dict)

    wordsdf = create_dataframe_from_dict(words, item_dtype)
    wordsdf = replace_nulls_with_nearest_non_nulls_by_lemma(wordsdf)
    wordsdf = replace_nulls_with_nearest_non_nulls_by_similarity(wordsdf)
    dfdebug(wordsdf, 'word', b'i', get_linenumber())
    return wordsdf


def merge_word_records(word_dicts, item_dtype_dict):
    # TODO should be using pandas for this
    new_dict = {}
    count_dict = {}

    for w in word_dicts:
        for k, v in w.items():
            count_dict.setdefault(k, 0)

            # if floats
            if item_dtype_dict[k] == 'f':
                if isinstance(v, (int, float)):
                    new_dict.setdefault(k, 0)
                    count_dict[k] += 1
                    new_dict[k] += v
            # otherwise strings
            else:
                new_dict.setdefault(k, [])
                if v is not None:
                    new_dict[k].append(v)

    # take the averages of the summed values if they were not None
    for k, v in new_dict.items():

        if item_dtype_dict[k] == 'f':
            if count_dict[k] > 0:
                new_dict[k] = new_dict[k] / count_dict[k]
        else:
            new_dict[k] = set(new_dict[k])
            new_dict[k] = ','.join(new_dict[k])

    for k in item_dtype_dict:
        if k not in new_dict:
            new_dict[k] = None

    return new_dict


def create_dataframe_from_dict(words, columns_types):
    df = pd.DataFrame.from_dict(words)
    df = df.T
    df = df.reset_index(drop=True)

    if columns_types:
        logging.debug('setting types of columns...')
        for c, d in columns_types:
            try:
                df[c] = df[c].astype(d)
            except KeyError:
                logging.debug('Column {} is not present in dataframe'.format(c))
                pass
    logging.debug('processing {} rows'.format(len(df)))
    df.set_index('word', inplace=True)
    df.replace({None: np.NaN}, inplace=True)
    df.replace({'<NA>': np.NaN}, inplace=True)
    df.replace({b'None': None}, inplace=True)
    return df


def replace_nulls_with_nearest_non_nulls_by_lemma(wordsdf):
    # this copies the part of speech, is this necessary?
    selected_columns = wordsdf.copy()[[x for x in wordsdf.columns if x != 'word']]
    wordsdfcopy = selected_columns.reset_index(drop=True)
    wordsdfcopy = wordsdfcopy.set_index('word_lemma').dropna(how='all')
    wordsdf = wordsdf.reset_index()
    wordsdf = wordsdf.set_index('word_lemma')
    wordsdf.update(wordsdfcopy, overwrite=False)
    wordsdf = wordsdf.reset_index()
    wordsdf.set_index('word', inplace=True)
    return wordsdf


def replace_nulls_with_nearest_non_nulls_by_similarity(df):
    null_filter = df.isnull().any(axis=1)
    wordsdf_nulls = df[null_filter]
    wordsdf_notnull = df[~null_filter]
    longer_words = wordsdf_notnull.index.get_level_values('word')[wordsdf_notnull.index.str.len() > 4]
    # return a list of tuples that can be used to update null values, e.g. (earthy, earth)
    # earthy can be updated with the values from earth
    longer_words_contained_in_nulls = [(y, x) for y in [y for y in wordsdf_nulls.index.get_level_values('word')]
                                       for x in longer_words if y.startswith(x) and len(y) > len(x)]
    longer_words_df = pd.DataFrame(longer_words_contained_in_nulls, columns=['null_word', 'non_null_word'])
    # if there are multiple versions of y (null_word) stored in longerwords_df, take the record where x (non_null_word)
    # is the longest word that fits in y that is not y itself
    value_counts = longer_words_df['null_word'].value_counts()
    dupe_words = value_counts[value_counts > 1].index
    dupes = longer_words_df[longer_words_df['null_word'].isin(dupe_words)]
    non_dupes = longer_words_df[~longer_words_df['null_word'].isin(dupe_words)]
    non_dupes.set_index('null_word', inplace=True)
    dupes['non_null_word_length'] = dupes['non_null_word'].str.len()
    dupes = dupes.sort_values(['null_word', 'non_null_word_length'], ascending=[True, False]).groupby(
        'null_word').first()
    dupes.drop(columns=['non_null_word_length'], inplace=True)
    longer_words_df = pd.concat([dupes, non_dupes], axis=0)
    assert (~longer_words_df.index.duplicated().any())
    longer_words_df = pd.merge(longer_words_df, df, how='left', left_on='non_null_word', right_index=True)
    longer_words_df.drop(['word_lemma', 'non_null_word'], inplace=True, axis=1)
    longer_words_df.index.rename('word', inplace=True)
    df.update(longer_words_df, overwrite=False)
    df.reset_index(inplace=True)
    return df


def create_all_measures_dataframe(df, index=None):
    df.reset_index(inplace=True)
    # in case the index has already been reset
    df.drop(columns=['index'], inplace=True)

    result = df
    result.drop(columns='word_lemma', inplace=True)
    column_names = list(result.columns.values)
    result['word'] = lower_and_decode(result['word'])
    # TODO use columns_types to figure out which columns are not numeric
    non_numeric = ['word', 'part_of_speech']
    for column in result.columns:
        if column not in non_numeric:
            result[column] = result[column].astype(np.float)
    if result['part_of_speech'].dtype.name == 'category':
        result['part_of_speech'].cat.add_categories(UNKNOWN_CATEGORY, inplace=True)
    result['part_of_speech'].fillna(value=UNKNOWN_CATEGORY, inplace=True)

    result = result.groupby(['word', 'part_of_speech'], as_index=True).agg(['mean']).sort_index(level='word')
    #result = result.dropna(axis=0, how='all')
    result = result.reset_index()
    dfdebug(result, 'word', 'i', get_linenumber())
    dfdebug(result, 'word', b'i', get_linenumber())
    # Order getting messed up with part_of_speech and ages being mixed because of group by
    for idn, nn in enumerate(non_numeric):
        if column_names[idn] != nn:
            column_names[column_names.index(nn)], column_names[idn] = column_names[idn], nn

    # re-index on word (or whatever is passed in)
    if index:
        logging.debug('setting index to {}'.format(index))
        result = result.set_index([index])
        result.index.rename(index)
    logging.debug('index names {}'.format(result.index.names))
    result.columns = [x for x in column_names if x not in result.index.names]
    dfdebug(result, 'word', 'i', get_linenumber())
    dfdebug(result, 'word', b'i', get_linenumber())
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
