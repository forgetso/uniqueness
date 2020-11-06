from backend.db.layer import get_all_measures_raw, get_all_measures_processed, save_all_measures
from backend.lib.multiprocessor import job_pool
import pandas as pd
from backend.settings import JOB_NUMBER, LOG_LEVEL, DATA_FOLDER
import numpy as np
import logging
import time
from backend.lib.helpers.dataframedebug import time_taken
import argparse
from nltk import SnowballStemmer, LancasterStemmer
from backend.lib.helpers.dataframedebug import dfdebug
from lemminflect import getAllLemmas
from backend.lib.us2gb import us2gb
from pathlib import Path
from backend.lib.sharedmodel import SharedModelManager
from backend.lib.sense2vec.sense2vec import Sense2Vec
import gc
import cupy

logging.getLogger().setLevel(logging.DEBUG)
UNKNOWN_CATEGORY = '<unknown>'

DECIMAL_PLACES = 3
SIMILARITY_MODEL = 'wikipedia'
WORD_MEASURES_DTYPES = [
    ('word', '|S'),
    ('word_lemma', 'str'),
    ('age', 'f'),
    ('part_of_speech', 'category'),
    ('arousal', 'f'),
    ('dominance', 'f'),
    ('valence', 'f'),
    ('bigram_count', 'f'),
    ('bigram_freq', 'f'),
    ('frequency', 'f'),
    ('ortho_frequency', 'f'),
    ('ortho_neighbours', 'f'),
    ('trigram_count', 'f'),
    ('trigram_freq', 'f'),
    ('unconstrained_bigram_count', 'f'),
    ('unconstrained_bigram_freq', 'f'),
    ('unconstrained_trigram_count', 'f'),
    ('unconstrained_trigram_freq', 'f'),
    ('unconstrained_unigram_count', 'f'),
    ('unconstrained_unigram_freq', 'f'),
    ('unigram_count', 'f'),
    ('unigram_freq', 'f'),
    ('pknown', 'f'),
    ('prevalence', 'f'),
    ('nphon', 'f'),
    ('scdcnt', 'f'),
    ('sfreq', 'f'),
    ('strbpav', 'f'),
    ('strdens', 'f'),
    ('strlcpospav', 'f'),
    ('strpospav', 'f'),
    ('strtpav', 'f'),
    ('unsbpav', 'f'),
    ('unsdens', 'f'),
    ('unslcpospav', 'f'),
    ('unspospav', 'f'),
    ('unstpav', 'f'),
    ('concreteness_measure', 'f'),
    ('sstem', 'str'),
    ('lstem', 'str'),
]

ss = SnowballStemmer('english')
ls = LancasterStemmer()


def format_word_data(cur, save=None):
    item_dtype = WORD_MEASURES_DTYPES
    words = []
    word = None
    for item in cur:
        row_dict = dict(item)
        word_set = {row_dict['c_word'], row_dict['a_word'], row_dict['o_word'], row_dict['e_word'], row_dict['p_word'],
                    row_dict['phon_word']}
        word_set.discard(None)
        for w in ['c_word', 'a_word', 'o_word', 'e_word', 'p_word', 'phon_word']:
            row_dict.pop(w)
        for word in word_set:
            break
        assert (word is not None)
        row_dict['word'] = word
        row_dict['startswith'] = ord(word[0]) - 64
        words.append(row_dict)
    words = sorted(words, key=lambda k: k['word'])
    words = pd.DataFrame(words)
    words['changes'] = words['startswith'] - words['startswith'].shift(1) != 0
    words.drop('startswith', inplace=True, axis=1)
    # set up model share
    shm_models = SharedModelManager(models=[SIMILARITY_MODEL], from_disk=True)

    # split the job into multiprocesses, making sure that words starting with the same letter remain in the same group
    jobs = job_pool(words, JOB_NUMBER, format_word_data_multiprocess, split_col='changes', item_dtype=item_dtype)
    total_lines = sum([len(x) for x in jobs])
    logging.debug('jobs returned {} lines'.format(total_lines))
    df = pd.concat(jobs)

    # this one has to be run on a single core for now to avoid having to learn how to share memory blocks on GPU
    # https://docs.chainer.org/en/v1.3.1/_modules/cupy/cuda/memory.html
    # https://stackoverflow.com/questions/57752516/how-to-use-cuda-pinned-zero-copy-memory-for-a-memory-mapped-file/57762413#57762413

    # similarity replacement based on most_similar word from 3rd part model
    start = time.time()
    for model_name, model in shm_models.data.items():
        logging.debug('Using model {}'.format(model_name))
        df = replace_nulls_with_nearest_non_nulls_by_similarity(df, model)
        # should currently only be passing one model in for similarities
        break
    time_taken(start, replace_nulls_with_nearest_non_nulls_by_similarity)

    logging.debug('concat returned {} lines'.format(len(df)))
    return df


def format_word_data_multiprocess(df, item_dtype, shared_model):
    item_dtype_dict = {k: v for k, v in item_dtype}

    # set data types for each column
    start = time.time()
    df = set_data_types(df, item_dtype)
    time_taken(start, set_data_types)

    # merge duplicates
    start = time.time()
    df = merge_word_records(df, item_dtype_dict)
    time_taken(start, merge_word_records)

    # add missing lemmas
    start = time.time()
    df = add_missing_lemmas(df)
    time_taken(start, add_missing_lemmas)

    # british - american spellings
    start = time.time()
    df = replace_nulls_with_non_nulls_by_american_british_spellings(df)
    time_taken(start, replace_nulls_with_non_nulls_by_american_british_spellings)

    # lemma
    start = time.time()
    df = replace_nulls_with_nearest_non_nulls_by_lemma(df)
    time_taken(start, replace_nulls_with_nearest_non_nulls_by_lemma)

    # stems (same as lemma but calculated in this process)
    start = time.time()
    df = replace_nulls_with_nearest_non_nulls_by_stems(df)
    time_taken(start, replace_nulls_with_nearest_non_nulls_by_stems)

    # form
    start = time.time()
    df = replace_nulls_with_nearest_non_nulls_by_form(df)
    time_taken(start, replace_nulls_with_nearest_non_nulls_by_form)

    return df


def set_data_types(df, columns_types):
    if columns_types:
        logging.debug('setting types of columns...')
        for c, d in columns_types:
            try:
                # logging.debug('setting {} as {}'.format(c, d))
                df[c] = df[c].astype(d)
                df[c].replace({None: np.NaN}, inplace=True)

            except KeyError:
                # logging.debug('Column {} is not present in dataframe'.format(c))
                pass
    logging.debug('processing {} rows'.format(len(df)))
    return df


def merge_word_records(df, item_dtype_dict):
    # join any str columns with commas and return the mean of all the numeric columns
    # TODO consider passing back words with their parts of speech and try to infer which part of speech is intended in
    #  the response
    # TODO work out how to aggregate each column. taking the mean of age doesn't make sense - should it be the max? min?
    for c, d in item_dtype_dict.items():
        if c in df.columns:
            if d != 'f':
                if df[c].dtype.str.startswith('|S'):
                    df[c] = df[c].str.decode('utf8').astype(str)
                # this is for the join function, unique_chars
                df[c].replace({None: UNKNOWN_CATEGORY}, inplace=True)
                df[c].replace({'None': UNKNOWN_CATEGORY}, inplace=True)
                df[c].replace({'<NA>': UNKNOWN_CATEGORY}, inplace=True)
                df[c].replace({np.nan: UNKNOWN_CATEGORY}, inplace=True)
    unique_chars = lambda x: ','.join([x for x in x.unique() if x != UNKNOWN_CATEGORY])
    agg_dict = {x: (unique_chars if item_dtype_dict[x] != 'f' else 'mean') for x in df.columns if x in item_dtype_dict}
    agg_dict.pop('word', None)
    df = df.replace(0, np.NaN)
    df['word'] = df['word'].str.encode('utf8')
    df = df.groupby('word').agg(agg_dict)
    # make sure there are no nan words in the index
    assert len(df[df.index.isnull()]) == 0
    df.replace(UNKNOWN_CATEGORY, np.nan)
    return df


def add_missing_lemmas(df):
    # replace empty string lemmas with None
    df['word_lemma'] = df['word_lemma'].replace({'': None})

    # locate words where lemmas are missing
    nulls = df.loc[df['word_lemma'].isnull() == True].copy()
    nulls.reset_index(inplace=True)

    # obviously if there are no nulls, return the dataframe
    if len(nulls) == 0:
        logging.debug('add_missing_lemmas: Did not find any null lemmas')
        return df

    nulls['word'] = nulls['word'].str.decode('utf8')
    logging.debug('add_missing_lemmas: There are {} words without lemmas'.format(len(nulls)))

    # get the word_lemma and the part of speech according to lemminflect library
    nulls[['word_lemma', 'part_of_speech']] = nulls.apply(lambda row: pd.Series(get_lemma(getAllLemmas, row['word'])),
                                                          axis=1)

    # reduce the nulls list to only those words that have non null lemma or non null part of speech
    nulls = nulls[(nulls['word_lemma'].notnull()) | (nulls['part_of_speech'].notnull())]

    nulls['word'] = nulls['word'].str.encode('utf8')
    nulls.set_index('word', inplace=True)
    logging.debug('add_missing_lemmas: Lemmas returned by lemminflect {}'.format(np.sum(nulls['word_lemma'].notnull())))

    # take the value from the original df if it exists, else take the value from the nulls df
    df['word_lemma'] = df['word_lemma'].combine_first(nulls['word_lemma'])
    df['part_of_speech'] = df['part_of_speech'].combine_first(nulls['part_of_speech'])
    df['word_lemma'] = df['word_lemma'].replace({None: np.nan})
    df['part_of_speech'] = df['part_of_speech'].replace({None: np.nan})

    assert np.nan not in df.index

    return df


def get_lemma(getAllLemmas, word):
    word = str(word)
    lemmas = getAllLemmas(word)
    tmp = (None, None)
    if lemmas:
        # TODO work out whether this is actually worth doing - maybe just return lemmas that are shorter than the word
        tmp = (list(set([l[0] for l in lemmas.values()]))[0], ','.join(sorted(lemmas.keys())).lower())
    return tmp


def replace_nulls_with_non_nulls_by_american_british_spellings(df):
    gb2us = {v: k for k, v in us2gb.items()}
    words_decoded = df.index.get_level_values('word').str.decode('utf8')
    # replace is not a function on indexes https://github.com/pandas-dev/pandas/issues/19495
    df['us_version'] = words_decoded.map(lambda x: index_replace(x, gb2us))
    df['gb_version'] = words_decoded.map(lambda x: index_replace(x, us2gb))
    # first take any US measurements and copy into the GB words - it's far more likely measures will be for US spellings
    us_indexed = df[~df['us_version'].isnull()].reset_index(drop=True).copy()
    us_indexed['us_version'] = us_indexed['us_version'].str.encode('utf8')
    us_indexed = us_indexed.rename({'us_version': 'word'}, axis=1).reset_index(drop=True).set_index('word')
    us_indexed.drop(columns=['gb_version'], inplace=True)
    df.update(us_indexed, overwrite=False)
    # then pass any measurements that were present for GB words but missing for US words, onto the US words
    gb_indexed = df[~df['gb_version'].isnull()].reset_index(drop=True).copy()
    gb_indexed['gb_version'] = gb_indexed['gb_version'].str.encode('utf8')
    gb_indexed = gb_indexed.rename({'gb_version': 'word'}, axis=1).reset_index(drop=True).set_index('word')
    gb_indexed.drop(columns=['us_version'], inplace=True)
    df.update(gb_indexed, overwrite=False)
    df.drop(columns=['us_version', 'gb_version'], inplace=True)
    return df


def index_replace(x, d):
    return d[x] if x in d else None


def replace_nulls_with_nearest_non_nulls_by_lemma(df):
    # this copies the part of speech, is this necessary?
    selected_columns = df.copy()[[x for x in df.columns if x != 'word']]
    wordsdfcopy = selected_columns.reset_index(drop=True)
    wordsdfcopy = wordsdfcopy.set_index('word_lemma').dropna(how='all')
    wordsdfcopy = wordsdfcopy.loc[wordsdfcopy.index.dropna()]
    wordsdfcopy = wordsdfcopy.groupby(['word_lemma']).mean()
    # wordsdfcopy.reset_index(level='part_of_speech', inplace=True)
    df = df.reset_index()
    df = df.set_index('word_lemma')
    df.update(wordsdfcopy, overwrite=False)
    df = df.reset_index()
    df.set_index('word', inplace=True)
    if 'index' in df.columns:
        df.rename({'index': 'word'}, inplace=True, axis=1)

    return df


def replace_nulls_with_nearest_non_nulls_by_stems(df):
    df['sstem'] = df.index.get_level_values('word').str.decode('utf8').map(ss.stem)
    df['lstem'] = df.index.get_level_values('word').str.decode('utf8').map(ls.stem)

    # group up and mean the data by the two types of stems
    snowball_groupby = df.groupby('sstem', as_index=True).agg(['mean'])
    snowball_groupby.columns = [x[0] for x in snowball_groupby.columns]
    snowball_groupby.index.rename('word', inplace=True)
    lancaster_groupby = df.groupby('lstem', as_index=True).agg(['mean'])
    lancaster_groupby.columns = [x[0] for x in lancaster_groupby.columns]
    lancaster_groupby.index.rename('word', inplace=True)

    # first, pull data from the df that is missing from stems in the groupbys
    lancaster_groupby.index = lancaster_groupby.index.str.encode('utf8')
    snowball_groupby.index = snowball_groupby.index.str.encode('utf8')
    snowball_groupby.update(lancaster_groupby, overwrite=False)
    lancaster_groupby.update(df, overwrite=False)

    # then add the data back into the df by matching onto the sstems
    snowball_groupby.index = snowball_groupby.index.str.decode('utf8')
    df.reset_index(inplace=True)
    df.set_index('sstem', inplace=True)
    df.update(snowball_groupby, overwrite=False)

    # then add the data back into the df by matching onto the lstems
    lancaster_groupby.index = lancaster_groupby.index.str.decode('utf8')
    df.reset_index(inplace=True)
    df.set_index('lstem', inplace=True)
    df.update(lancaster_groupby, overwrite=False)
    df.reset_index(inplace=True)

    # then add the data back into the df by matching sstems onto the lstems
    # e.g. qatar( sstem==qatar) qatari (lstem==qatar)
    df.set_index('sstem', inplace=True)
    df.update(lancaster_groupby, overwrite=False)
    df.reset_index(inplace=True)

    df.set_index('word', inplace=True)

    return df


def replace_nulls_with_nearest_non_nulls_by_similarity(df, model):
    # use a word embedding vector space to find similar words (e.g. sofa - couch) and take metrics
    # concreteness, age of acquisition, valence, dominance, arousal, frequency, prevalence if they are not present
    df.reset_index(inplace=True)
    df['word'] = df['word'].str.decode('utf8')
    # most similar takes ages so we need to limit what we run it on

    subset = df.loc[
        df[['concreteness_measure', 'age', 'valence', 'dominance', 'arousal', 'frequency', 'prevalence']].isnull().any(
            axis=1)]
    logging.debug('Checking most similar for {} words'.format(len(subset)))

    lexemes = subset['word'].apply(model.vocab.__getitem__)
    vectors = lexemes.apply(lambda x: x.vector)
    vectors = vectors.tolist()
    vectors = np.array(vectors, dtype='float64')
    keys = lexemes.apply(lambda x: x.orth).values
    # to get spacy to work on the GPU we need to put this array on the GPU first
    cupy_model = cupy.array(model.vocab.vectors.data.base)
    cupy_vectors = cupy.array(vectors)
    result_keys, _, scores = most_similar(cupy_vectors, model=model, n=2, batch_size=16, data=cupy_model)
    result = list(zip(result_keys.flatten(), scores.flatten()))
    result = [(model.vocab.strings[key], score) for key, score in result if key not in keys]
    subsetsim = pd.concat([subset.reset_index(drop=True), pd.DataFrame(result, columns=['sim_word', 'sim_score'])],
                          axis=1,
                          ignore_index=True)
    for metric in ['concreteness_measure', 'age', 'valence', 'dominance', 'arousal', 'frequency', 'prevalence']:
        logging.debug('Checking for nulls in metric - {}'.format(metric))

        nulls = df[(df[metric].isnull()) & (df['most_similar'].notnull())]
    #
    # gc.collect()
    # df['word'] = df['word'].str.encode('utf8')
    # df.set_index('word', inplace=True)

    return df


def most_similar(queries, *, model, batch_size=1024, n=1, sort=True, data=None):
    """For each of the given vectors, find the n most similar entries
    to it, by cosine.

    Queries are by vector. Results are returned as a `(keys, best_rows,
    scores)` tuple. If `queries` is large, the calculations are performed in
    chunks, to avoid consuming too much memory. You can set the `batch_size`
    to control the size/space trade-off during the calculations.

    queries (ndarray): An array with one or more vectors.
    batch_size (int): The batch size to use.
    n (int): The number of entries to return for each query.
    sort (bool): Whether to sort the n entries returned by score.
    data (arr): Allow different array of data - i.e. if array has since been loaded onto GPU
    RETURNS (tuple): The most similar entries as a `(keys, best_rows, scores)`
        tuple.
    """
    if data is None:
        data = model.vectors.data.base
    filled = sorted(list({row for row in model.vocab.vectors.key2row.values()}))

    xp = cupy

    norms = xp.linalg.norm(data[filled], axis=1, keepdims=True)
    norms[norms == 0] = 1
    vectors = data[filled] / norms

    best_rows = xp.zeros((queries.shape[0], n), dtype='i')
    scores = xp.zeros((queries.shape[0], n), dtype='f')
    # Work in batches, to avoid memory problems.
    for i in range(0, queries.shape[0], batch_size):
        batch = queries[i: i + batch_size]
        batch_norms = xp.linalg.norm(batch, axis=1, keepdims=True)
        batch_norms[batch_norms == 0] = 1
        batch /= batch_norms
        # batch   e.g. (1024, 300)
        # vectors e.g. (10000, 300)
        # sims    e.g. (1024, 10000)
        sims = xp.dot(batch, vectors.T)
        best_rows[i:i + batch_size] = xp.argpartition(sims, -n, axis=1)[:, -n:]
        scores[i:i + batch_size] = xp.partition(sims, -n, axis=1)[:, -n:]

        if sort and n >= 2:
            sorted_index = xp.arange(scores.shape[0])[:, None][i:i + batch_size], xp.argsort(scores[i:i + batch_size],
                                                                                             axis=1)[:, ::-1]
            scores[i:i + batch_size] = scores[sorted_index]
            best_rows[i:i + batch_size] = best_rows[sorted_index]

    if isinstance(best_rows, cupy.core.core.ndarray):
        best_rows = cupy.asnumpy(best_rows)
    for i, j in np.ndindex(best_rows.shape):
        best_rows[i, j] = filled[best_rows[i, j]]
    # Round values really close to 1 or -1
    scores = xp.around(scores, decimals=4, out=scores)
    # Account for numerical error we want to return in range -1, 1
    scores = xp.clip(scores, a_min=-1, a_max=1, out=scores)
    row2key = {row: key for key, row in model.vocab.vectors.key2row.items()}
    keys = xp.asarray(
        [[row2key[row] for row in best_rows[i] if row in row2key]
         for i in range(len(queries))], dtype="uint64")
    if isinstance(scores, cupy.core.core.ndarray):
        scores = cupy.asnumpy(scores)
    if len(keys) and isinstance(keys[0], cupy.core.core.ndarray):
        keys = np.array([cupy.asnumpy(key) for key in keys])
    return (keys, best_rows, scores)


def replace_nulls_with_nearest_non_nulls_by_form(df):
    # TODO make this smarter. Maybe do it column by column? A word can have some null values in col X,Y and be missing Z.
    #  another similar word might have a value for Z but be missing X and Y. Currently, the copying would only work in
    #  one direction - from the word with the least nulls to the word with more nulls.
    # 2 is a totally arbitrary number...
    null_filter = df.isnull().sum(axis=1) >= 2
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
    dupe_words = value_counts.loc[value_counts > 1].index
    dupes = longer_words_df.loc[longer_words_df['null_word'].isin(dupe_words)].copy()
    non_dupes = longer_words_df.loc[~longer_words_df['null_word'].isin(dupe_words)].copy()
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
    non_numeric = ['word', 'part_of_speech', 'sstem', 'lstem']
    for column in result.columns:
        if column not in non_numeric:
            result[column] = result[column].astype(np.float)
    if result['part_of_speech'].dtype.name == 'category':
        result['part_of_speech'].cat.add_categories(UNKNOWN_CATEGORY, inplace=True)
    result['part_of_speech'].fillna(value=UNKNOWN_CATEGORY, inplace=True)

    result = result.groupby(['word', 'part_of_speech', 'sstem', 'lstem'], as_index=True).agg(['mean']).sort_index(
        level='word')
    result = result.dropna(axis=0, how='all')
    result = result.reset_index()
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


def get_all_measures(cur=None, rebuild=None):
    if rebuild:
        if not cur:
            cur = get_all_measures_raw()
        item_list = format_word_data(cur)
        df = create_all_measures_dataframe(item_list, index='word')
        # fill any nan values with the mean of the column
        # df.fillna(df.mean(), inplace=True)
        # df = df.round(DECIMAL_PLACES)
        save_all_measures(df.reset_index().replace({np.nan: None}))
    else:
        cur = get_all_measures_processed()
        df = pd.DataFrame([x for x in cur], columns=[desc[0] for desc in cur.description])
        df = set_data_types(df, WORD_MEASURES_DTYPES)
        df.drop('id', axis=1, inplace=True)
    return df


def setup():
    parser = argparse.ArgumentParser(description='Create a table of words and all possible measures')
    parser.add_argument('--rebuild', action='store_true', help="rebuild the measures and save to database")
    args = parser.parse_args()
    get_all_measures(rebuild=args.rebuild)


if __name__ == "__main__":
    setup()
