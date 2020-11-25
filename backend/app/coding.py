from backend.lib.helpers.string import syllables
from backend.lib.helpers.pairs import CueResponsePairs
from backend.db.layer import save_coded_pairs, delete_coded_pairs, get_word_filter
from backend.lib.setup import db_setup
from nltk.stem import PorterStemmer
import re
from backend.lib.multiprocessor import job_pool
import time
import argparse
from backend.settings import JOB_NUMBER, LOG_LEVEL, DATA_FOLDER, SEMANTIC_SPACE_FOLDER, WORD_FREQUENCIES
import os
import numpy as np
import pandas as pd
import logging
from backend.lib.measures import UNKNOWN_CATEGORY, DECIMAL_PLACES
from backend.lib.sharedmanager import SharedManager
from backend.lib.sharedmodel import SharedModelManager
import gc
from backend.app.ngrams import attach_ngrams
import jellyfish as jf
import cupy as cp

pd.options.mode.chained_assignment = "raise"
logging.basicConfig(level=LOG_LEVEL)
ps = PorterStemmer()

MIN_CUE_LENGTH = 3
PUNCTUATION_REGEX = re.compile(r"[\.\-\',_$%£\{\};&\(\)]=@")
DROP_BEFORE_SAVING = [
    "{}_clean",
    "{}_decoded",
    "{}_split",
    "{}_correction",
    # "{}_part_of_speech",
    # "{}_sense",
    "{}_bad_spell",
    "{}_best_sense",
    "{}_stem",
]
MODELS = ["s2v", "crawl", "wikipedia", "dict2vec", "attractrepel"]
LANG_MODELS = ["attractrepel"]
LANG = 'en'
PROVERB_THRESHOLD_SIMILARITY = 0.5


def main(
        save=None, source=None, word_filter=None, cur=None, rebuild=None, models=None, measures=None
):
    df = None
    shm_data = None
    shm_models = None
    try:
        if word_filter:
            word_filter = get_word_filter()

        # get the cue response pairs, filtering by word_filter
        pairs = CueResponsePairs(source=source, cur=cur, word_filter=word_filter)

        # setup data to be shared between the processes
        shm_data = SharedManager(
            word_filter=word_filter, rebuild=rebuild, from_disk=True
        )
        shm_models = SharedModelManager(models=models, from_disk=True)

        if save:
            delete_existing_data()

        start = time.perf_counter()
        logging.debug("dispatching jobs...")
        jobs = job_pool(
            pairs.distinct_pairs,
            JOB_NUMBER,
            get_coding,
            shared_data=shm_data.shares,
            shared_model=shm_models.shares,
            save=save,
            measures=measures
        )

        gc.collect()
        get_result(start, jobs, pairs.distinct_pairs)
    except KeyboardInterrupt:
        cleanup(shm_data, shm_models)
    finally:
        cleanup(shm_data, shm_models)
        cleanup(shm_data, shm_models)
    return df


def cleanup(shm_data, shm_models):
    logging.debug('attempting cleanup...')
    if shm_data is not None:
        shm_data.close_shared_memories()
    if shm_models is not None:
        shm_models.close_shared_memories()


def delete_existing_data():
    logging.debug("deleting existing coding")
    delete_coded_pairs()


def get_result(start, jobs, distinct_cue_response_pairs):
    df = pd.concat(jobs)
    end = time.perf_counter()
    logging.debug("time taken {}".format(end - start))
    # take the individual results and sum them
    totals = df.count()
    # check the percentages of each categorisation
    percent_totals = totals.divide(len(distinct_cue_response_pairs)) * float(100)
    print(percent_totals.to_string())
    pd.set_option("display.max_rows", df.shape[0] + 1)
    # proverbdf = df.loc[~df['s2v_similarity'].isnull() & ~df['proverb_similarity'].isnull()]
    df.to_csv(DATA_FOLDER + "/coding.csv")


def get_lang_nlp_similarity(w1, w2, model):
    lw1 = ''.join([LANG, '_', w1])
    lw2 = ''.join([LANG, '_', w2])
    sim = get_nlp_similarity(lw1, lw2, model)
    return sim


def get_nlp_similarity(w1, w2, model):
    sim = None
    if w1 and w2:
        if hasattr(model, 'similarity'):
            sim = model.similarity(w1, w2)
        else:
            w1v = model(w1)
            w2v = model(w2)
            sim = w1v.similarity(w2v)
    return sim


def get_coding(df, shared_data, shared_model, save, measures):
    # TODO refactor this whole process to check individual words first and then map them to cue response pairs afterwards
    # var df: a dataframe of distinct cue - response pairs
    # var shared_data: dict of shared memories for measures, and other data
    # var shared_model: dict of shared memories for models to be used in similarity calculations
    # var save: whether to save the result to the database
    # This algorithm is for coding DISTINCT cue response pairs
    logging.debug("in job pid {}...".format(os.getpid()))

    # get the various shared dataframes from shared memoryq
    shm_data = SharedManager(from_shared=True, shares=shared_data)
    all_measures = shm_data.data["all_measures"]
    all_measures["part_of_speech"] = all_measures["part_of_speech"].str.decode("utf8")
    all_measures["word"] = all_measures["word"].str.decode("utf8")
    all_measures.set_index("word", inplace=True)
    all_words = shm_data.data["all_words"]
    all_words.set_index("word", inplace=True)

    # store the len for checks
    df_len = len(df)

    # Splits are needed for many of the functions below
    df["response_split"] = (
        df.index.get_level_values("response_clean").str.decode("utf8").str.split("_")
    )
    df["cue_split"] = (
        df.index.get_level_values("cue_clean").str.decode("utf8").str.split("_")
    )

    df = basic_measures(df)

    assert len(df) == df_len

    # attach the measures from all measures to the cues/responses and then subtract one from the other to get the diff
    if measures:
        measures = attach_meaures(df, all_measures)
        # various word measures related to the response and cue that have been subtracted from each other to give distances
        assert len(measures) == len(df)
        df.sort_index(inplace=True)
        measures.sort_index(inplace=True)
        assert df.index.equals(measures.index)
        df = pd.concat([df, measures], axis=1)
        df = df.loc[:, ~df.columns.duplicated()]
        df.reset_index(inplace=True)
        assert len(df) == df_len

        # check if the part of speech changes from cue to response, based on pos column and s2vsense
        df = check_senses_and_pos(df)
        assert len(df) == df_len

    ####################################################################################################################
    ########################### NOTE - Every measure before here must apply to BOTH cue and response ###################
    ####################################################################################################################

    # calculate similarities using word embedding models
    df, similarity_cols = calculate_similarities(df, shared_model)

    # based on the average number of ideas (Adj, Adv, Verb, Prep) per 10 words from the last 10 sentences
    # here an idea is given the inverse mean score from the models - so the less related two words are, the more likely
    # they are to be original ideas
    if len(similarity_cols):
        df['idea_density'] = df.apply(
            lambda x: idea_density(x["cue_part_of_speech"], x["response_part_of_speech"], x[similarity_cols]),
            axis=1)

    # get the angle between the ngrams vectors for the cue and response
    gc.collect()
    try:
        df.set_index(['cue_clean', 'response_clean'], inplace=True)
    except KeyError:
        pass

    df = attach_ngrams(df)
    df.reset_index(inplace=True)

    # get rid of unnecessary columns
    df = drop_before_saving(df)

    # round the data to default DPs
    df = df.round(DECIMAL_PLACES)

    # save data to the db
    df = df.rename({"cue_raw": "cue", "response_raw": "response"}, axis=1)

    if save:
        df_with_nones = df.replace({np.nan: None})
        save_coded_pairs(df_with_nones)

    logging.debug("Job with PID {} reached end".format(os.getpid()))
    shm_data.close_shared_memories()

    return df


def calculate_similarities(df, shared_model):
    shm_model = None
    similarity_cols = []
    # get one model at a time to save on RAM
    for model_name, share in shared_model.items():
        logging.debug('model {}'.format(model_name))
        shm_model = SharedModelManager(from_shared=True, shares={model_name: share})
        cue_col = "cue_decoded"
        response_col = "response_decoded"
        model_col = "{}_similarity".format(model_name)
        similarity_cols.append(model_col)

        # model similarities
        model = shm_model.data[model_name]

        # we need the best senses from s2v for later on
        if hasattr(model, 'get_best_sense'):
            cue_col = "cue_best_sense"
            response_col = "response_best_sense"
            df[cue_col] = df["cue_decoded"].apply(model.get_best_sense)
            df[response_col] = df["response_decoded"].apply(model.get_best_sense)

        # some models require the language to be prepended to the word e.g. en_dog, en_god
        # TODO move these calculations to the GPU and put them in a pipeline (spacy?)
        if model_name in LANG_MODELS:
            df[model_col] = df.apply(
                lambda row: get_lang_nlp_similarity(
                    row[cue_col], row[response_col], shm_model.data[model_name]
                ),
                axis=1,
            )
        else:
            df[model_col] = df.apply(
                lambda row: get_nlp_similarity(
                    row[cue_col], row[response_col], shm_model.data[model_name]
                ),
                axis=1,
            )
        shm_model.close_shared_memories()
        gc.collect()
    if shm_model is not None:
        shm_model.close_shared_memories()
    return df, similarity_cols


def drop_before_saving(df):
    # get rid of unnecessary columns
    drop_before_saving = []
    for x in ["cue", "response"]:
        for col in DROP_BEFORE_SAVING:
            col_name = col.format(x)
            if col_name in df.columns:
                drop_before_saving.append(col_name)
    df = df.drop(drop_before_saving, axis=1)
    if 'index' in df.columns:
        df.drop('index', inplace=True, axis=1)
    return df


def basic_measures(df):
    for prefix in ['cue', 'response']:
        # rename cue_clean to clean, etc.
        df = prefixes(df, prefix, add=False)
        df["length"] = df["decoded"].str.len()
        df["syllables"] = np.vectorize(syllables)(df["raw"])
        df["capitalised"] = df["raw"].str[0].str.isupper()
        df["capitalised_any"] = df.apply(lambda row: capitalised(row['raw']), axis=1)
        df["number_of_words"] = df["decoded"].str.split("_").str.len()
        df["chars"] = df["decoded"].apply(lambda x: ''.join(sorted(set(x))))
        # TODO move this somewhere else
        # df["hamming"] = df.apply(lambda row: jf.hamming_distance(row['cue_decoded'], row['response_decoded']), axis=1)
        # TODO add number of phonemes
        # rename the columns again
        df = prefixes(df, prefix, add=True)
    return df


def capitalised(cue):
    return sum([w.isupper() * 1 for w in cue])


def first_char_equal(cue, response):
    try:
        result = cue[0] == response[0]
    except IndexError:
        result = False
    return result


def idea_density(cue_pos, response_pos, models):
    model_mean = np.mean(models)
    if cue_pos == 'noun' and response_pos in ('adj', 'verb', 'adv', 'preposition'):
        # LOG(1 / C2) * B2 + LOG(1 / (1 + C2))
        # 0.01 just to make sure there are no div/0
        return np.log(1 / (0.01 + model_mean)) + np.log(1 / (1 + model_mean))
    return np.log(1 / (1 + model_mean))


def attach_meaures(df, all_measures):
    # trim down the cues and responses so we're only running functions once per word
    distinct_responses = pd.DataFrame(
        df.index.get_level_values("response_clean").str.decode("utf8").drop_duplicates()
    )
    distinct_cues = pd.DataFrame(
        df.index.get_level_values("cue_clean").str.decode("utf8").drop_duplicates()
    )

    response_measures = get_measures(all_measures, distinct_responses)
    cue_measures = get_measures(all_measures, distinct_cues)
    # attach the various measures back to the full lists of cues and responses
    response_measures_full = pd.DataFrame(
        df.index.get_level_values("response_clean").str.decode("utf8")
    ).merge(
        response_measures,
        left_on="response_clean",
        right_on="response_clean",
        how="left",
    )

    cue_measures_full = pd.DataFrame(
        df.index.get_level_values("cue_clean").str.decode("utf8")
    ).merge(cue_measures, left_on="cue_clean", right_on="cue_clean", how="left")

    assert len(cue_measures_full) == len(response_measures_full)

    # make sure there are no overlapping column names
    cue_measures_full = prefixes(cue_measures_full, 'cue', add=True)
    response_measures_full = prefixes(response_measures_full, 'response', add=True)
    measures_full = pd.concat([cue_measures_full.reset_index(), response_measures_full.reset_index()], axis=1)

    cue_numeric = cue_measures_full.drop(
        ["cue_clean", "cue_part_of_speech", "cue_lstem", "cue_sstem"], axis=1
    )
    res_numeric = response_measures_full.drop(
        ["response_clean", "response_part_of_speech", "response_lstem", "response_sstem"], axis=1
    )

    # fill nans with means
    cue_numeric = cue_numeric.fillna(cue_numeric.mean())
    res_numeric = res_numeric.fillna(res_numeric.mean())

    # normalize the vectors before calculating similarities
    cue_gpu = cp.array(cue_numeric, dtype=float)
    res_gpu = cp.array(res_numeric, dtype=float)
    cue_gpu_norm = cue_gpu / cp.linalg.norm(cue_gpu, axis=0, keepdims=True)
    res_gpu_norm = res_gpu / cp.linalg.norm(res_gpu, axis=0, keepdims=True)

    # similarity metrics
    measures_full["cosine"] = cp.asnumpy(rowwise_cosine_angle(cp, cue_gpu_norm, res_gpu_norm))
    measures_full["manhattan"] = cp.asnumpy(manhattan(cp, cue_gpu_norm, res_gpu_norm))
    measures_full["chebyshev"] = cp.asnumpy(chebyshev(cp, cue_gpu_norm, res_gpu_norm))
    measures_full["cue_clean"] = measures_full["cue_clean"].str.encode("utf8")
    measures_full["response_clean"] = measures_full["response_clean"].str.encode("utf8")

    measures_full.set_index(["cue_clean", "response_clean"], inplace=True)
    return measures_full


def prefixes(df, suffix, add=True):
    start = '{}_'.format(suffix)
    if add:
        cols = {k: '{}{}'.format(start, k) for k in df if not re.match('(cue|response)_', k)}
    else:
        cols = {k: k.replace(start, '') for k in df if re.match('(cue|response)_', k)}
    df = df.rename(columns=cols)
    return df


def rowwise_cosine_angle(xp, a, b):
    divisors = xp.empty(shape=(a.shape[0], 1), dtype=float)
    for i in range(a.shape[0]):
        divisors[i] = xp.linalg.norm(a[i:i + 1]) * xp.linalg.norm(b[i:i + 1])
    return xp.rad2deg(xp.arccos(xp.divide(xp.einsum('ij,ij->i', a, b), divisors.T))).T


# should these be calculated after normalization?
def manhattan(xp, a, b):
    return xp.abs(a - b).sum(axis=1)


def chebyshev(xp, a, b):
    return xp.abs(a - b).max(axis=1)


def nanindex(list_or_string, word):
    try:
        return list_or_string.index(word)
    except ValueError:
        return np.nan


def get_measures(all_measures, distinct_words):
    col_name = distinct_words.columns[0]
    all_measures_words = all_measures.index.get_level_values(level="word")
    words_intersect = np.intersect1d(distinct_words, all_measures_words)
    words_measures = all_measures[all_measures.index.isin(words_intersect)]
    words_measures_full = pd.merge(
        pd.DataFrame(distinct_words),
        pd.DataFrame(words_measures),
        left_on=col_name,
        right_on="word",
        how="left",
    )

    multi_words = distinct_words[distinct_words[col_name].str.contains("_")]
    split_words = multi_words[col_name].str.split("_")
    # split is  a list of words [first, second, third] which we want to return [True, True, True] as being in all_measures
    # then we want to get all these measures and take the mean
    # this is done for each split list
    split_words_measures = pd.DataFrame(
        [
            all_measures[all_measures.index.isin(split)].mean().T
            for split in split_words.values
        ],
        columns=all_measures.columns,
    )
    split_words_measures.index = pd.Index(multi_words[col_name])
    words_measures_full.set_index(col_name, inplace=True)
    words_measures_full.update(split_words_measures, overwrite=False)
    words_measures_full.reset_index(inplace=True)
    return words_measures_full


def check_senses_and_pos(df):
    cue_pos = df["cue_part_of_speech"].copy()
    cue_sense = df["cue_sense"].copy()
    response_pos = df["response_part_of_speech"].copy()
    response_sense = df["response_sense"].copy()
    # part of speech is the same for cue and response if... (NaNs are not equal so don't need to worry about these)
    pos_equal = np.logical_or.reduce(
        (
            np.equal(cue_pos, response_pos),
            np.equal(cue_pos, response_sense),
            np.equal(cue_sense, response_pos),
            np.equal(cue_sense, response_sense),
        )
    )
    # the remaining records to be checked
    remaining = df[~pos_equal].copy(deep=True)
    # not nans or not "unknown" str
    cue_pos_not_null = remaining["cue_part_of_speech"].copy() != UNKNOWN_CATEGORY
    cue_sense_not_null = ~remaining["cue_sense"].copy().isnull()
    response_pos_not_null = remaining["response_part_of_speech"].copy() != UNKNOWN_CATEGORY
    response_sense_not_null = ~remaining["response_sense"].copy().isnull()

    # we need at least one of the cue pos/sense and one of the res pos/sense to not be NaN and also not equal
    pos_diff = np.logical_or.reduce(
        (
            np.logical_and(cue_pos_not_null, cue_sense_not_null),
            np.logical_and(cue_pos_not_null, response_sense_not_null),
            np.logical_and(response_pos_not_null, cue_sense_not_null),
            np.logical_and(response_pos_not_null, response_sense_not_null),
        )
    )
    # everything else we cannot tell as the pos/senses are NaN for both cue and/or response cols
    pos_diff_indexes = remaining.index[pos_diff == True].copy()
    pos_equal_indexes = df.index[pos_equal == True].copy()
    # pos_unknown_indexes = remaining[~pos_diff]
    df["pos_changes"] = np.nan
    df.loc[df.index.isin(pos_diff_indexes), "pos_changes"] = True
    df.loc[df.index.isin(pos_equal_indexes), "pos_changes"] = False
    return df


def setup():
    parser = argparse.ArgumentParser(description="Import various types of word data")
    parser.add_argument(
        "--source", default=None, help="code responses from a particular source only"
    )
    parser.add_argument(
        "--word_filter",
        default=False,
        action="store_true",
        help="restrict responses to those that appear in a word filter list",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        default=False,
        help="whether to save the coding, default is False",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        default=False,
        help="whether to rebuild the word measures before coding",
    )
    parser.add_argument(
        "--models",
        default=False,
        action='store_true'
    )
    parser.add_argument(
        "--measures",
        default=False,
        action='store_true'
    )

    args = parser.parse_args()

    models = []
    if args.models:
        models = MODELS

    main(save=args.save, source=args.source, rebuild=args.rebuild, models=models, word_filter=args.word_filter,
         measures=args.measures)


if __name__ == "__main__":
    setup()
