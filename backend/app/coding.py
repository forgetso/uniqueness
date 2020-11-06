from backend.lib.helpers.string import syllables
from backend.lib.helpers.pairs import CueResponsePairs
from backend.db.layer import save_coded_pairs, delete_coded_pairs
from backend.lib.setup import db_setup
from nltk.stem import PorterStemmer
import re
from backend.lib.multiprocessor import job_pool
from backend.lib.word_filter import get_word_filter
import time
import argparse
from backend.settings import JOB_NUMBER, LOG_LEVEL, DATA_FOLDER, SEMANTIC_SPACE_FOLDER, WORD_FREQUENCIES
import os
import numpy as np
import pandas as pd
import logging
from backend.app.semantic_atlas import check_semantic_atlas
from backend.lib.measures import UNKNOWN_CATEGORY, DECIMAL_PLACES
from backend.lib.sense2vec.sense2vec import Sense2Vec
from backend.lib.sharedmanager import SharedManager
from backend.lib.sharedmodel import SharedModelManager
import gc
from pathlib import Path
from backend.app.ngrams import attach_ngrams
import jellyfish as jf

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
    # "{}_pos",
    "{}_sense",
    "{}_bad_spell",
    "{}_best_sense",
    "{}_stem"
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
    # make sure all of the required columns are in the coding table
    logging.debug("setting up db")
    try:
        db_setup()
    except Exception as e:
        logging.error("Database setup error {}".format(e))
        exit(0)


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
        w1v = model(w1)
        w2v = model(w2)
        sim = w1v.similarity(w2v)
    return sim


def get_coding(df, shared_data, shared_model, save, measures):
    # TODO refactor this whole process to check individual words first and then map them to cue response pairs afterwards
    # var df: a dataframe of distinct cue - response pairs
    # var shared: a list of shared memory objects for reading
    # var save: whether to save the result to the database
    # This algorithm is for coding DISTINCT cue response pairs once they have had their spelling corrected. It considers
    # categories such as prevalence, concreteness, age of acquisition, emotions, orthographic neighbourhoods,
    # phonographic neighbourhoods, active brain region.
    logging.debug("in job pid {}...".format(os.getpid()))

    # get the various shared dataframes from shared memory
    shm_data = SharedManager(from_shared=True, shares=shared_data)
    all_measures = shm_data.data["all_measures"]
    all_measures["part_of_speech"] = all_measures["part_of_speech"].str.decode("utf8")
    all_measures["word"] = all_measures["word"].str.decode("utf8")
    all_measures.set_index("word", inplace=True)
    all_words = shm_data.data["all_words"]
    all_words.set_index("word", inplace=True)

    # store the len for checks
    df_len = len(df)

    # calculate similarities using word embedding models

    df, similarity_cols = calculate_similarities(df, shared_model)

    # Splits are needed for many of the functions below
    df["response_split"] = (
        df.index.get_level_values("response_clean").str.decode("utf8").str.split("_")
    )
    df["cue_split"] = (
        df.index.get_level_values("cue_clean").str.decode("utf8").str.split("_")
    )

    # should exact match be considered?
    df = basic_coding(df)

    # add the semantic brain area data
    df = check_semantic_atlas(df)
    assert len(df) == df_len

    # attach the measures from all measures to the cues/responses and then subtract one from the other to get the diff
    if measures:
        measures_diff = get_measures_diff(df, all_measures)
        # various word measures related to the response and cue that have been subtracted from each other to give distances
        assert len(measures_diff) == len(df)
        df.sort_index(inplace=True)
        measures_diff.sort_index(inplace=True)
        assert df.index.equals(measures_diff.index)
        df = pd.concat([df, measures_diff], axis=1)
        df = df.loc[:, ~df.columns.duplicated()]
        df.reset_index(inplace=True)
        assert len(df) == df_len

        # check if the part of speech changes from cue to response, based on pos column and s2vsense
        df = check_senses_and_pos(df)
        assert len(df) == df_len

    # based on the average number of ideas (Adj, Adv, Verb, Prep) per 10 words from the last 10 sentences
    # here an idea is given the inverse mean score from the models - so the less related two words are, the more likely
    # they are to be original ideas
    if len(similarity_cols):
        df['idea_density'] = df.apply(lambda x: idea_density(x['cue_pos'], x['response_pos'], x[similarity_cols]),
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
        save_coded_pairs(df_with_nones, df_with_nones.columns)

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
    return df


def basic_coding(df):
    df["length"] = df["cue_decoded"].str.len() - df["response_decoded"].str.len()
    df["syllables"] = np.vectorize(syllables)(df["cue_raw"]) - np.vectorize(syllables)(
        df["response_raw"]
    )
    df["capitalised"] = np.equal(
        df["cue_raw"].str[0].str.isupper(), df["response_raw"].str[0].str.isupper()
    )

    # the difference in number of capital letters between cue and response
    df["capitalised_any"] = df.apply(lambda row: capitalised_diff(row['cue_raw'], row['response_raw']), axis=1)

    # TODO this assumes there is no punctuation in the cue... does not work
    df["punctuation"] = df["response_raw"].str.contains(PUNCTUATION_REGEX)

    df["number_of_words"] = (
            df["cue_decoded"].str.split("_").str.len()
            - df["response_decoded"].str.split("_").str.len()
    )
    # TODO add test for character intersection and probably move it to it's own function
    df["cue_chars"] = df["cue_decoded"].apply(set)
    df["response_chars"] = df["response_decoded"].apply(set)
    df["intersection"] = df.apply(
        lambda row: set.intersection(row["cue_chars"], row["response_chars"]), axis=1
    )
    df["intersection"] = df["intersection"].apply(lambda x: list(x))
    df["char_intersect"] = df["intersection"].str.len() / df["cue_decoded"].str.len()
    df["first_char_equal"] = df.apply(lambda row: first_char_equal(row['cue_decoded'], row['response_decoded']), axis=1)
    df["hamming"] = df.apply(lambda row: jf.hamming_distance(row['cue_decoded'], row['response_decoded']), axis=1)
    df.drop(columns=["cue_chars", "response_chars", "intersection"], inplace=True)

    return df


def capitalised_diff(cue, response):
    return sum([w.isupper() * 1 for w in cue]) - sum([w.isupper() * 1 for w in response])


def first_char_equal(cue, response):
    try:
        result = cue[0] == response[0]
    except IndexError:
        result = False
    return result


def idea_density(cue_pos, response_pos, models):
    if cue_pos == 'noun' and response_pos in ('adj', 'verb', 'adv', 'preposition'):
        return float(1) - sum(models) / len(models)
    return 0


def get_measures_diff(df, all_measures):
    # trim down the cues and responses so we're only running functions once per word
    distinct_responses = pd.DataFrame(
        df.index.get_level_values("response_clean").str.decode("utf8").drop_duplicates()
    )
    distinct_cues = pd.DataFrame(
        df.index.get_level_values("cue_clean").str.decode("utf8").drop_duplicates()
    )
    response_measures = check_measures(all_measures, distinct_responses)
    cue_measures = check_measures(all_measures, distinct_cues)
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

    measures_diff = cue_measures_full.drop(
        ["cue_clean", "part_of_speech", "lstem", "sstem"], axis=1
    ).subtract(
        response_measures_full.drop(
            ["response_clean", "part_of_speech", "lstem", "sstem"], axis=1
        )
    )

    # take the absolute diff
    measures_diff = np.abs(measures_diff)

    # TODO something with lstem and sstem
    measures_diff["cue_clean"] = cue_measures_full["cue_clean"].str.encode("utf8")
    measures_diff["response_clean"] = response_measures_full[
        "response_clean"
    ].str.encode("utf8")
    # add the parts of speech back into the measures
    measures_diff["cue_pos"] = cue_measures_full["part_of_speech"]
    measures_diff["response_pos"] = response_measures_full["part_of_speech"]

    measures_diff.set_index(["cue_clean", "response_clean"], inplace=True)
    return measures_diff


def nanindex(list_or_string, word):
    try:
        return list_or_string.index(word)
    except ValueError:
        return np.nan


def check_measures(all_measures, distinct_words):
    col_name = distinct_words.columns[0]
    # TODO use the stem column that is already present in the overall dataframe
    stems = pd.DataFrame(
        [[response, ps.stem(response)] for response in distinct_words[col_name]],
        columns=[col_name, "stem"],
    )

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
    stem_measures = all_measures[all_measures.index.isin(stems["stem"])]
    stem_measures_full = pd.merge(
        stems, pd.DataFrame(stem_measures), left_on="stem", right_on="word", how="left"
    )
    stem_measures_full.drop("stem", axis=1, inplace=True)
    # replace the nan values in measures with the stem measures
    words_measures_full.update(stem_measures_full, overwrite=False)
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


# def check_corrections(df):
#     # rules for handling spelling corrections
#     # - check if there are spelling corrections for cue/response
#     # - check if the sense, as defined by sense2vec, is numeric or a named_entity
#     # - only use the spelling correction if the sense is not a numeric or named_entity
#     # - if both cue and response are incorrect/correct, return False
#     # - if one of cue/response is incorrect and one is correct, return True
#     unfiltered_senses = np.array([x.lower() for x in NUM_TYPES + NAMED_ENTITITES])
#     for x in ["cue", "response"]:
#         sensecol = "{}_sense".format(x)
#         decodedcol = "{}_decoded".format(x)
#         corr_col = "{}_correction".format(x)
#         spellcol = "{}_bad_spell".format(x)
#         indexcol = "{}_clean".format(x)
#         df[spellcol] = np.logical_and.reduce(
#             (
#                 ~df[sensecol].isin(unfiltered_senses),
#                 df[corr_col].notnull(),
#                 df[corr_col] != df[decodedcol],
#             )
#         )
#
#         rename_index = np.where(df[spellcol] == True)
#         rename_dict = dict(
#             zip(
#                 df.index.get_level_values(indexcol)[rename_index].values,
#                 df[corr_col].str.encode("utf8").values[rename_index],
#             )
#         )
#         if rename_dict:
#             df.rename(rename_dict, inplace=True, axis=0)
#     df["bad_spell"] = ~np.logical_or(
#         np.logical_and(df["cue_bad_spell"], df["response_bad_spell"]),
#         np.logical_and(~df["cue_bad_spell"], ~df["response_bad_spell"]),
#     )
#     return df


def check_senses_and_pos(df):
    cue_pos = df["cue_pos"].copy()
    cue_sense = df["cue_sense"].copy()
    response_pos = df["response_pos"].copy()
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
    cue_pos_not_null = remaining["cue_pos"].copy() != UNKNOWN_CATEGORY
    cue_sense_not_null = ~remaining["cue_sense"].copy().isnull()
    response_pos_not_null = remaining["response_pos"].copy() != UNKNOWN_CATEGORY
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
