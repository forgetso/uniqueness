import argparse
from backend.db.db import get_cue_response_pairs, save_corrections, delete_corrections
import pandas as pd
import numpy as np
from backend.lib.helpers.string import check_spelling
from backend.lib.dataframe import get_unpacked_words
from backend.lib.multiprocessor import job_pool
import logging
from backend.settings import JOB_NUMBER
import time


# gets the cue response pairs, pulls all cues and responses into a unique list and then spell checks them
# saves the spell check suggestion when there is only one suggestion otherwise does not save anything
# TODO save the corrections when there are more than one suggestion and then work out which one to use, if any
# TODO properly consider corrections for multi word responses
def main(source=None):
    start = time.perf_counter()
    cur = get_cue_response_pairs(source=source)
    df = pd.DataFrame([x for x in cur], columns=[desc[0] for desc in cur.description])
    delete_corrections(df['rowid'].values.tolist())
    logging.debug('dispatching jobs...')
    jobs = job_pool(df, JOB_NUMBER, get_corrections_multiprocess)
    end = time.perf_counter()
    logging.debug('time taken {}'.format(end - start))
    # take the individual results and sum them
    totals = sum(jobs)
    logging.info('{} corrections saved'.format(totals))
    return


def get_corrections_multiprocess(df):
    # will be adding these in this process
    df.drop(['cue_correction', 'response_correction'], inplace=True, axis=1)
    unpacked_words = get_unpacked_words(df)

    # get the corrections attached first of all and worry about how to apply them later
    unpacked_words['correction'] = np.vectorize(get_corrections)(
        np.empty(unpacked_words['word'].shape),
        unpacked_words['word'])

    df = merge_corrections(df, unpacked_words, 'cue')
    df = merge_corrections(df, unpacked_words, 'response')

    cue_corrections = format_corrections_for_db(df, 'cue')
    response_corrections = format_corrections_for_db(df, 'response')
    corrections = pd.concat([cue_corrections, response_corrections])
    # belt and braces
    corrections.drop_duplicates(inplace=True, ignore_index=True)
    corrections_list = corrections.values.tolist()
    if len(corrections_list):
        save_corrections(corrections_values=corrections_list)
    return len(corrections_list)


def get_corrections(cue, response):
    correction = None
    _incorrect_spelling, corrections_list, corrections_dict = check_spelling(cue, response)
    if len(corrections_list) == 1 and corrections_list[0].term != response.lower():
        # TODO smarter spelling corrections here, use relation stength of suggestions to cue, soundex, etc.
        correction = corrections_list[0].term
    return correction


def merge_corrections(df, unpacked_words, corrections_type):
    df = df.merge(unpacked_words, left_on='{}_clean'.format(corrections_type), right_on='word', how='left')
    df.rename({'correction': '{}_correction'.format(corrections_type)}, axis=1, inplace=True)
    df.drop({'word'}, inplace=True, axis=1)
    return df


def format_corrections_for_db(df, corrections_type):
    correction_col = '{}_correction'.format(corrections_type)
    word_col = '{}_clean'.format(corrections_type)
    corrections = df[~df[correction_col].isnull()]
    corrections = corrections[[word_col, correction_col, 'rowid', ]]
    corrections.rename(
        {word_col: 'original_word',
         correction_col: 'corrections'},
        inplace=True, axis=1)
    corrections['type'] = corrections_type[0]
    return corrections


def setup():
    parser = argparse.ArgumentParser(description='correct incorrectly spelled cues and responses')
    parser.add_argument('--source', default=None, help="code responses from a particular source only")
    args = parser.parse_args()
    if args.source:
        main(source=args.source)
    else:
        main()


if __name__ == "__main__":
    setup()
