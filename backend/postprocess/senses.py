import argparse
from backend.db.db import get_cue_response_pairs, update_senses
import pandas as pd
from backend.lib.sense2vec.sense2vec import Sense2Vec
from backend.settings import SENSE2VEC_FOLDER, LOG_LEVEL, JOB_NUMBER
import numpy as np
from backend.lib.measures import get_unpacked_words
from backend.lib.multiprocessor import job_pool
import logging

logging.basicConfig(level=LOG_LEVEL)
s2v = Sense2Vec().from_disk(path=SENSE2VEC_FOLDER)


def main(source=None):
    cur = get_cue_response_pairs(source=source)
    df = pd.DataFrame([x for x in cur], columns=[desc[0] for desc in cur.description])
    # will be adding these in this process
    df.drop(["cue_sense", "response_sense"], inplace=True, axis=1)
    logging.debug("dispatching jobs...")
    jobs = job_pool(df, JOB_NUMBER, get_senses)
    s2v.vectors.close_shared_memory()
    s2v.vectors.unlink_shared_memory()
    return


def get_senses(df):
    unpacked_words = get_unpacked_words(df)
    senses = get_s2v_sense(unpacked_words["word"])
    unpacked_words = pd.concat([unpacked_words.reset_index(drop=True), senses], axis=1)
    df = merge_senses(df, unpacked_words, "cue")
    df = merge_senses(df, unpacked_words, "response")
    update_senses(df)


def get_s2v_sense(words):
    senses = pd.DataFrame(np.vectorize(s2v.get_best_sense)(words), columns=["sense"])
    senses["sense"].fillna(value="", inplace=True)
    senses = (
        senses["sense"]
        .str.split("|", expand=True)
        .rename({1: "sense"}, axis=1)
        .reset_index(drop=True)
    )
    senses.drop(0, inplace=True, axis=1)

    return senses


def merge_senses(df, unpacked_words, word_type):
    df = df.merge(
        unpacked_words,
        left_on="{}_clean".format(word_type),
        right_on="word",
        how="left",
    )
    df.rename({"sense": "{}_sense".format(word_type)}, axis=1, inplace=True)
    df.drop({"word"}, inplace=True, axis=1)
    return df


def setup():
    parser = argparse.ArgumentParser(
        description="correct incorrectly spelled cues and responses"
    )
    parser.add_argument(
        "--source", default=None, help="code responses from a particular source only"
    )
    args = parser.parse_args()
    if args.source:
        main(source=args.source)
    else:
        main()


if __name__ == "__main__":
    setup()
