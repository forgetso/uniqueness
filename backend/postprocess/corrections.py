import argparse
from backend.db.db import get_cue_response_pairs
import pandas as pd
from backend.lib.helpers.dataframe import unpack_splits
import numpy as np
from backend.lib.helpers.string import check_spelling


def main(source=None):
    cur = get_cue_response_pairs(source=source)
    df = pd.DataFrame([x for x in cur], columns=[desc[0] for desc in cur.description])
    # TODO do this for unique lists of words
    cue_raw = df['cue']
    response_raw = df['response_raw']
    # get the words from the columns as big lists of individual words (instead of multi word cues/responses)
    cues_unpacked = unpack_splits(cue_raw.str.split('_'), length=1)
    responses_unpacked = unpack_splits(response_raw.str.split('_'), length=1)
    unpacked_words = pd.DataFrame(
        np.unique(np.sort(np.append(cues_unpacked[:, 0], responses_unpacked[:, 0].T, axis=0))),
        columns=['word'])
    # get the corrections attached first of all and worry about how to apply them later
    unpacked_words['corrections'] = np.vectorize(get_corrections)(np.empty(unpacked_words['word'].shape),
                                                                  unpacked_words['word'])

    # TODO use the correction only if the sense is not a person, a company, a website, etc.
    final = get_s2v_sense()
    return


def get_corrections(cue, response):
    correction = None
    _incorrect_spelling, corrections_list, corrections_dict = check_spelling(cue, response)

    if len(corrections_list) == 1 and corrections_list[0].term != response.lower():
        # TODO smarter spelling corrections here, use relation stength of suggestions to cue, soundex, etc.
        correction = corrections_list[0].term
    return correction


def get_s2v_sense():
    return


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
