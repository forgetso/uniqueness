import logging
import unidecode
import numpy as np
from nltk import PorterStemmer
from backend.settings import LOG_LEVEL
from backend.db.db import get_cue_response_pairs
import pandas as pd

ps = PorterStemmer()
logging.basicConfig(level=LOG_LEVEL)


class CueResponsePairs:
    def __init__(self, source=None, cur=None, word_filter=None):
        self.distinct_pairs = None
        self.word_filter = word_filter
        self.source = source
        if not cur:
            cur = get_cue_response_pairs(source=source)

        self.all_pairs = pd.DataFrame(
            [x for x in cur], columns=[desc[0] for desc in cur.description]
        )
        self.all_pairs["cue_clean"] = self.all_pairs["cue_clean"].str.encode("utf8")
        self.all_pairs["response_clean"] = self.all_pairs["response_clean"].str.encode("utf8")
        self.format_cue_response_pairs()

    def __get__(self, key):
        return self.distinct_pairs[key]

    def format_cue_response_pairs(self):
        all_pairs = self.all_pairs
        word_filter = self.word_filter
        if hasattr(word_filter, 'size'):
            intersection = np.intersect1d(
                np.intersect1d(all_pairs["cue_clean"], all_pairs["response_clean"]),
                word_filter,
            )
            logging.debug("{} in all pairs".format(len(all_pairs)))

            if not len(intersection):
                exit("No intersection between allowed words and cue response pairs")
            # TODO work out whether to restrict responses
            # all_pairs = all_pairs[
            #     np.logical_and(np.equal(all_pairs["cue_clean"].isin(intersection), True),
            #                    np.equal(all_pairs["response_clean"].isin(intersection), True))
            # ]
            all_pairs = all_pairs[np.equal(all_pairs["cue_clean"].isin(intersection), True)]
            logging.debug("{} in all pairs after word filter".format(len(all_pairs)))
        # we take a subset of columns as we're not interested in source, rowid, etc.
        distinct_pairs = all_pairs[
            [
                "cue_raw",
                "cue_clean",
                "cue_correction",
                "cue_sense",
                "response_raw",
                "response_clean",
                "response_correction",
                "response_sense",
            ]
        ].copy()
        distinct_pairs = distinct_pairs.drop_duplicates()
        cue_clean_utf8 = distinct_pairs["cue_clean"].str.decode("utf8")
        response_clean_utf8 = distinct_pairs["response_clean"].str.decode("utf8")
        distinct_pairs["cue_decoded"] = np.vectorize(unidecode.unidecode)(
            cue_clean_utf8
        )
        distinct_pairs["response_decoded"] = np.vectorize(unidecode.unidecode)(
            response_clean_utf8
        )
        distinct_pairs["cue_stem"] = cue_clean_utf8.apply(ps.stem)
        distinct_pairs["response_stem"] = response_clean_utf8.apply(ps.stem)
        distinct_pairs = distinct_pairs.set_index(["cue_clean", "response_clean"])
        logging.debug("{} distinct cue response pairs".format(len(distinct_pairs)))
        self.distinct_pairs = distinct_pairs
