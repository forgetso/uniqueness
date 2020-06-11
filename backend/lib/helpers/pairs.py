import logging
import unidecode
import numpy as np
from nltk import PorterStemmer
from backend.settings import LOG_LEVEL

ps = PorterStemmer()
logging.basicConfig(level=LOG_LEVEL)


def format_cue_response_pairs(all_pairs, word_filter=None):
    if word_filter is not None:
        intersection = np.intersect1d(np.intersect1d(all_pairs['cue_clean'], all_pairs['response_clean']),
                                      word_filter)
        logging.debug('{} in all pairs'.format(len(all_pairs)))

        if not len(intersection):
            exit('No intersection between allowed words and cue response pairs')
        all_pairs = all_pairs[np.equal(all_pairs['cue_clean'].isin(intersection), True)]
        logging.debug('{} in all pairs after word filter'.format(len(all_pairs)))
    # we take a subset of columns as we're not interested in source, rowid, etc.
    distinct_pairs = all_pairs[['cue_raw',
                                'cue_clean',
                                'cue_correction',
                                'cue_sense',
                                'response_raw',
                                'response_clean',
                                'response_correction',
                                'response_sense']].copy()
    distinct_pairs = distinct_pairs.drop_duplicates()
    cue_clean_utf8 = distinct_pairs['cue_clean'].str.decode('utf8')
    response_clean_utf8 = distinct_pairs['response_clean'].str.decode('utf8')
    distinct_pairs['cue_decoded'] = np.vectorize(unidecode.unidecode)(cue_clean_utf8)
    distinct_pairs['response_decoded'] = np.vectorize(unidecode.unidecode)(response_clean_utf8)
    distinct_pairs['cue_stem'] = cue_clean_utf8.apply(ps.stem)
    distinct_pairs['response_stem'] = response_clean_utf8.apply(ps.stem)
    distinct_pairs = distinct_pairs.set_index(['cue_clean', 'response_clean'])
    logging.debug('{} distinct cue response pairs'.format(len(distinct_pairs)))

    return distinct_pairs
