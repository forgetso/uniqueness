import logging
import unidecode
import numpy as np
from nltk import PorterStemmer

ps = PorterStemmer()
logging.basicConfig(level=logging.DEBUG)


def format_cue_response_pairs(all_pairs, word_filter=None):
    all_pairs['response'] = [x.replace(' ', '_').replace('-', '_').lower() for x in all_pairs['response']]

    if word_filter is not None:
        intersection = np.intersect1d(np.intersect1d(all_pairs['cue_clean'], all_pairs['response_clean']),
                                      word_filter)
        logging.debug('{} in all pairs'.format(len(all_pairs)))

        if not len(intersection):
            exit('No intersection between allowed words and cue response pairs')
        all_pairs = all_pairs[np.equal(all_pairs['cue_clean'].isin(intersection), True)]
        logging.debug('{} in all pairs after word filter'.format(len(all_pairs)))

    distinct_pairs = all_pairs[['cue', 'response', 'response_raw', 'cue_clean', 'response_clean']].copy()
    distinct_pairs = distinct_pairs.set_index(['cue_clean', 'response_clean'])
    distinct_pairs = distinct_pairs.drop_duplicates()
    distinct_pairs['cue_decoded'] = np.vectorize(unidecode.unidecode)(distinct_pairs['cue'])
    distinct_pairs['response_decoded'] = np.vectorize(unidecode.unidecode)(distinct_pairs['response'])
    distinct_pairs['cue_stem'] = distinct_pairs['cue'].apply(ps.stem)
    distinct_pairs['response_stem'] = distinct_pairs['response'].apply(ps.stem)
    logging.debug('{} distinct cue response pairs'.format(len(distinct_pairs)))
    all_pairs = all_pairs.set_index(['cue_clean', 'response_clean'])

    return all_pairs, distinct_pairs
