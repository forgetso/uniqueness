import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG)


def format_cue_response_pairs(all_pairs, word_filter=None):
    all_pairs['response'] = [x.replace(' ', '_').replace('-', '_').lower() for x in all_pairs['response']]
    if word_filter is not None:
        intersection = np.intersect1d(np.intersect1d(all_pairs['cue_clean'], all_pairs['response_clean']),
                                      word_filter)
        logging.debug('{} in all pairs'.format(len(all_pairs)))

        if not len(intersection):
            exit(0)
        all_pairs = all_pairs[all_pairs['cue_clean'].isin(intersection) == True]
        logging.debug('{} in all pairs after word filter'.format(len(all_pairs)))

    distinct_cue_response_pairs = all_pairs[['cue', 'response', 'response_raw', 'cue_clean', 'response_clean']].copy()
    distinct_cue_response_pairs = distinct_cue_response_pairs.set_index(['cue_clean', 'response_clean'])
    distinct_cue_response_pairs = distinct_cue_response_pairs.drop_duplicates()
    logging.debug('{} distinct cue response pairs'.format(len(distinct_cue_response_pairs)))
    all_pairs = all_pairs.set_index(['cue_clean', 'response_clean'])

    return all_pairs, distinct_cue_response_pairs
