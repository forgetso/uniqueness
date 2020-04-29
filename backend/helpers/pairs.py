import numpy as np


def format_cue_response_pairs(all_pairs, word_filter=None):
    cue_response_pairs = []
    distinct_cue_response_pairs = set()

    for pair in all_pairs:
        if word_filter is not None:
            intersection = np.intersect1d(np.intersect1d(all_pairs['cue_clean'], all_pairs['response_clean']),
                                          word_filter)
            print(len(intersection))
            if not len(intersection):
                continue

        pair = dict(pair)
        pair['response'] = pair['response_raw'].replace(' ', '_').replace('-', '_').lower()
        pair['key'] = (pair['cue_clean'], pair['response_clean'])
        distinct_cue_response_pairs.add((pair['key'], pair['cue'], pair['response'], pair['response_raw']))
        cue_response_pairs.append(pair)

    return cue_response_pairs, distinct_cue_response_pairs
