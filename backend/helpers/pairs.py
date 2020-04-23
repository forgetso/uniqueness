def format_cue_response_pairs(all_pairs):
    cue_response_pairs = []
    distinct_cue_response_pairs = set()

    for pair in all_pairs:
        pair = dict(pair)
        pair['response'] = pair['response_raw'].replace(' ', '_').lower()
        pair['key'] = (pair['cue_clean'], pair['response_clean'])
        distinct_cue_response_pairs.add((pair['key'], pair['cue'], pair['response'], pair['response_raw']))
        cue_response_pairs.append(pair)

    return cue_response_pairs, distinct_cue_response_pairs