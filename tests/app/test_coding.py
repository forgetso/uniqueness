from backend.app.coding import check_syntagmatic_pd, check_phonology, check_affix_pd, relation_coding, basic_coding
import pandas as pd
import numpy as np


def test_check_syntagmatic():
    data = [
        [b'back', b'and_forth', 1],
        [b'x', b'x_y', 1],
        [b'x', b'y_x', 1],
        [b'cats', b'its_raining_cats_and_dogs', 1],
        [b'fear', b'less', 1],
        [b'nut', b'pea', 1],
        [b'plus', b'or_minus', 1],
        [b'anywhere', b'but_here', 1],
        [b'random', b'not_positionally_related', 1]
    ]
    df = pd.DataFrame(data, columns=['cue_clean', 'response_clean', 'dummy_col'])
    df.set_index(['cue_clean', 'response_clean'], inplace=True)
    result = check_syntagmatic_pd(df)
    expected = pd.DataFrame(['consecutivexy_cue_not_entered',
                             'consecutivexy',
                             'consecutiveyx',
                             'othercollocational',
                             'consecutivexy',
                             'consecutiveyx',
                             'consecutivexy_cue_not_entered',
                             'consecutivexy_cue_not_entered',
                             False
                             ], columns=['syntagmatic'])

    # None is not equal to None so we use False here instead of None
    # also np.nan is not equal to np.nan which is why we didn't use it
    result.fillna(value=False, inplace=True)
    result.reset_index(inplace=True)
    comparison = np.equal(result['syntagmatic'], expected['syntagmatic'])
    assert (np.all(comparison))


def test_check_phonology():
    cue = 'elastic'
    response = 'plastic'
    type = check_phonology(cue, response)
    assert (type == 'soundex')

    cue = 'goon'
    response = 'spoon'
    type = check_phonology(cue, response)
    assert (type == 'soundex')

    cue = 'bored'
    response = 'bore'
    type = check_phonology(cue, response)
    assert (type == 'levenshtein')

# def test_check_affix():
#     cue = 'respondent'
#     response = 'responded'
#     affix = check_affix_pd(cue, response)
#     assert (affix)
#
#     cue = 'light'
#     response = 'lighter'
#     affix = check_affix_pd(cue, response)
#     assert (affix)
#
#
# def test_relation_coding():
#     cue = 'poor'
#     response = 'rich'
#     relations, totals = relation_coding(cue, response)
#     assert (relations['antonyms'])
#     assert (totals['antonyms'] == 1)
#
#
# def test_basic_coding():
#     cue = "Blahblahblah'"
#     response = 'Chat_and_business'
#     basic = basic_coding(cue, response, response_raw=response, response_split=response.split('_'))
#     assert (basic['length'] == 17)
#     assert (basic['syllables'] == 5)
#     assert (basic['syllables_diff'] == 2)
#     assert (basic['capitalised'])
#     assert (basic['punctuation'])
#     assert (basic['number_of_words'] == 3)
