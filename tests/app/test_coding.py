from backend.app.coding import check_syntagmatic, check_phonology, check_affix, relation_coding, basic_coding
import pandas as pd
import numpy as np
from nltk import PorterStemmer


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
    result = check_syntagmatic(df)
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


def test_check_affix():
    ps = PorterStemmer()
    data = [
        [b'light', b'lighter', ps.stem('light'), ps.stem('lighter')],
        [b'respondent', b'responded', ps.stem('respondent'), ps.stem('responded')],
        [b'athletic', b'athletes', ps.stem('athletic'), ps.stem('athletes')],
    ]
    df = pd.DataFrame(data, columns=['cue_clean', 'response_clean', 'cue_stem', 'response_stem'])
    df.set_index(['cue_clean', 'response_clean'], inplace=True)
    df = check_affix(df)
    expected = pd.DataFrame([True, True, True, ], columns=['affix'])
    df.reset_index(inplace=True)
    assert (df['affix'].equals(expected['affix']))

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
