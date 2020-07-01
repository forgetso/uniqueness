from backend.app.coding import check_syntagmatic, check_phonology, check_affix, basic_coding, check_senses_and_pos, \
    check_corrections, get_all_measures, check_measures
from backend.lib.helpers.dataframe import UNKNOWN_CATEGORY

import pandas as pd
import numpy as np
from nltk import PorterStemmer

DATA = [
    [b'back', b'and_forth', 'Back', 'Forth'],
    [b'x', b'x_y', 'x', 'x_y'],
    [b'x', b'y_x', 'x', 'y_x'],
    [b'cats', b'its_raining_cats_and_dogs', 'cats', 'It\'s raining cats and dogs'],
    [b'fear', b'less', 'fear', 'less'],
    [b'nut', b'pea', 'nut', 'pea'],
    [b'plus', b'or_minus', 'plus', 'or minus'],
    [b'anywhere', b'but_here', 'anywhere', 'but here'],
    [b'random', b'not_positionally_related', 'R\'andom', 'not positionally related']
]


# Tests for the coding of cue response pairs.

def test_basic_coding():
    df = pd.DataFrame(DATA, columns=['cue_clean', 'response_clean', 'cue_raw', 'response_raw'])
    cue_clean_utf8 = df['cue_clean'].str.decode('utf8')
    response_clean_utf8 = df['response_clean'].str.decode('utf8')
    df['cue_decoded'] = cue_clean_utf8
    df['response_decoded'] = response_clean_utf8
    df = basic_coding(df)
    assert (np.array_equal(df['capitalised'].values, [True, True, True, False, True, True, True, True, False]))
    assert (df['length'].values[0] == len(DATA[0][0]) - len(DATA[0][1]))
    assert (df['length'].values[3] == len(DATA[3][0]) - len(DATA[3][1]))
    assert (np.array_equal(df['number_of_words'].values, [-1, -1, -1, -4, 0, 0, -1, -1, -2]))
    assert (np.array_equal(df['punctuation'].values, [True, True, True, False, True, True, True, True, False]))


def test_check_measures():
    all_measures = get_all_measures()
    words = pd.DataFrame(['abacus', 'monkey', 'madeupword', 'made_up_word'])
    result = check_measures(all_measures, distinct_words=words)
    print(result)
    np.testing.assert_allclose(result[result[0] == 'made_up_word']['concreteness_measure'].values, [3.303333])


def test_check_syntagmatic():
    df = pd.DataFrame(DATA, columns=['cue_clean', 'response_clean', 'cue_raw', 'response_raw'])
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
    result['syntagmatic'].fillna(value=False, inplace=True)
    result.reset_index(inplace=True)
    comparison = np.equal(result['syntagmatic'], expected['syntagmatic'])
    assert (np.all(comparison))


def test_check_phonology():
    cue = 'elastic'
    response = 'plastic'
    type = check_phonology(cue, response)
    assert (type == 'soundex')

    # cue = 'goon'
    # response = 'spoon'
    # type = check_phonology(cue, response)
    # assert (type == 'soundex')
    #
    # cue = 'bored'
    # response = 'bore'
    # type = check_phonology(cue, response)
    # assert (type == 'levenshtein')


def test_check_affix():
    # It is important to note that the coding is the diff between the cue and response.
    # So if affix returns True for the cue and True for the response, the result will be 0 (True - True).
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


def test_check_corrections():
    # It is important to note that the coding is looking for one of cue and response being correct/incorrect.
    # So if corrections returns True for the cue and True for the response, the result will be False.
    # If corrections returns False for the cue and False for the response, the result will be False.
    data = [
        [b'light', b'lightr', 'light', 'lightr', None, 'lighter', 'org', 'org'],
        [b'respondent', b'respondad', 'respondent', 'respondad', None, 'responded', 'noun', 'noun'],
        [b'athletik', b'athleetes', 'athletik', 'athleetes', 'athletic', 'athletes', 'adj', np.nan],
        [b'carri', b'bread', 'carri', 'bread', 'carry', 'bread', 'adj', 'noun'],
    ]
    df = pd.DataFrame(data, columns=['cue_clean',
                                     'response_clean',
                                     'cue_decoded',
                                     'response_decoded',
                                     'cue_correction',
                                     'response_correction',
                                     'cue_sense',
                                     'response_sense'])
    df.set_index(['cue_clean', 'response_clean'], inplace=True)
    result = check_corrections(df)
    print(result)
    expected = pd.DataFrame([False, True, False, True], columns=['bad_spell'])
    df.reset_index(inplace=True)
    assert (result is not None)
    assert ('bad_spell' in result.columns)
    assert (result['bad_spell'].equals(expected['bad_spell']))


def test_check_senses_and_pos():
    # cue_pos, cue_sense, response_pos, response_sense
    # to generate the below np.array(list(itertools.combinations_with_replacement(x, r=4)))

    data = [[np.nan, np.nan, np.nan, np.nan],  # nan
            [np.nan, np.nan, np.nan, 'noun'],  # nan
            [np.nan, np.nan, np.nan, 'verb'],  # nan
            [np.nan, np.nan, 'noun', 'noun'],  # True
            [np.nan, np.nan, 'noun', 'verb'],  # True
            [np.nan, np.nan, 'verb', 'verb'],  # True
            [np.nan, 'noun', 'noun', 'noun'],  # False
            [np.nan, 'noun', 'noun', 'verb'],  # False
            [np.nan, 'noun', 'verb', 'verb'],  # True
            [np.nan, 'verb', 'verb', 'verb'],  # False
            ['noun', 'noun', 'noun', 'noun'],  # False
            ['noun', 'noun', 'noun', 'verb'],  # False
            ['noun', 'noun', 'verb', 'verb'],  # True
            ['noun', 'verb', 'verb', 'verb'],  # False
            ['verb', 'verb', 'verb', 'verb']]  # False
    df = pd.DataFrame(data, columns=['cue_pos', 'cue_sense', 'response_pos', 'response_sense'])
    # these were originally nas but have since changed to the string, 'unknown'
    df['cue_pos'].fillna(UNKNOWN_CATEGORY, inplace=True)
    df['response_pos'].fillna(UNKNOWN_CATEGORY, inplace=True)
    res = [np.nan, np.nan, np.nan, True, True, True, False, False, True, False, False, False, True, False, False]
    res = pd.DataFrame(res, columns=['pos_changes']).fillna(0)
    df = check_senses_and_pos(df)
    df = df.fillna(0)
    print(res['pos_changes'])
    print(df['pos_changes'])
    # np.nan != np.nan so need to fill these with a value that will be equal
    assert (np.array_equal(df['pos_changes'], res['pos_changes']))
