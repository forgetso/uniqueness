from backend.app.coding import check_syntagmatic, check_phonology, check_affix, basic_coding, check_senses_and_pos
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
    result.fillna(value=False, inplace=True)
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
    res = [np.nan, np.nan, np.nan, True, True, True, False, False, True, False, False, False, True, False, False]
    res = pd.DataFrame(res, columns=['pos_changes']).fillna(0)
    df = check_senses_and_pos(df)
    df = df.fillna(0)

    # np.nan != np.nan so need to fill these with a value that will be equal
    assert (np.array_equal(df['pos_changes'], res['pos_changes']))
