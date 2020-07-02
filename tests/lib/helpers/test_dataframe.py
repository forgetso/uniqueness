from backend.lib.helpers.dataframe import create_all_measures_dataframe, merge_word_records, \
    replace_nulls_with_nearest_non_nulls_by_lemma, replace_nulls_with_nearest_non_nulls_by_similarity, \
    create_dataframe_from_dict

from backend.db.db import format_word_data, WORD_MEASURES_DTYPES
import numpy as np
from decimal import Decimal

DATA = [
    {'word_lemma': 'made', 'age': 1, 'part_of_speech': 'verb', 'word': 'made', 'arousal': 0, 'dominance': 0,
     'valence': 0,
     'concreteness_measure': 2.51,
     'c_word': 'made',
     'a_word': 'made',
     'o_word': 'made',
     'e_word': None,
     'phon_word': None
     },
    {'word_lemma': 'up', 'age': 2, 'part_of_speech': 'adv', 'word': 'up', 'arousal': 0, 'dominance': 0,
     'valence': 0,
     'concreteness_measure': 3.83,
     'c_word': None,
     'a_word': None,
     'o_word': None,
     'e_word': 'up',
     'phon_word': None
     },
    {'word_lemma': 'word', 'age': 3, 'part_of_speech': 'noun', 'word': 'word', 'arousal': 4, 'dominance': 6.22,
     'valence': 5.77,
     'concreteness_measure': 3.56,
     'c_word': 'word',
     'a_word': 'word',
     'o_word': 'word',
     'e_word': None,
     'phon_word': None
     }
]


def test_create_all_measures_dataframe():
    item_dtype = [
        ('word', '|S'),
        ('word_lemma', '|S'),
        ('age', 'f'),
        ('part_of_speech', 'category'),
        ('arousal', 'f'),
        ('dominance', 'f'),
        ('valence', 'f')
    ]
    result = create_all_measures_dataframe(df=format_word_data(DATA, item_dtype=item_dtype))
    to_test = ['age', 'arousal', 'concreteness_measure']
    print(result)
    for word_dict in DATA:
        for test_col in to_test:
            print('testing {} {}'.format(word_dict['word'], test_col))
            print(result.loc[result['word'] == word_dict['word']])
            np.testing.assert_allclose(result.loc[result['word'] == word_dict['word']][test_col].values,
                                       [word_dict[test_col]])


def test_merge_word_records():
    item_dtype = [
        ('word', '|S'),
        ('age', 'f'),
        ('part_of_speech', 'category'),
        ('arousal', 'f'),
        ('dominance', 'f'),
        ('valence', 'f')
    ]
    item_dtype_dict = {k: v for k, v in item_dtype}
    w1 = {'age': 0, 'part_of_speech': 'noun', 'arousal': 2, 'dominance': 6, 'valence': None}
    w2 = {'age': 0, 'part_of_speech': 'verb', 'arousal': 2, 'dominance': 6, 'valence': None}
    w3 = {'age': 3, 'part_of_speech': None, 'arousal': 5, 'dominance': None, 'valence': None}
    merged = merge_word_records([w1, w2, w3], item_dtype_dict=item_dtype_dict)
    assert isinstance(merged, dict)
    assert merged['age'] == 1
    assert merged['arousal'] == 3
    assert merged['dominance'] == 6
    assert set(merged['part_of_speech'].split(',')) == {'noun', 'verb'}
    assert merged['valence'] == None


def test_replace_nulls_with_nearest_non_nulls():
    words = {
        'abandon': {'age': 1, 'part_of_speech': 'noun', 'word_lemma': 'abandon', 'word': 'abandon'},
        'abandonee': {'age': None, 'part_of_speech': None, 'word_lemma': 'abandon', 'word': 'abandonee'},
        'abandoner': {'age': None, 'part_of_speech': None, 'word_lemma': 'abandon', 'word': 'abandoner'},
        'earth': {'age': 1, 'part_of_speech': 'noun', 'word_lemma': 'earth', 'word': 'earth'},
        'earthy': {'age': None, 'part_of_speech': None, 'word_lemma': None, 'word': 'earthy'}
    }
    initialdf = create_dataframe_from_dict(words, WORD_MEASURES_DTYPES)
    result = replace_nulls_with_nearest_non_nulls_by_lemma(initialdf)
    assert result.loc[b'abandonee']['age'] == 1
    assert result.loc[b'abandonee']['part_of_speech'] == 'noun'


def test_replace_nulls_with_nearest_non_nulls_by_similarity():
    # having word in the dict as well as the outer key is a bit redundant but it makes life easier when we are
    # identifying if words are already present or not and then later when we translate to a dataframe
    words = {
        'abandon': {'age': 1, 'part_of_speech': 'noun', 'word_lemma': 'abandon', 'word': 'abandon'},
        'abandonee': {'age': None, 'part_of_speech': None, 'word_lemma': 'abandon', 'word': 'abandonee'},
        'abandoner': {'age': None, 'part_of_speech': None, 'word_lemma': 'abandon', 'word': 'abandoner'},
        'earth': {'age': 1, 'part_of_speech': 'noun', 'word_lemma': 'earth', 'word': 'earth'},
        'earthy': {'age': None, 'part_of_speech': None, 'word_lemma': None, 'word': 'earthy'},
        'car': {'age': 2, 'part_of_speech': 'noun', 'word_lemma': 'car', 'word': 'car'},
        'carpool': {'age': None, 'part_of_speech': None, 'word_lemma': None, 'word': 'carpool'},
    }
    initialdf = create_dataframe_from_dict(words, WORD_MEASURES_DTYPES)
    df = replace_nulls_with_nearest_non_nulls_by_similarity(initialdf)
    assert np.array_equal(df.loc[df['word'] == b'earthy']['age'].values, [1])
    assert np.array_equal(df.loc[df['word'] == b'earthy']['part_of_speech'].values, ['noun'])
    assert np.isnan(df.loc[df['word'] == b'earthy']['word_lemma'].values[0])
    assert np.isnan(df.loc[df['word'] == b'carpool']['word_lemma'].values[0])
    assert np.isnan(df.loc[df['word'] == b'carpool']['age'].values[0])
    assert np.isnan(df.loc[df['word'] == b'carpool']['part_of_speech'].values[0])


def test_format_word_data():
    # TODO replace cur with hard coded data
    # cur = get_all_measures_sql()
    cur = [{'word_lemma': 'a', 'part_of_speech': 'det', 'c_word': 'a', 'a_word': 'a', 'o_word': 'a', 'e_word': None,
            'phon_word': 'a', 'age': Decimal('2.89'), 'arousal': None, 'dominance': None, 'valence': None,
            'bigram_count': None, 'bigram_freq': None, 'frequency': Decimal('69523.7'),
            'ortho_frequency': Decimal('950.59'),
            'ortho_neighbours': 25, 'trigram_count': None, 'trigram_freq': None, 'unconstrained_bigram_count': None,
            'unconstrained_bigram_freq': None, 'unconstrained_trigram_count': None, 'unconstrained_trigram_freq': None,
            'unconstrained_unigram_count': Decimal('41883'), 'unconstrained_unigram_freq': Decimal('381780.33'),
            'unigram_count': Decimal('1'), 'unigram_freq': Decimal('69523.7'), 'pknown': Decimal('0.98'),
            'prevalence': Decimal('1.917'), 'nphon': 1, 'scdcnt': 8382, 'sfreq': Decimal('20415.27'),
            'strbpav': Decimal('0'),
            'strdens': 44, 'strlcpospav': Decimal('0.06666667'), 'strpospav': Decimal('0.00286029'),
            'strtpav': Decimal('0'),
            'unsbpav': Decimal('0'), 'unsdens': 44, 'unslcpospav': Decimal('0.07692308'),
            'unspospav': Decimal('0.00333062'),
            'unstpav': Decimal('0'), 'concreteness_measure': Decimal('1.46')},
           {'word_lemma': 'a', 'part_of_speech': 'det', 'c_word': 'a', 'a_word': 'a', 'o_word': 'a', 'e_word': None,
            'phon_word': 'a', 'age': Decimal('2.89'), 'arousal': None, 'dominance': None, 'valence': None,
            'bigram_count': None, 'bigram_freq': None, 'frequency': Decimal('69523.7'),
            'ortho_frequency': Decimal('950.59'),
            'ortho_neighbours': 25, 'trigram_count': None, 'trigram_freq': None, 'unconstrained_bigram_count': None,
            'unconstrained_bigram_freq': None, 'unconstrained_trigram_count': None, 'unconstrained_trigram_freq': None,
            'unconstrained_unigram_count': Decimal('41883'), 'unconstrained_unigram_freq': Decimal('381780.33'),
            'unigram_count': Decimal('1'), 'unigram_freq': Decimal('69523.7'), 'pknown': Decimal('0.98'),
            'prevalence': Decimal('1.917'), 'nphon': 1, 'scdcnt': 8382, 'sfreq': Decimal('20415.27'),
            'strbpav': Decimal('0'),
            'strdens': 21, 'strlcpospav': Decimal('0.06666667'), 'strpospav': Decimal('0.03588891'),
            'strtpav': Decimal('0'),
            'unsbpav': Decimal('0'), 'unsdens': 26, 'unslcpospav': Decimal('0.07692308'),
            'unspospav': Decimal('0.04449866'),
            'unstpav': Decimal('0'), 'concreteness_measure': Decimal('1.46')},
           {'word_lemma': None, 'part_of_speech': None, 'c_word': 'aardvark', 'a_word': None, 'o_word': None,
            'e_word': None,
            'phon_word': None, 'age': None, 'arousal': None, 'dominance': None, 'valence': None, 'bigram_count': None,
            'bigram_freq': None, 'frequency': None, 'ortho_frequency': None, 'ortho_neighbours': None,
            'trigram_count': None,
            'trigram_freq': None, 'unconstrained_bigram_count': None, 'unconstrained_bigram_freq': None,
            'unconstrained_trigram_count': None, 'unconstrained_trigram_freq': None,
            'unconstrained_unigram_count': None,
            'unconstrained_unigram_freq': None, 'unigram_count': None, 'unigram_freq': None, 'pknown': None,
            'prevalence': None, 'nphon': None, 'scdcnt': None, 'sfreq': None, 'strbpav': None, 'strdens': None,
            'strlcpospav': None, 'strpospav': None, 'strtpav': None, 'unsbpav': None, 'unsdens': None,
            'unslcpospav': None,
            'unspospav': None, 'unstpav': None, 'concreteness_measure': Decimal('4.68')},
           {'word_lemma': None, 'part_of_speech': None, 'c_word': 'aback', 'a_word': None, 'o_word': 'aback',
            'e_word': None,
            'phon_word': 'aback', 'age': None, 'arousal': None, 'dominance': None, 'valence': None,
            'bigram_count': Decimal('28.25'), 'bigram_freq': Decimal('1312.2'), 'frequency': Decimal('3.39'),
            'ortho_frequency': Decimal('0'), 'ortho_neighbours': 2, 'trigram_count': Decimal('7.67'),
            'trigram_freq': Decimal('151.94'), 'unconstrained_bigram_count': Decimal('1807.25'),
            'unconstrained_bigram_freq': Decimal('8228.52'), 'unconstrained_trigram_count': Decimal('279.33'),
            'unconstrained_trigram_freq': Decimal('1515.22'), 'unconstrained_unigram_count': Decimal('24413.2'),
            'unconstrained_unigram_freq': Decimal('195527.95'), 'unigram_count': Decimal('221.6'),
            'unigram_freq': Decimal('6583.34'), 'pknown': Decimal('0.86'), 'prevalence': Decimal('1.077'), 'nphon': 4,
            'scdcnt': 15, 'sfreq': Decimal('0.29'), 'strbpav': Decimal('0.00249014'), 'strdens': 2,
            'strlcpospav': Decimal('0.01996641'), 'strpospav': Decimal('0.02795435'), 'strtpav': Decimal('0.00008732'),
            'unsbpav': Decimal('0.00307521'), 'unsdens': 2, 'unslcpospav': Decimal('0.02081854'),
            'unspospav': Decimal('0.03124957'), 'unstpav': Decimal('0.00013788'),
            'concreteness_measure': Decimal('1.65')},
           {'word_lemma': 'abacus', 'part_of_speech': 'noun', 'c_word': 'abacus', 'a_word': 'abacus',
            'o_word': 'abacus',
            'e_word': None, 'phon_word': 'abacus', 'age': Decimal('8.69'), 'arousal': None, 'dominance': None,
            'valence': None, 'bigram_count': Decimal('29'), 'bigram_freq': Decimal('163.96'),
            'frequency': Decimal('0.48'),
            'ortho_frequency': Decimal('0'), 'ortho_neighbours': 0, 'trigram_count': Decimal('3.25'),
            'trigram_freq': Decimal('5.62'), 'unconstrained_bigram_count': Decimal('1855.6'),
            'unconstrained_bigram_freq': Decimal('9411.06'), 'unconstrained_trigram_count': Decimal('107.75'),
            'unconstrained_trigram_freq': Decimal('571.67'), 'unconstrained_unigram_count': Decimal('31056.67'),
            'unconstrained_unigram_freq': Decimal('221355.86'), 'unigram_count': Decimal('586.33'),
            'unigram_freq': Decimal('4150.41'), 'pknown': Decimal('0.93'), 'prevalence': Decimal('1.428'), 'nphon': 6,
            'scdcnt': 10, 'sfreq': Decimal('0.24'), 'strbpav': Decimal('0.00465286'), 'strdens': 0,
            'strlcpospav': Decimal('0.05596900'), 'strpospav': Decimal('0.04516777'), 'strtpav': Decimal('0.00012216'),
            'unsbpav': Decimal('0.00533857'), 'unsdens': 0, 'unslcpospav': Decimal('0.05996655'),
            'unspospav': Decimal('0.04967565'), 'unstpav': Decimal('0.00017278'),
            'concreteness_measure': Decimal('4.52')},
           {'word_lemma': 'abandon', 'part_of_speech': 'verb', 'c_word': 'abandon', 'a_word': 'abandon',
            'o_word': 'abandon',
            'e_word': 'abandon', 'phon_word': 'abandon', 'age': Decimal('8.32'), 'arousal': Decimal('3.73'),
            'dominance': Decimal('3.32'), 'valence': Decimal('2.84'), 'bigram_count': Decimal('58.33'),
            'bigram_freq': Decimal('440.72'), 'frequency': Decimal('16.78'), 'ortho_frequency': Decimal('0'),
            'ortho_neighbours': 0, 'trigram_count': Decimal('5'), 'trigram_freq': Decimal('24.14'),
            'unconstrained_bigram_count': Decimal('3223.83'), 'unconstrained_bigram_freq': Decimal('29746.01'),
            'unconstrained_trigram_count': Decimal('256.6'), 'unconstrained_trigram_freq': Decimal('7018.6'),
            'unconstrained_unigram_count': Decimal('32376.29'), 'unconstrained_unigram_freq': Decimal('270405.62'),
            'unigram_count': Decimal('451.14'), 'unigram_freq': Decimal('3326.59'), 'pknown': Decimal('1.00'),
            'prevalence': Decimal('2.427'), 'nphon': 7, 'scdcnt': 312, 'sfreq': Decimal('8.10'),
            'strbpav': Decimal('0.00793518'), 'strdens': 2, 'strlcpospav': Decimal('0.05686242'),
            'strpospav': Decimal('0.04814775'), 'strtpav': Decimal('0.00039479'), 'unsbpav': Decimal('0.00867981'),
            'unsdens': 2, 'unslcpospav': Decimal('0.05836835'), 'unspospav': Decimal('0.05069134'),
            'unstpav': Decimal('0.00044939'), 'concreteness_measure': Decimal('2.54')},
           {'word_lemma': 'abandon', 'part_of_speech': 'verb', 'c_word': 'abandoned', 'a_word': 'abandoned',
            'o_word': 'abandoned', 'e_word': None, 'phon_word': 'abandoned', 'age': Decimal('8.32'), 'arousal': None,
            'dominance': None, 'valence': None, 'bigram_count': Decimal('176.5'), 'bigram_freq': Decimal('732.42'),
            'frequency': Decimal('37.78'), 'ortho_frequency': Decimal('0'), 'ortho_neighbours': 0,
            'trigram_count': Decimal('21.57'), 'trigram_freq': Decimal('126.45'),
            'unconstrained_bigram_count': Decimal('3865.62'), 'unconstrained_bigram_freq': Decimal('30190.19'),
            'unconstrained_trigram_count': Decimal('329.57'), 'unconstrained_trigram_freq': Decimal('6445.29'),
            'unconstrained_unigram_count': Decimal('34587.78'), 'unconstrained_unigram_freq': Decimal('286726.48'),
            'unigram_count': Decimal('764.78'), 'unigram_freq': Decimal('2444.27'), 'pknown': Decimal('1.00'),
            'prevalence': Decimal('2.576'), 'nphon': 8, 'scdcnt': 538, 'sfreq': Decimal('13.29'),
            'strbpav': Decimal('0.00790571'), 'strdens': 2, 'strlcpospav': Decimal('0.06053913'),
            'strpospav': Decimal('0.04881351'), 'strtpav': Decimal('0.00043160'), 'unsbpav': Decimal('0.00850058'),
            'unsdens': 2, 'unslcpospav': Decimal('0.06268797'), 'unspospav': Decimal('0.05103952'),
            'unstpav': Decimal('0.00054783'), 'concreteness_measure': Decimal('2.52')},
           {'word_lemma': None, 'part_of_speech': None, 'c_word': 'abandonee', 'a_word': None, 'o_word': None,
            'e_word': None,
            'phon_word': None, 'age': None, 'arousal': None, 'dominance': None, 'valence': None, 'bigram_count': None,
            'bigram_freq': None, 'frequency': None, 'ortho_frequency': None, 'ortho_neighbours': None,
            'trigram_count': None,
            'trigram_freq': None, 'unconstrained_bigram_count': None, 'unconstrained_bigram_freq': None,
            'unconstrained_trigram_count': None, 'unconstrained_trigram_freq': None,
            'unconstrained_unigram_count': None,
            'unconstrained_unigram_freq': None, 'unigram_count': None, 'unigram_freq': None, 'pknown': None,
            'prevalence': None, 'nphon': None, 'scdcnt': None, 'sfreq': None, 'strbpav': None, 'strdens': None,
            'strlcpospav': None, 'strpospav': None, 'strtpav': None, 'unsbpav': None, 'unsdens': None,
            'unslcpospav': None,
            'unspospav': None, 'unstpav': None, 'concreteness_measure': Decimal('2.92')},
           {'word_lemma': None, 'part_of_speech': None, 'c_word': 'abandoner', 'a_word': None, 'o_word': None,
            'e_word': None,
            'phon_word': None, 'age': None, 'arousal': None, 'dominance': None, 'valence': None, 'bigram_count': None,
            'bigram_freq': None, 'frequency': None, 'ortho_frequency': None, 'ortho_neighbours': None,
            'trigram_count': None,
            'trigram_freq': None, 'unconstrained_bigram_count': None, 'unconstrained_bigram_freq': None,
            'unconstrained_trigram_count': None, 'unconstrained_trigram_freq': None,
            'unconstrained_unigram_count': None,
            'unconstrained_unigram_freq': None, 'unigram_count': None, 'unigram_freq': None, 'pknown': None,
            'prevalence': None, 'nphon': None, 'scdcnt': None, 'sfreq': None, 'strbpav': None, 'strdens': None,
            'strlcpospav': None, 'strpospav': None, 'strtpav': None, 'unsbpav': None, 'unsdens': None,
            'unslcpospav': None,
            'unspospav': None, 'unstpav': None, 'concreteness_measure': Decimal('2.5')},
           {'word_lemma': 'abandonment', 'part_of_speech': 'noun', 'c_word': 'abandonment', 'a_word': 'abandonment',
            'o_word': 'abandonment', 'e_word': 'abandonment', 'phon_word': 'abandonment', 'age': Decimal('10.27'),
            'arousal': Decimal('4.95'), 'dominance': Decimal('2.64'), 'valence': Decimal('2.63'),
            'bigram_count': Decimal('48.7'), 'bigram_freq': Decimal('260.55'), 'frequency': Decimal('5.53'),
            'ortho_frequency': Decimal('0'), 'ortho_neighbours': 0, 'trigram_count': Decimal('19.78'),
            'trigram_freq': Decimal('166.37'), 'unconstrained_bigram_count': Decimal('3300.8'),
            'unconstrained_bigram_freq': Decimal('28030.02'), 'unconstrained_trigram_count': Decimal('502.67'),
            'unconstrained_trigram_freq': Decimal('6320.77'), 'unconstrained_unigram_count': Decimal('34724.18'),
            'unconstrained_unigram_freq': Decimal('290522.85'), 'unigram_count': Decimal('343'),
            'unigram_freq': Decimal('773.13'), 'pknown': Decimal('0.99'), 'prevalence': Decimal('2.185'), 'nphon': 11,
            'scdcnt': 45, 'sfreq': Decimal('0.96'), 'strbpav': Decimal('0.00900070'), 'strdens': 0,
            'strlcpospav': Decimal('0.07732630'), 'strpospav': Decimal('0.06347543'), 'strtpav': Decimal('0.00070974'),
            'unsbpav': Decimal('0.00964252'), 'unsdens': 0, 'unslcpospav': Decimal('0.08282489'),
            'unspospav': Decimal('0.06528369'), 'unstpav': Decimal('0.00074690'),
            'concreteness_measure': Decimal('2.54')}]

    unpacked_data = format_word_data(cur, WORD_MEASURES_DTYPES)
    all_words = ['a', 'aardvark', 'aback', 'abacus', 'abandon', 'abandoned', 'abandonee', 'abandoner', 'abandonment']
    words_in_data = {x for x in unpacked_data['word'].str.decode('utf8').values}
    for word in all_words:
        assert word.lower() in words_in_data
    return
