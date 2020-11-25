from backend.app.coding import basic_measures, check_senses_and_pos, get_measures, main, idea_density
from backend.lib.measures import UNKNOWN_CATEGORY, get_all_measures
from decimal import Decimal
import pandas as pd
import numpy as np
from nltk import PorterStemmer
from tests.db.cur import TestCur

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

PAIRS = [
    [1361814, 92089, 'absent', 'absent', None, 'adj', 'empty', 'empty', None, 'adj', 'humanbraincloud'],
    [1359600, 91776, 'hollow', 'hollow', None, 'adj', 'empty', 'empty', None, 'adj', 'humanbraincloud'],
    [1351431, 91721, 'insipid', 'insipid', None, 'adj', 'empty', 'empty', None, 'adj', 'humanbraincloud'],
    [1353030, 92070, 'my mind', 'my_mind', None, 'org', 'empty', 'empty', None, 'adj', 'humanbraincloud'],
    [1111111, 11111, 'artifact', 'artifact', None, 'noun', 'fossil', 'fossil', None, 'noun', 'humanbraincloud'],
    [2222222, 22222, 'defense', 'defense', None, 'noun', 'offense', 'offense', None, 'noun', 'humanbraincloud'],
    [3333333, 33333, 'qatari', 'qatari', None, 'noun', 'qataris', 'qataris', None, 'noun', 'humanbraincloud']
]


# Tests for the coding of cue response pairs.

def test_basic_coding():
    df = pd.DataFrame(DATA, columns=['cue_clean', 'response_clean', 'cue_raw', 'response_raw'])
    cue_clean_utf8 = df['cue_clean'].str.decode('utf8')
    response_clean_utf8 = df['response_clean'].str.decode('utf8')
    df['cue_decoded'] = cue_clean_utf8
    df['response_decoded'] = response_clean_utf8
    df = basic_measures(df)
    assert (np.array_equal(df['capitalised'].values, [True, True, True, False, True, True, True, True, False]))
    assert (df['length'].values[0] == len(DATA[0][0]) - len(DATA[0][1]))
    assert (df['length'].values[3] == len(DATA[3][0]) - len(DATA[3][1]))
    assert (np.array_equal(df['number_of_words'].values, [-1, -1, -1, -4, 0, 0, -1, -1, -2]))
    assert (np.array_equal(df['punctuation'].values, [True, True, True, False, True, True, True, True, False]))


def test_check_measures():
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
            'concreteness_measure': Decimal('2.54')},
           {'word_lemma': 'made', 'part_of_speech': 'noun', 'c_word': 'made', 'a_word': 'made',
            'o_word': 'made', 'e_word': 'made', 'phon_word': 'made', 'age': Decimal('10.27'),
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
            'concreteness_measure': Decimal('2.54')},
           {'word_lemma': 'up', 'part_of_speech': 'noun', 'c_word': 'up', 'a_word': 'up',
            'o_word': 'up', 'e_word': 'up', 'phon_word': 'up', 'age': Decimal('10.27'),
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
            'concreteness_measure': Decimal('2.54')},
           {'word_lemma': 'word', 'part_of_speech': 'noun', 'c_word': 'word', 'a_word': 'word',
            'o_word': 'word', 'e_word': 'word', 'phon_word': 'word', 'age': Decimal('10.27'),
            'arousal': Decimal('4.95'), 'dominance': Decimal('2.64'), 'valence': Decimal('2.63'),
            'bigram_count': Decimal('48.7'), 'bigram_freq': Decimal('260.55'), 'frequency': Decimal('5.53'),
            'ortho_frequency': Decimal('0'), 'ortho_neighbours': 0, 'trigram_count': Decimal('19.78'),
            'trigram_freq': Decimal('166.37'), 'unconstrained_bigram_count': Decimal('3300.8'),
            'unconstrained_bigram_freq': Decimal('28030.02'), 'unconstrained_trigram_count': Decimal('502.67'),
            'unconstrained_trigram_freq': Decimal('6320.77'), 'unconstrained_unigram_count': Decimal('34724.18'),
            'unconstrained_unigram_freq': Decimal('290522.85'), 'unigram_count': Decimal('343'),
            'unigram_freq': Decimal('773.13'), 'pknown': Decimal('0.99'), 'prevalence': Decimal('2.185'), 'nphon': 11,
            'scdcnt': 45, 'sfreq': Decimal('0.96'), 'strbpav': Decimal('0.00900070'), 'strdens': 0,
            'strlcpospav': Decimal('0.07732630'), 'strpospav': Decimal('0.06347543'),
            'strtpav': Decimal('0.00070974'),
            'unsbpav': Decimal('0.00964252'), 'unsdens': 0, 'unslcpospav': Decimal('0.08282489'),
            'unspospav': Decimal('0.06528369'), 'unstpav': Decimal('0.00074690'),
            'concreteness_measure': Decimal('2.54')}

           ]
    all_measures = get_all_measures(cur)
    words = pd.DataFrame(['abacus', 'abandonment', 'madeupword', 'made_up_word'])
    result = get_measures(all_measures, distinct_words=words)
    print(result)
    np.testing.assert_allclose(result[result[0] == 'made_up_word']['concreteness_measure'].values, [2.54])


# def test_check_syntagmatic():
#     df = pd.DataFrame(DATA, columns=['cue_clean', 'response_clean', 'cue_raw', 'response_raw'])
#     df.set_index(['cue_clean', 'response_clean'], inplace=True)
#     result = check_syntagmatic(df)
#     expected = pd.DataFrame(['consecutivexy_cue_not_entered',
#                              'consecutivexy',
#                              'consecutiveyx',
#                              'othercollocational',
#                              'consecutivexy',
#                              'consecutiveyx',
#                              'consecutivexy_cue_not_entered',
#                              'consecutivexy_cue_not_entered',
#                              False
#                              ], columns=['syntagmatic'])
#
#     # None is not equal to None so we use False here instead of None
#     # also np.nan is not equal to np.nan which is why we didn't use it
#     result['syntagmatic'].fillna(value=False, inplace=True)
#     result.reset_index(inplace=True)
#     comparison = np.equal(result['syntagmatic'], expected['syntagmatic'])
#     assert (np.all(comparison))
#
#
# def test_check_phonology():
#     cue = 'elastic'
#     response = 'plastic'
#     type = check_phonology(cue, response)
#     assert (type == 'soundex')
#
#     # cue = 'goon'
#     # response = 'spoon'
#     # type = check_phonology(cue, response)
#     # assert (type == 'soundex')
#     #
#     # cue = 'bored'
#     # response = 'bore'
#     # type = check_phonology(cue, response)
#     # assert (type == 'levenshtein')
#
#
# def test_check_affix():
#     # It is important to note that the coding is the diff between the cue and response.
#     # So if affix returns True for the cue and True for the response, the result will be 0 (True - True).
#     ps = PorterStemmer()
#     data = [
#         [b'light', b'lighter', ps.stem('light'), ps.stem('lighter')],
#         [b'respondent', b'responded', ps.stem('respondent'), ps.stem('responded')],
#         [b'athletic', b'athletes', ps.stem('athletic'), ps.stem('athletes')],
#     ]
#     df = pd.DataFrame(data, columns=['cue_clean', 'response_clean', 'cue_stem', 'response_stem'])
#     df.set_index(['cue_clean', 'response_clean'], inplace=True)
#     df = check_affix(df)
#     expected = pd.DataFrame([True, True, True, ], columns=['affix'])
#     df.reset_index(inplace=True)
#     assert (df['affix'].equals(expected['affix']))
#
#
# def test_check_corrections():
#     # It is important to note that the coding is looking for one of cue and response being correct/incorrect.
#     # So if corrections returns True for the cue and True for the response, the result will be False.
#     # If corrections returns False for the cue and False for the response, the result will be False.
#     data = [
#         [b'light', b'lightr', 'light', 'lightr', None, 'lighter', 'org', 'org'],
#         [b'respondent', b'respondad', 'respondent', 'respondad', None, 'responded', 'noun', 'noun'],
#         [b'athletik', b'athleetes', 'athletik', 'athleetes', 'athletic', 'athletes', 'adj', np.nan],
#         [b'carri', b'bread', 'carri', 'bread', 'carry', 'bread', 'adj', 'noun'],
#
#     ]
#     df = pd.DataFrame(data, columns=['cue_clean',
#                                      'response_clean',
#                                      'cue_decoded',
#                                      'response_decoded',
#                                      'cue_correction',
#                                      'response_correction',
#                                      'cue_sense',
#                                      'response_sense'])
#     df.set_index(['cue_clean', 'response_clean'], inplace=True)
#     result = check_corrections(df)
#     print(result)
#     expected = pd.DataFrame([False, True, False, True], columns=['bad_spell'])
#     df.reset_index(inplace=True)
#     assert (result is not None)
#     assert ('bad_spell' in result.columns)
#     assert (result['bad_spell'].equals(expected['bad_spell']))


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


def test_main():
    # This is an end to end test of the coding process, which involves pulling out all the measures from the database
    # and all of the cue response pairs. The measures are then applied to cue response pairs.
    # TODO use test data for the measures instead of connecting to the actual DB
    descriptors = ['rowid',
                   'user_id',
                   'cue_raw',
                   'cue_clean',
                   'cue_correction',
                   'cue_sense',
                   'response_raw',
                   'response_clean',
                   'response_correction',
                   'response_sense',
                   'source']
    df = main(cur=TestCur(PAIRS, descriptors=[[x] for x in descriptors]), source='humanbraincloud')
    pd.set_option('display.max_columns', df.shape[1] + 1)
    df_filter = np.logical_and(df['cue'] == 'absent', df['response'] == 'empty')
    assert round(df[df_filter].at[0, 'age'], 2) == round(6.5 - 4.79, 2)  # acquisition age
    assert round(df[df_filter].at[0, 'valence'], 2) == round(4.1 - 3.78, 2)  # emotions
    assert round(df[df_filter].at[0, 'arousal'], 2) == round(3.72 - 2.25, 2)
    assert round(df[df_filter].at[0, 'dominance'], 2) == round(4.21 - 3.61, 2)
    assert round(df[df_filter].at[0, 'bigram_freq'], 2) == round(640.93 - 294.5, 2)  # orthographic neighbours
    assert round(df[df_filter].at[0, 'ortho_neighbours'], 2) == round(0 - 1, 2)
    assert round(df[df_filter].at[0, 'trigram_freq'], 2) == round(236.51 - 86.32, 2)
    assert round(df[df_filter].at[0, 'prevalence'], 2) == round(2.173 - 2.576, 2)  # prevalence
    assert round(df[df_filter].at[0, 'concreteness_measure'], 2) == round(2.7 - 3.43, 2)  # concreteness
    # there are multiple results in the phongraphic table for empty so the mean should be populated in the measures table
    assert round(df[df_filter].at[0, 'nphon'], 2) == round(6 - ((5 + 4) / 2), 2)

    # GB spellings previously were not getting measures attached if the measures were listed under the US spellings
    df_filter = np.logical_and(df['cue'] == 'defense', df['response'] == 'offense')
    assert round(df[df_filter].at[0, 'concreteness_measure'], 2) is not None  # concreteness
    df_filter = np.logical_and(df['cue'] == 'artifact', df['response'] == 'fossil')
    assert round(df[df_filter].at[0, 'concreteness_measure'], 2) is not None  # concreteness
    df_filter = np.logical_and(df['cue'] == 'qatari', df['response'] == 'qataris')
    assert np.isnan(df[df_filter].at[0, 'valence'])


def test_idea_density():
    # idea_density(cue_pos, response_pos, models)
    models = ['model1', 'model2']
    model_cols = ['{}_similarity'.format(x) for x in models]
    data = [
        ['verb', 'noun', 0.346, 0.123],
        ['noun', 'verb', 0.346, 0.123],
        ['noun', 'noun', 0.346, 0.123],
        ['noun', 'adj', 0.346, 0.123],

    ]
    cols = ['cue_pos', 'response_pos']
    cols.extend(model_cols)
    df = pd.DataFrame(data, columns=cols)
    df['idea_density'] = df.apply(lambda row: idea_density(row['cue_pos'], row['response_pos'], row[model_cols]),
                                  axis=1)
    assert np.array_equal(df['idea_density'].values, [0, 1 - (0.346 + 0.123) / 2, 0, 1 - (0.346 + 0.123) / 2])
