from backend.lib.helpers.dataframe import create_dataframe_in_parallel
from backend.db.db import explode_word_stems
import numpy as np

DATA = [
    {'word_lemma': 'made', 'age': 1, 'part_of_speech': 'verb', 'word': 'made', 'arousal': 0, 'dominance': 0,
     'valence': 0,
     'concreteness_measure': 2.51},
    {'word_lemma': 'up', 'age': 2, 'part_of_speech': 'adv', 'word': 'up', 'arousal': 0, 'dominance': 0,
     'valence': 0,
     'concreteness_measure': 3.83},
    {'word_lemma': 'word', 'age': 3, 'part_of_speech': 'noun', 'word': 'word', 'arousal': 4, 'dominance': 6.22,
     'valence': 5.77,
     'concreteness_measure': 3.56}
]


def test_create_dataframe_in_parallel():
    result = create_dataframe_in_parallel(item_list=explode_word_stems(DATA))
    to_test = ['age', 'arousal', 'concreteness_measure']
    for word_dict in DATA:
        for test_col in to_test:
            print('testing {} {}'.format(word_dict['word'], test_col))
            np.testing.assert_allclose(result.loc[result['word'] == word_dict['word']][test_col].values,
                                       [word_dict[test_col]])


def test_create_dataframe_in_parallel_column_types():
    item_dtype = [
        ('word', '|S'),
        ('age', 'f'),
        ('part_of_speech', 'category'),
        ('arousal', 'f'),
        ('dominance', 'f'),
        ('valence', 'f'),
        ('concreteness_measure', 'f')
    ]

    result = create_dataframe_in_parallel(item_list=explode_word_stems(DATA), columns_types=item_dtype)
    to_test = ['age', 'arousal', 'concreteness_measure']
    for word_dict in DATA:
        for test_col in to_test:
            print('testing {} {}'.format(word_dict['word'], test_col))
            np.testing.assert_allclose(result.loc[result['word'] == word_dict['word']][test_col].values,
                                       [word_dict[test_col]])
