from backend.lib.helpers.dataframe import create_dataframe_in_parallel
from backend.db.db import explode_word_stems
import numpy as np

DATA = [
    {'word_lemma': 'made', 'age': 1, 'part_of_speech': 'verb', 'word': 'made', 'arousal': 0, 'dominance': 0,
     'valence': 0,
     'concreteness_measure': 2.52},
    {'word_lemma': 'up', 'age': 2, 'part_of_speech': 'adv', 'word': 'up', 'arousal': 0, 'dominance': 0,
     'valence': 0,
     'concreteness_measure': 3.83},
    {'word_lemma': 'word', 'age': 3, 'part_of_speech': 'noun', 'word': 'word', 'arousal': 4, 'dominance': 6.22,
     'valence': 5.77,
     'concreteness_measure': 3.56}
]


def test_create_dataframe_in_parallel():
    item_dtype = [
        ('word', '|S'),
        ('age', 'f'),
        ('part_of_speech', 'category'),
        ('arousal', 'f'),
        ('dominance', 'f'),
        ('valence', 'f'),
        ('concreteness_measure', 'f')
    ]
    result = create_dataframe_in_parallel(item_list=explode_word_stems(DATA))

    for word_dict in DATA:
        np.testing.assert_array_equal(result.loc[result['word'] == word_dict['word']]['age'].values,
                                      [float(word_dict['age'])])

    result = create_dataframe_in_parallel(item_list=explode_word_stems(DATA), columns_types=item_dtype)
    for word_dict in DATA:
        np.testing.assert_array_equal(result.loc[result['word'] == word_dict['word']]['age'].values,
                                      [float(word_dict['age'])])
