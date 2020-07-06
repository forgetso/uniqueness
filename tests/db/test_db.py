from backend.db.db import get_all_measures, get_orthographic
import pandas as pd


# TODO these are essentially integration tests as they rely on data from the db - create a mock DB layer

def test_get_all_measures():
    am = get_all_measures()
    assert (am.index.duplicated().any() == False)
    assert (round(am.loc['alcoholism']['unigram_count'], 1) == 549.8)
    assert (round(am.loc['made']['concreteness_measure'], 2) == 2.52)
    assert (round(am.loc['up']['concreteness_measure'], 2) == 3.83)
    assert (round(am.loc['word']['concreteness_measure'], 2) == 3.56)
    assert (round(am.loc['absent']['arousal'], 2) == 3.72)
    assert (round(am.loc['empty']['arousal'], 2) == 2.25)

def test_all_measures_contains_all_words():
    # all_words = pd.Series(get_orthographic(), name='word').astype(str)
    all_words = {x for x in get_orthographic()}
    am = get_all_measures()
    returned_words = {x for x in am.index.get_level_values('word').astype(str)}
    for w in all_words:
        if w not in ('i'):
            assert w in returned_words
    # pd.testing.assert_series_equal(all_words, pd.Series(am.index.get_level_values('word').astype(str)))
