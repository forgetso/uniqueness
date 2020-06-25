from backend.db.db import get_all_measures, get_orthographic


# TODO these are essentially integration tests as they rely on data from the db - create a mock DB layer

def test_get_all_measures():
    am = get_all_measures()
    assert (am.index.duplicated().any() == False)
    assert (round(am.loc['alcoholism']['unigram_count'], 1) == 549.8)
    assert (round(am.loc['made']['concreteness_measure'], 2) == 2.51)
    assert (round(am.loc['up']['concreteness_measure'], 2) == 3.83)
    assert (round(am.loc['word']['concreteness_measure'], 2) == 3.56)


def test_all_measures_contains_all_words():
    all_words = get_orthographic()
    am = get_all_measures()
    print(am)
    for word in all_words:
        assert (word in am.index.get_level_values('word'))


def test_explode_word_stems():
    # TODO
    return
