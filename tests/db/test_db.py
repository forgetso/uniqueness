from backend.db.db import get_all_measures, get_orthographic, get_all_measures_sql, explode_word_stems


# TODO these are essentially integration tests as they rely on data from the db - create a mock DB layer

def test_get_all_measures():
    am = get_all_measures()
    print(am.loc['true'])
    print(am[am.index.duplicated()])
    assert (am.index.duplicated().any() == False)
    assert (round(am.loc['alcoholism']['unigram_count'], 1) == 549.8)
    assert (round(am.loc['made']['concreteness_measure'], 2) == 2.51)
    assert (round(am.loc['up']['concreteness_measure'], 2) == 3.83)
    assert (round(am.loc['word']['concreteness_measure'], 2) == 3.56)


def test_all_measures_contains_all_words():
    all_words = get_orthographic()
    am = get_all_measures()
    for word in all_words:
        assert (word.lower() in am.index.get_level_values('word'))


def test_explode_word_stems():
    cur = get_all_measures_sql()
    unpacked_data = explode_word_stems(cur)
    all_words = get_orthographic()
    words_in_data = {x['word'] for x in unpacked_data}
    for word in all_words:
        assert word.lower() in words_in_data
    return
