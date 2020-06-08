from backend.db.db import get_all_measures


def test_get_all_measures():
    # TODO this is essentially an integration test as it relies on data from the db - create a mock DB layer
    am = get_all_measures()
    assert (am.index.duplicated().any() == False)
    assert (round(am.loc['alcoholism']['unigram_count'], 1) == 549.8)
