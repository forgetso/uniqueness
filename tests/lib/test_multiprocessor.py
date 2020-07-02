from backend.lib.multiprocessor import chunks, amend_groups
from tests.lib.helpers.test_dataframe import DATA
from backend.settings import JOB_NUMBER
import numpy as np


def test_chunks_with_dummy_data():
    total = len(DATA)
    chunk_size = total / (JOB_NUMBER * 2)
    result = chunks(DATA, chunk_size)
    assert (len(result) == 1)


def test_chunks():
    result = chunks(['a'], 2)
    assert (result == [['a']])

    result = chunks(['a', 'b', 'c'], 4)
    assert (result == [['a'], ['b'], ['c']])

    result = chunks(['a', 'b', 'c'], 2)
    assert (result == [['a', 'b'], ['c']])

    result = chunks(['a', 'b', 'c'], 1)
    assert (result == [['a', 'b', 'c']])

    # with pytest.raises(TypeError):
    #     chunks(DATA, 'blah')
    #
    # with pytest.raises(TypeError):
    #     chunks(DATA, np.nan)
    #
    # with pytest.raises(TypeError):
    #     chunks(DATA, {})
    #
    # with pytest.raises(TypeError):
    #     chunks(DATA, None)


def test_amend_groups():
    chunks = [
        [
            {'word': 'aa'},
            {'word': 'ab'},
            {'word': 'ac'},
            {'word': 'ad'},
            {'word': 'ar'}
        ],
        [
            {'word': 'ag'},
            {'word': 'bg'},
            {'word': 'bv'}
        ],
        [
            {'word': 'bc'},
            {'word': 'qw'}
        ],
    ]
    result = amend_groups(chunks, groupby=['word', 0])

    print(result)
    assert (np.all([x['word'][0] == 'a' for x in result[0]]))
    assert (np.all([x['word'][0] == 'b' for x in result[1]]))
    assert (np.all([x['word'][0] == 'q' for x in result[2]]))

    chunks = [
        [
            {'word': 'aa'},
            {'word': 'ab'},
        ],
        [
            {'word': 'ag'},
            {'word': 'bc'},
        ],
        [
            {'word': 'qw'},
            {'word': 'qx'},
            {'word': 'qq'},
        ],
    ]
    result = amend_groups(chunks, groupby=['word', 0])
    assert (len(result) == 3)
    assert (np.all([x['word'][0] == 'a' for x in result[0]]))
    assert (np.all([x['word'][0] == 'b' for x in result[1]]))
    assert (np.all([x['word'][0] == 'q' for x in result[2]]))


def test_chunks_with_amend_groups():
    words = [
        {'word': 'czechoslovaks'},
        {'word': 'czechoslovakian'},
        {'word': 'czechoslovakians'},
        {'word': 'czechs'},
        {'word': 'd'},
        {'word': 'dabchick'},
        {'word': 'dabchicks'},
        {'word': 'dachas'},
        {'word': 'qw'},
        {'word': 'qx'},
        {'word': 'qq'},
    ]
    chnks = chunks(words, 8)
    assert (len(chnks) == 8)
    result = amend_groups(chnks, groupby=['word', 0])
    found = False
    print(result)
    for res in result[0]:
        if 'czechs' == res['word']:
            found = True
    assert (found)

    words = [
        {'word': 'hypothesis'},
        {'word': 'hypothesized'},
        {'word': 'hypothesizes'},
        {'word': 'hypothetical'},
        {'word': 'hypothetically'},
        {'word': 'hypoxia'},
        {'word': 'hypoxic'},
        {'word': 'hyrax'},
        {'word': 'hyssop'},
        {'word': 'hysterectomies'},
        {'word': 'hysterectomy'},
        {'word': 'hysteria'},
        {'word': 'hysteric'},
        {'word': 'hysterical'},
        {'word': 'hysterically'},
        {'word': 'hysterics'},
        {'word': 'hz'},
        {'word': 'i'},
        {'word': 'ia'},
        {'word': 'iago'},
        {'word': 'iamb'},
        {'word': 'iambi'},
        {'word': 'iambic'},
        {'word': 'iambics'},
        {'word': 'iambs'},
        {'word': 'iambus'},
        {'word': 'iambuses'},
        {'word': 'iatrogenic'},
        {'word': 'ib'},
        {'word': 'iba'},
    ]
    chnks = chunks(words, JOB_NUMBER * 2)
    assert (len(chnks) == JOB_NUMBER * 2)
    result = amend_groups(chnks, groupby=['word', 0])
    found = False
    print(result)
    for res in result[1]:
        if 'i' == res['word']:
            found = True
    assert (found)
