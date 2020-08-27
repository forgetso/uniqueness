from backend.lib.multiprocessor import chunks, amend_groups, create_shared_block, get_shared_memories
from tests.lib.test_measures import DATA
from backend.settings import JOB_NUMBER
import numpy as np
import pandas as pd
from multiprocessing import shared_memory


def test_chunks_with_dummy_data():
    total = len(DATA)
    chunk_size = total / (JOB_NUMBER * 2)
    result = chunks(DATA, chunk_size)
    assert len(result) == 1


def test_chunks():
    result = chunks(["a"], 2)
    assert result == [["a"]]

    result = chunks(["a", "b", "c"], 4)
    assert result == [["a"], ["b"], ["c"]]

    result = chunks(["a", "b", "c"], 2)
    assert np.array_equal([x.tolist() for x in result], [["a", "b"], ["c"]])

    result = chunks(["a", "b", "c"], 1)
    assert np.array_equal([x.tolist() for x in result], [["a", "b", "c"]])

    # TODO check for things that should cause errors


def test_amend_groups():
    chunks = [
        pd.DataFrame(
            [
                {"word": "aa"},
                {"word": "ab"},
                {"word": "ac"},
                {"word": "ad"},
                {"word": "ar"},
            ]
        ),
        pd.DataFrame([{"word": "ag"}, {"word": "bg"}, {"word": "bv"}]),
        pd.DataFrame([{"word": "bc"}, {"word": "qw"}]),
    ]
    result = amend_groups(chunks, groupby=["word", 0])
    print(result)
    print([x for x in result[0]["word"].values])
    print([x for x in result[1]["word"].values])
    print([x for x in result[2]["word"].values])
    assert np.all([x[0] == "a" for x in result[0]["word"].values])
    assert np.all([x[0] == "b" for x in result[1]["word"].values])
    assert np.all([x[0] == "q" for x in result[2]["word"].values])

    chunks = [
        pd.DataFrame([{"word": "aa"}, {"word": "ab"}, ]),
        pd.DataFrame([{"word": "ag"}, {"word": "bc"}, ]),
        pd.DataFrame([{"word": "qw"}, {"word": "qx"}, {"word": "qq"}, ]),
    ]
    result = amend_groups(chunks, groupby=["word", 0])
    assert len(result) == 3
    assert np.all([x[0] == "a" for x in result[0]["word"].values])
    assert np.all([x[0] == "b" for x in result[1]["word"].values])
    assert np.all([x[0] == "q" for x in result[2]["word"].values])


def test_chunks_with_amend_groups():
    words = pd.DataFrame(
        [
            {"word": "czechoslovaks"},
            {"word": "czechoslovakian"},
            {"word": "czechoslovakians"},
            {"word": "czechs"},
            {"word": "d"},
            {"word": "dabchick"},
            {"word": "dabchicks"},
            {"word": "dachas"},
            {"word": "qw"},
            {"word": "qx"},
            {"word": "qq"},
        ]
    )
    chnks = chunks(words, 8)
    assert len(chnks) == 8
    result = amend_groups(chnks, groupby=["word", 0])
    found = False
    print(result)
    assert "czechs" == result[0]["word"][3]

    words = pd.DataFrame(
        [
            {"word": "hypothesis"},
            {"word": "hypothesized"},
            {"word": "hypothesizes"},
            {"word": "hypothetical"},
            {"word": "hypothetically"},
            {"word": "hypoxia"},
            {"word": "hypoxic"},
            {"word": "hyrax"},
            {"word": "hyssop"},
            {"word": "hysterectomies"},
            {"word": "hysterectomy"},
            {"word": "hysteria"},
            {"word": "hysteric"},
            {"word": "hysterical"},
            {"word": "hysterically"},
            {"word": "hysterics"},
            {"word": "hz"},
            {"word": "i"},
            {"word": "ia"},
            {"word": "iago"},
            {"word": "iamb"},
            {"word": "iambi"},
            {"word": "iambic"},
            {"word": "iambics"},
            {"word": "iambs"},
            {"word": "iambus"},
            {"word": "iambuses"},
            {"word": "iatrogenic"},
            {"word": "ib"},
            {"word": "iba", "other_random_col": 1},
        ]
    )
    chnks = chunks(words, JOB_NUMBER * 2)
    assert len(chnks) == JOB_NUMBER * 2
    result = amend_groups(chnks, groupby=["word", 0])
    print(result)
    assert "i" in result[1]["word"].values





# def test_get_shared_memories():
#     to_share = np.random.rand(1000, 1000)
#     shm, np_array, dtypes_sorted = create_shared_block(
#         "test", to_share, [(None, "float32")]
#     )
#     shares = [[shm.name, np_array.shape, dtypes_sorted]]
#     print(shares)
#     mems = get_shared_memories(shares)
#     assert ("test" in mems)
#     np.testing.assert_array_almost_equal(mems["test"], to_share)
