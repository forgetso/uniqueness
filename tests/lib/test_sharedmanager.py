import numpy as np
from multiprocessing import shared_memory
from backend.lib.sharedmanager import SharedManager
import pandas as pd


def test_make_dtypes_numpy_compatible():
    sm = SharedManager()
    to_share = [[0.345, 0.234546, "blahblah", "cc"], [0.3434, 0.45454, "blah", "c"]]
    df = pd.DataFrame(to_share, columns=["num1", "num2", "str8", "str2"])
    dtypes_cols = sm.make_dtypes_numpy_compatible(df)
    assert dtypes_cols == [
        ("num1", "float64"),
        ("num2", "float64"),
        ("str8", "|S8"),
        ("str2", "|S2"),
    ]


def test_create_shared_block():
    sm = SharedManager()
    # test sharing a numpy array
    to_share = np.random.rand(1000, 2)
    shm, np_array = sm.create_shared_block("test_numpy", to_share)
    from_shared = shared_memory.SharedMemory(name=shm.name)
    shared_arr = np.ndarray(shape=np_array.shape, dtype="float32", buffer=shm.buf)
    assert isinstance(from_shared, shared_memory.SharedMemory)
    np.testing.assert_array_almost_equal(to_share, shared_arr)

    # test sharing a pandas dataframe
    # strings are converted to bytes so just pass in bytes to begin with
    to_share = [(0.345, 0.234546, b"blahblah", b"cc"), (0.3434, 0.45454, b"blah", b"c")]
    to_share_pandas = pd.DataFrame(to_share, columns=["num1", "num2", "str8", "str2"])
    shm, np_array = sm.create_shared_block("test_pandas", to_share_pandas)
    sh_m = shared_memory.SharedMemory(name=shm.name)

    # column names and dtypes will need to be sent through to the code that is retrieving the shared memory
    np_array_retrieved = np.ndarray(
        shape=np_array.shape, dtype=np_array.dtype, buffer=sh_m.buf
    )
    assert isinstance(sh_m, shared_memory.SharedMemory)
    pd.testing.assert_frame_equal(to_share_pandas,
                                  pd.DataFrame(np_array_retrieved, columns=["num1", "num2", "str8", "str2"]))


def test_load_from_disk_and_share():
    sm = SharedManager(from_disk=True)
    data = ['all_measures', 'all_words', 'affixes']
    for name in data:
        assert name in sm.shares
