from backend.lib.multiprocessor import chunks
from tests.lib.helpers.test_dataframe import DATA
from backend.settings import JOB_NUMBER
import pytest
import numpy as np


def test_chunks():
    result = chunks(['a'], 2)
    assert (result == [['a']])

    result = chunks(['a', 'b', 'c'], n=4)
    assert (result == [['a', 'b', 'c']])

    result = chunks(['a', 'b', 'c'], n=2)
    assert (result == [['a', 'b'], ['c']])

    result = chunks(['a', 'b', 'c'], n=1)
    assert (result == [['a'], ['b'], ['c']])

    total = len(DATA)
    chunk_size = int(total / JOB_NUMBER)
    result = chunks(DATA, int(chunk_size))
    assert (result == [DATA])

    with pytest.raises(TypeError):
        chunks(DATA, 'blah')

    with pytest.raises(TypeError):
        chunks(DATA, np.nan)

    with pytest.raises(TypeError):
        chunks(DATA, {})

    with pytest.raises(TypeError):
        chunks(DATA, None)
