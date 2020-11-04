import pandas as pd
from tests.app.test_coding import DATA
from backend.app.ngrams import attach_ngrams
import numpy as np


def test_ngrams():
    df = pd.DataFrame(DATA, columns=['cue_clean', 'response_clean', 'cue_raw', 'response_raw'])
    df.set_index(['cue_clean', 'response_clean'], inplace=True)
    result = attach_ngrams(df)
    np.testing.assert_approx_equal(result.loc[b'fear', b'less']['ngram_angle'], 0.001315, significant=5)
