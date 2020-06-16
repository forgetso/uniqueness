from backend.app.semantic_atlas import check_semantic_atlas
import pandas as pd
from nltk import PorterStemmer

ps = PorterStemmer()


# TODO more tests of semantic atlas coding

def test_check_semantic_atlas():
    df = pd.DataFrame(
        {'cue_clean': [b'test', b'bread', b'multiple', b'both_multiple', b'single'],
         'response_clean': [b'tester', b'butter', b'multiple_choice', b'both_long', b'double_single'],
         'cue_stem': ['test', 'bread', ps.stem('multiple'), ps.stem('both multiple'), ps.stem('single')],
         'response_stem': ['test', 'butter', ps.stem('multiple choice'), ps.stem('both long'), ps.stem('double single')]
         })
    df['cue_split'] = df['cue_clean'].str.decode('utf8').str.split('_')
    df['response_split'] = df['response_clean'].str.decode('utf8').str.split('_')
    df.set_index(['cue_clean', 'response_clean'], inplace=True)
    result = check_semantic_atlas(df)
    assert (len(result) == len(df))
