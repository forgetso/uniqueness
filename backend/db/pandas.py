import pandas as pd
from backend.settings import SYNONYMS_ANTONYMS_FILE, CWD


# data layer for tables stored as files

def get_synonyms_antonyms():
    df = pd.read_csv(CWD + SYNONYMS_ANTONYMS_FILE)
    return df

