import pandas as pd
from backend.settings import WORD_FREQUENCIES


def get_word_filter():
    # Take only words in the 2k to 3k band from the British National Corpus as per Fitzgerald et. al
    df = pd.read_csv(WORD_FREQUENCIES, sep='\t')
    word_filter = df[2000:3999].copy()
    word_filter['word'] = word_filter['Word'].str.lower().str.encode('utf8')
    return word_filter['word'].values


if __name__ == "__main__":
    get_word_filter()
