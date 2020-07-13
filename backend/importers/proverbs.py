# Import libraries to build Word2Vec model, and load Newsgroups data
import os
import sys
import re
from gensim.models import Word2Vec
from gensim.models.phrases import Phraser, Phrases
import pandas as pd
from backend.settings import DATA_FOLDER
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
PROVERBS_FILE = ''.join([DATA_FOLDER, '/proverbs.csv'])

def main():
    df = pd.read_csv(PROVERBS_FILE)
    df['tokenized'] = df['text'].apply(word_tokenize)
    return df

if __name__ == "__main__":
    main()
