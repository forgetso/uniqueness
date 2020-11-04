import os
from multiprocessing import cpu_count
import logging
from pathlib import Path
CWD = os.getcwd()
SHARE_PREFIX = 'unq'
PROJECT_ROOT = '/home/chris/dev/uniqueness'
PG_USER = 'backend'
ZIP_FILE = ''.join([CWD, '/data/SWOW-EN.complete.csv.zip'])
DATA_FILE = 'SWOW-EN.complete.csv'
AFFIX_FILE = ''.join([CWD, '/data/affixes.csv'])
NORMS_FOLDER = ''.join([CWD, '/data/South_Florida_Norms'])
PREVALENCES_FILE = ''.join([CWD, '/data/prevalences.csv'])
CONCRETENESS_FILE = ''.join([CWD, '/data/concreteness.txt'])
AGE_FILE = ''.join([CWD, '/data/age_of_acquisition.csv'])
EMOTIONS_FILE = ''.join([CWD, '/data/affective_ratings.csv'])
SENSE2VEC_FOLDER = ''.join([CWD, '/data/s2v'])
SEMANTIC_SPACE_FOLDER = ''.join([CWD, '/data/pragmatic_atlas'])
ORTHOGRAPHIC_FILE = ''.join([CWD, '/data/orthographic_neighbourhoods.csv'])
PHONOGRAPHIC_FILE = ''.join([CWD, '/data/IPhOD2_Words.txt'])
ALL_WORDS_FILE = ''.join([CWD, '/data/words_dictionary.json'])
DATA_FOLDER = ''.join([CWD, '/data'])
JOB_NUMBER = int(cpu_count() / 2)
LOG_LEVEL = logging.DEBUG
WORD_FREQUENCIES = ''.join([CWD, '/data/bncfreqs.csv'])
NGRAMS_DATA = Path(CWD).joinpath('data/ngrams.csv')
SYNONYMS_ANTONYMS_FILE = '/data/synonyms_antonyms.csv'
