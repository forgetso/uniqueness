import os
from multiprocessing import cpu_count
import logging
from pathlib import Path

#TODO Path join all paths
SHARE_PREFIX = 'unq'
PROJECT_ROOT = '/home/chris/dev/uniqueness'
PG_USER = 'backend'
ZIP_FILE = ''.join([PROJECT_ROOT, '/data/SWOW-EN.complete.csv.zip'])
DATA_FILE = 'SWOW-EN.complete.csv'
AFFIX_FILE = ''.join([PROJECT_ROOT, '/data/affixes.csv'])
NORMS_FOLDER = ''.join([PROJECT_ROOT, '/data/South_Florida_Norms'])
PREVALENCES_FILE = ''.join([PROJECT_ROOT, '/data/prevalences.csv'])
CONCRETENESS_FILE = ''.join([PROJECT_ROOT, '/data/concreteness.txt'])
AGE_FILE = ''.join([PROJECT_ROOT, '/data/age_of_acquisition.csv'])
EMOTIONS_FILE = ''.join([PROJECT_ROOT, '/data/affective_ratings.csv'])
SENSE2VEC_FOLDER = ''.join([PROJECT_ROOT, '/data/s2v'])
SEMANTIC_SPACE_FOLDER = ''.join([PROJECT_ROOT, '/data/pragmatic_atlas'])
ORTHOGRAPHIC_FILE = ''.join([PROJECT_ROOT, '/data/orthographic_neighbourhoods.csv'])
PHONOGRAPHIC_FILE = ''.join([PROJECT_ROOT, '/data/IPhOD2_Words.txt'])
FREQUENCIES_FILE = Path(PROJECT_ROOT).joinpath('./data/SUBTLEXus74286wordstextversion.txt')
ALL_WORDS_FILE = ''.join([PROJECT_ROOT, '/data/words_dictionary.json'])
DATA_FOLDER = ''.join([PROJECT_ROOT, '/data'])
JOB_NUMBER = int(cpu_count() / 2)
LOG_LEVEL = logging.DEBUG
WORD_FREQUENCIES = ''.join([PROJECT_ROOT, '/data/bncfreqs.csv'])
NGRAMS_DATA = Path(PROJECT_ROOT).joinpath('data/ngrams.csv')
SYNONYMS_ANTONYMS_FILE = Path(PROJECT_ROOT).joinpath('./data/synonyms_antonyms.csv').absolute()
