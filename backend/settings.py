import os
from multiprocessing import cpu_count
import logging
CWD = os.getcwd()
PROJECT_ROOT = '/home/chris/dev/uniqueness'
PG_USER = 'backend'
PG_PASSWORD = "bhVNUe7BsZ3mp2UV"
ZIP_FILE = ''.join([CWD, '/data/SWOW-EN.complete.csv.zip'])
DATA_FILE = 'SWOW-EN.complete.csv'
AFFIX_FILE = ''.join([CWD, '/data/affixes.csv'])
NORMS_FOLDER = ''.join([CWD, '/data/South_Florida_Norms'])
PREVALENCES_FILE = ''.join([CWD, '/data/prevalences.csv'])
CONCRETENESS_FILE = ''.join([CWD, '/data/concreteness.txt'])
AGE_FILE = ''.join([CWD, '/data/age_of_acquisition.csv'])
EMOTIONS_FILE = ''.join([CWD, '/data/affective_ratings.csv'])
SENSE2VEC_FOLDER = ''.join([CWD, '/data/s2v_reddit_2019_lg'])
SEMANTIC_SPACE_FOLDER = ''.join([CWD, '/data/pragmatic_atlas'])
ORTHOGRAPHIC_FILE = ''.join([CWD, '/data/orthographic_neighbourhoods.csv'])
PHONOGRAPHIC_FILE = ''.join([CWD, '/data/IPhOD2_Words.txt'])
ALL_WORDS_FILE = ''.join([CWD, '/data/words_dictionary.json'])
DATA_FOLDER = ''.join([CWD, '/data'])
JOB_NUMBER = int(cpu_count() / 2)
LOG_LEVEL = logging.INFO
IP='77.102.139.244'