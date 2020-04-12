import os
from backend.settings_private import *
CWD = os.getcwd()
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
