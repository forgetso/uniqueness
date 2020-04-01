import os

CWD = os.getcwd()
PG_USER = 'backend'
PG_PASSWORD = 'thewayoftheword'
ZIP_FILE = ''.join([CWD, '/data/SWOW-EN.complete.csv.zip'])
DATA_FILE = 'SWOW-EN.complete.csv'
AFFIX_FILE = ''.join([CWD, '/data/affixes.csv'])
NORMS_FOLDER = ''.join([CWD, '/data/South_Florida_Norms'])
PREVALENCES_FOLDER = ''.join([CWD, '/data/prevalences.csv'])
SENSE2VEC_FOLDER = ''.join([CWD, '/data/s2v_reddit_2019_lg'])
