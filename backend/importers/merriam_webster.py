from pathlib import Path
import time
import json
import argparse
from backend.lib.multiprocessor import job_pool
import multiprocessing as mp
from os import listdir
from backend.importers.common import get_external_backup
import logging
from backend.settings import AWS_USERNAME, AWS_SERVER
import pandas as pd
import re

SOURCE = 'merriam_webster'
DATABASE = 'merriam_webster'
HOME = str(Path.home())


def main(download=False):
    target = '{}/dev/uniqueness/data/'.format(HOME)
    id_rsa = '{}/.ssh/id_rsa'.format(HOME)
    if download:
        get_external_backup(id_rsa, AWS_USERNAME, AWS_SERVER, "/tmp/{}.json".format(DATABASE), target)
    files_to_load = ['{}{}'.format(target, x) for x in listdir(target) if DATABASE in x]
    multiprocess_data(files_to_load)


def multiprocess_data(files_to_load):
    json_data = []
    for target_file in files_to_load:
        with open(target_file, 'r') as f:
            json_data.extend(list(json.loads(f.read())))
    print('Processing {} dictionary entries...'.format(len(json_data)))

    start = time.perf_counter()
    job_number = mp.cpu_count() - 1
    jobs = job_pool(json_data, job_number, process_data)
    df = pd.DataFrame([j for i in jobs for j in i])
    df.to_csv('{}/dev/uniqueness/data/synonyms_antonyms.csv'.format(HOME), index=False)
    end = time.perf_counter()

    logging.info('time taken {}'.format(end - start))


def process_data(json_data):
    words = []
    done = set()
    for record in json_data:
        word = list(record.keys())[1]
        if word not in done:
            done.add(word)

            synonyms = process_related(record[word]['synonyms'])
            antonyms = process_related(record[word]['antonyms'])
            for pos in synonyms:
                word_record = dict(word=word)
                word_record['part_of_speech'] = pos
                word_record['synonyms'] = ','.join(synonyms[pos])
                word_record['antonyms'] = ','.join(antonyms.get(pos, []))
                words.append(word_record)

    return words


def process_related(record):
    # "synonyms": {
    #     "verb (1)": "jump, start, startle"
    # },
    result = {}
    for k, v in record.items():
        part_of_speech = k.split(' ')[0]
        v_clean = re.sub('\s{2,}', ' ', v.strip().replace("\n", " "))
        words = list(set(v_clean.split(', ')))
        result[part_of_speech] = words
    return result


def setup():
    parser = argparse.ArgumentParser(description='Import human brain cloud data')
    parser.add_argument('--download', action='store_true', help='Download the data from AWS')
    args = parser.parse_args()
    main(download=args.download)


if __name__ == "__main__":
    setup()
