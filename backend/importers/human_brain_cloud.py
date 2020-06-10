from pathlib import Path
import subprocess
import time
import json
import argparse
from backend.db.db import delete_responses, save_cue_response_pairs
from backend.lib.multiprocessor import job_pool
import multiprocessing as mp
from backend.lib.helpers.string import clean_word
from os import listdir

SOURCE = 'humanbraincloud'


def main(download=False):
    home = str(Path.home())
    target = '{}/dev/uniqueness/data/'.format(home)
    if download:
        id_rsa = '{}/.ssh/id_rsa'.format(home)
        p = subprocess.Popen(
            ["scp", id_rsa, "ubuntu@ec2-52-14-41-217.us-east-2.compute.amazonaws.com:/tmp/humanbrain.json", target])
        while p.poll() is None:
            print('\nWaiting for data...\n')
            time.sleep(5)
    files_to_load = ['{}{}'.format(target, x) for x in listdir(target) if 'humanbrain' in x]
    multiprocess_human_brain_cloud_data(files_to_load)


def multiprocess_human_brain_cloud_data(files_to_load, delete=None):
    json_data = []
    for target_file in files_to_load:
        with open(target_file, 'r') as f:
            json_data.extend(list(json.loads(f.read())))
    print('Processing {} responses...'.format(len(json_data)))
    delete_responses(SOURCE)
    start = time.perf_counter()
    job_number = mp.cpu_count() - 1
    jobs = job_pool(json_data, job_number, process_data)
    end = time.perf_counter()
    print('time taken {}'.format(end - start))

    # take the individual results and sum them
    print('saved {} records'.format(sum(jobs)))


def process_data(json_data):
    responses = []
    ids = set()
    # distinct_responses = set()
    count = 0
    for record in json_data:
        count += 1
        if isinstance(record['_id'], str):

            if record['_id'] not in ids:
                ids.add(record['_id'])
                formatted_record = format_record(record)
                responses.append(formatted_record)
                # distinct_responses.add(formatted_record['response'])
                # {'_id': '10537190',
                #  'playerCc': 'al',
                #  'playerId': '325299',
                #  'playerName': '',
                #  't': '2020-03-30T18:58:28Z',
                #  'w1': 'degenerate',
                #  'w2': 'mamus'}
        else:
            # a few records at the start were mistakenly stored grouped
            for sub_record in record['data']:
                if sub_record['id'] not in ids:
                    ids.add(sub_record['id'])
                    formatted_sub_record = format_record(sub_record)
                    responses.append(formatted_sub_record)
                    # distinct_responses.add(formatted_sub_record['response'])
    if responses:
        save_cue_response_pairs(responses)
        return len(responses)


def format_record(record):
    response = {}
    response['id'] = record.get('_id') or record.get('id')
    response['cue'] = record['w1']
    response['response'] = record['w2']
    response['cue_clean'] = clean_word(record['w1'])
    response['response_clean'] = clean_word(record['w2'])
    response['user_id'] = record['playerId']
    response['time_duration'] = 0
    response['created'] = record['t']
    response['source'] = SOURCE

    return response


def setup():
    parser = argparse.ArgumentParser(description='Import human brain cloud data')
    parser.add_argument('--download', action='store_true', help='Download the data from AWS')
    args = parser.parse_args()
    main(download=args.download)


if __name__ == "__main__":
    setup()
