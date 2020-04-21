from pathlib import Path
import subprocess
import time
import json
import argparse
import pprint
from backend.lib.db import delete_responses, save_cue_response_pairs_and_corrections
from backend.lib.multiprocessor import dispatch_jobs, jobs_running
from backend.app.coding import check_spelling
import multiprocessing as mp

SOURCE = 'humanbraincloud'


def main(download=False):
    home = str(Path.home())
    target = '{}/dev/uniqueness/data/'.format(home)
    target_file = '{}/humanbrain.json'.format(target)
    if download:
        id_rsa = '{}/.ssh/id_rsa'.format(home)
        p = subprocess.Popen(
            ["scp", id_rsa, "ubuntu@ec2-52-14-41-217.us-east-2.compute.amazonaws.com:/tmp/humanbrain.json", target])
        # scp -i ~/.ssh/id_rsa ubuntu@ec2-52-14-41-217.us-east-2.compute.amazonaws.com:/tmp/humanbrain.json ~/dev/uniqueness/data/humanbrain.json
        while p.poll() is None:
            print('Waiting for data...')
            time.sleep(5)
            multiprocess_human_brain_cloud_data(target_file)
    else:
        multiprocess_human_brain_cloud_data(target_file)


def multiprocess_human_brain_cloud_data(target_file):
    with open(target_file, 'r') as f:
        json_data = list(json.loads(f.read()))
    print('Processing {} responses...'.format(len(json_data)))
    delete_responses(SOURCE)
    start = time.perf_counter()
    job_number = mp.cpu_count() - 1
    jobs = dispatch_jobs(json_data, job_number, process_data)
    while jobs_running(jobs):
        print(
            '---------------------------------------------Jobs running-----------------------------------------------')
        time.sleep(5)
    else:
        end = time.perf_counter()
        print('time taken {}'.format(end - start))


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
        save_cue_response_pairs_and_corrections(responses)
    print('{} records saved'.format(len(responses)))


def format_record(record):
    response = {}
    response['id'] = record.get('_id') or record.get('id')
    response['cue'] = record['w1']
    response['response'] = record['w2']
    response['user_id'] = record['playerId']
    response['time_duration'] = 0
    response['created'] = record['t']
    response['source'] = SOURCE

    incorrect_spelling, corrections_list, corrections_dict = check_spelling(response['cue'], response['response'])
    # pull out the actual corrections strings into a list instead of the SuggestItem class
    corrections_dict = {k: [x.term for x in v] for k, v in corrections_dict.items()}
    response['corrections'] = [x.term for x in corrections_list]
    if len(corrections_list) == 1 and corrections_list[0].term != response['response']:
        #TODO smarter spelling corrections here, use relation stength of suggestions to cue, soundex, etc.
        response['correction'] = corrections_list[0].term
    # TODO include these later and store country for players using UI
    # response['country'] = record['playerCc']
    # response['playerName'] = record['playerName']
    return response


def setup():
    parser = argparse.ArgumentParser(description='Import human brain cloud data')
    parser.add_argument('--download', action='store_true', help='Download the data from AWS')
    args = parser.parse_args()
    main(download=args.download)


if __name__ == "__main__":
    setup()
