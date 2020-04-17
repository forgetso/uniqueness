from pathlib import Path
import subprocess
import time
import json
import argparse
import pprint
from backend.lib.db import save_cue_response_pairs, delete_responses, save_corrections_and_correction
from backend.lib.spelling import Spelling

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
            print('waiting for data...')
            time.sleep(5)
            process_human_brain_cloud_data(target_file)
    else:
        process_human_brain_cloud_data(target_file)


def process_human_brain_cloud_data(target_file):
    with open(target_file, 'r') as f:
        json_data = json.loads(f.read())

    responses = []
    ids = set()
    for record in json_data:
        if isinstance(record['_id'], str):

            if record['_id'] not in ids:
                ids.add(record['_id'])
                responses.append(format_record(record))
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
                    responses.append(format_record(sub_record))
    if responses:
        delete_responses(SOURCE)
        ids = save_cue_response_pairs(responses)
        for idx, response in enumerate(responses):
            if 'corrections' in response:
                correction = None
                if len(response['corrections']) == 1:
                    correction = response['corrections'][0]
                save_corrections_and_correction(ids[idx], response['response'], response['corrections'], correction)

    print('{} records saved'.format(len(ids)))


def format_record(record):
    response = {}
    response['id'] = record.get('_id') or record.get('id')
    response['cue'] = record['w1']
    response['response'] = record['w2']
    response['user_id'] = record['playerId']
    response['time_duration'] = 0
    response['created'] = record['t']
    response['source'] = SOURCE
    speller = Spelling(response['cue'], response['response'])
    corrections = speller.get_corrections()
    if corrections:
        response['corrections'] = corrections
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
