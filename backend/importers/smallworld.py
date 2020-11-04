from zipfile import ZipFile
from backend import settings
import io
import csv
import datetime
from backend.lib.multiprocessor import job_pool
import time
import multiprocessing as mp
import argparse
from backend.lib.helpers.string import clean_word, get_country_code
from backend.importers.common import save_responses_and_users
from backend.db.layer import delete_responses, delete_users
import logging
from backend.settings import LOG_LEVEL

SOURCE = 'smallworldofwords'

logging.basicConfig(level=LOG_LEVEL)


def main(limit=None):
    delete_responses(SOURCE)
    delete_users(SOURCE)
    with ZipFile(settings.ZIP_FILE) as zf:
        infile = zf.open(settings.DATA_FILE, mode='r')
        infile = io.TextIOWrapper(infile, encoding='utf8', newline='')

        reader = csv.DictReader(infile)
        rows = [r for r in reader]
    if limit:
        rows = rows[:limit]
    print('Processing {} rows from small world of words'.format(len(rows)))
    start = time.perf_counter()
    job_number = mp.cpu_count() - 1
    jobs = job_pool(rows, job_number, process_data)
    total = []
    users = []
    for responses_saved, u in jobs:
        total.append(responses_saved)
        users.append(u)

    end = time.perf_counter()

    logging.info('time taken {}'.format(end - start))

    # take the individual results and sum them
    logging.info('saved {} records'.format(sum(total)))
    logging.info('saved {} users'.format(sum(users)))


def process_data(rows):
    responses = []
    users = set()
    country_codes = dict()
    responses_saved = 0
    users_saved = 0
    user_fields = ('external_user_id', 'age', 'nativelanguage', 'gender', 'education', 'city', 'country', 'source')
    for row in rows:
        # this data is of no use
        if row['R1'] == 'Unknown word':
            continue
        row.pop('')
        row['participantID'] = int(row['participantID'])
        row['age'] = int(row['age'])
        row['created_at'] = datetime.datetime.strptime(row['created_at'], '%Y-%m-%d %H:%M:%S')
        response = row['R1']
        # don't want to do lookups for every row
        if row['country'] not in country_codes:
            country_code = get_country_code(row['country'])
            country_codes.setdefault(row['country'], country_code)
        users.add(
            (int(row['participantID']), row['age'], row['nativeLanguage'], row['gender'], row['education'], row['city'],
             country_codes[row['country']], SOURCE))
        # should this conversion be noted as raw and changed?
        # formatted_wordassoc.append(raw_row)
        responses.append({'user_id': int(row['participantID']),
                          'created': row['created_at'],
                          'cue': row['cue'],
                          'response': response,
                          'cue_clean': clean_word(row['cue']),
                          'response_clean': clean_word(response),
                          'source': SOURCE})

    # import the full data and also a trimmed version for running through the logistic regression
    user_list = list(users)
    print(len(users))
    if responses:
        responses_saved, users_saved = save_responses_and_users(responses, user_list, user_fields)

    return responses_saved, users_saved


def setup():
    parser = argparse.ArgumentParser(description='DB setup')
    parser.add_argument('--limit', type=int, nargs='+',
                        help='limit the number of rows imported')
    args = parser.parse_args()
    if args.limit:
        main(limit=int(args.limit[0]))
    else:
        main()


if __name__ == "__main__":
    setup()
