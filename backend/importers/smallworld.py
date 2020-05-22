from zipfile import ZipFile
from backend import settings
import io
import csv
import datetime
import backend.db.db as db
from backend.helpers.string import check_spelling
from backend.lib.setup import db_setup
from backend.lib.multiprocessor import dispatch_jobs, jobs_running
import time
import multiprocessing as mp
import argparse
from backend.helpers.string import clean_word

SOURCE = 'smallworldofwords'


def main(limit=None):
    db_setup(wordassoc=True)
    db.delete_responses(SOURCE)
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
        jobs = dispatch_jobs(rows, job_number, process_data)
        while jobs_running(jobs):
            print(
                '---------------------------------------------Jobs running-----------------------------------------------')
            time.sleep(5)
        else:
            end = time.perf_counter()
            print('time taken {}'.format(end - start))


def process_data(rows):
    formatted_wordassoc = []
    formatted_responses = []
    fieldnames = list(rows[0].keys())[1:]
    for row in rows:
        # this data is of no use
        if row['R1'] == 'Unknown word':
            continue
        corrections_list = None
        row.pop('')
        raw_row = tuple(row.values())
        row['participantID'] = int(row['participantID'])
        row['age'] = int(row['age'])
        row['created_at'] = datetime.datetime.strptime(row['created_at'], '%Y-%m-%d %H:%M:%S')
        row['corrections'] = {}
        response = row['R1']
        if len(row['cue']) and len(response):
            incorrect_spelling, corrections_list, corrections_dict = check_spelling(str(row['cue']), str(response))
            # pull out the actual corrections strings into a list instead of the SuggestItem class
            corrections_dict = {k: [x.term for x in v] for k, v in corrections_dict.items()}
            row['corrections'] = [x.term for x in corrections_list]
        row['correction'] = None
        if len(corrections_list) == 1 and corrections_list[0].term != response.lower():
            # TODO smarter spelling corrections here, use relation stength of suggestions to cue, soundex, etc.
            row['correction'] = corrections_list[0].term
        # should this conversion be noted as raw and changed?
        formatted_wordassoc.append(raw_row)
        formatted_responses.append({'user_id': row['participantID'],
                                    'created': row['created_at'],
                                    'cue': row['cue'],
                                    'response': response,
                                    'cue_clean': clean_word(row['cue']),
                                    'response_clean': clean_word(response),
                                    'source': SOURCE,
                                    'corrections': row['corrections'],
                                    'correction': row['correction']})

    # import the full data and also a trimmed version for running through the logistic regression
    db.execute_values_wrap('wordassoc', fieldnames, formatted_wordassoc)

    response_ids, correction_ids = db.save_cue_response_pairs_and_corrections(formatted_responses)


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
