import requests
import pymongo
import random
import string
import argparse
import time
import logging
from backend.scrapers.user_agents import user_agent_list
import json

logging.basicConfig(level=logging.DEBUG)

URLS_TO_VISIT = [
    'https://www.humanbraincloud.com',
    'https://www.humanbraincloud.com/api/?op=GetRandomWords&uid={random_id}&_={timestamp}',
    'https://www.humanbraincloud.com/api/cache/stats/MostSubmittedWords.json?uid={random_id}',
    'https://www.humanbraincloud.com/api/cache/stats/StrongestConnections.json?uid={random_id}',
    'https://www.humanbraincloud.com/api/cache/stats/MostProlific.json?uid={random_id}',
    'https://www.humanbraincloud.com/api/?op=GetPlayerData&uid={random_id}&_={timestamp}',
    'https://www.humanbraincloud.com/api/cache/stats/MiscGlobalStats.json?uid={random_id}&_={timestamp}',
    'https://www.humanbraincloud.com/api/cache/stats/MostProlificRecently.json?uid={random_id}'
]


def process():
    # wait a random amount of seconds so that the requests don't go out at exactly the same time every 5 minutes
    time.sleep(random.randint(1, 30))
    random_id = ''.join(random.choices(string.ascii_letters + string.digits, k=16)).lower()
    fake_visit(random_id)
    uri = 'https://humanbraincloud.com/api/cache/stats/RecentConnections.json?uid={random_id}&_={timestamp}'
    request_obj = get_uri(uri, random_id)

    if request_obj.status_code != 200:
        raise PermissionError(
            'Not able to access humanbraincloud.com. Server returned {}'.format(json.dumps(request_obj)))

    result = request_obj.json()
    if result['status'] != 'ok':
        raise ConnectionError('Error in data from humanbraincloud')

    if result['status'] == 'ok':
        db = pymongo.MongoClient(host='localhost')['humanbrain']
        for record in result['data']:
            record['_id'] = record['id']
            record.pop('id')
            db.data.update({'_id': record['_id']}, record, upsert=True)
        logging.debug('Collected {} records'.format(len(result['data'])))


def fake_visit(random_id):
    for uri in URLS_TO_VISIT:
        get_uri(uri, random_id)


def get_uri(uri, random_id):
    user_agent = random.choice(user_agent_list)
    headers = {
        'Accept': 'application/json, text/javascript, */*; q=0.01',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept-Language': 'en-GB,en-US;q=0.9,en;q=0.8',
        'Connection': 'keep-alive',
        'Host': 'www.humanbraincloud.com',
        'Referer': 'https: // www.humanbraincloud.com/',
        'Sec-Fetch-Dest': 'empty',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Site': 'same-origin',
        'User-Agent': user_agent,
        'X-Requested-With': 'XMLHttpRequest',
    }
    timestamp = int(time.time())
    uri_filled = uri.format(random_id=random_id, timestamp=timestamp)
    r = requests.get(uri_filled, headers=headers)
    return r


def setup():
    # parser = argparse.ArgumentParser(description='Import various types of word data')
    # parser.add_argument('--wordassoc', action='store_true', help='Small world of words data')
    # parser.add_argument('--responses', action='store_true', help='Responses table refresh')
    # parser.add_argument('--affix', action='store_true', help='Affixes')
    # parser.add_argument('--norms', action='store_true', help='Word association norms')
    # args = parser.parse_args()
    process()


if __name__ == "__main__":
    setup()
