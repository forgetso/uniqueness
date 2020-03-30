import requests
import pymongo
import random
import string
import argparse
import time


def process():
    uri = 'https://humanbraincloud.com/api/cache/stats/RecentConnections.json?uid={random_id}&_={timestamp}'

    random_id = ''.join(random.choices(string.ascii_letters + string.digits, k=16)).lower()
    timestamp = int(time.time())
    uri_filled = uri.format(random_id=random_id, timestamp=timestamp)
    request_obj = requests.get(uri_filled)

    result = request_obj.json()

    if result['status'] == 'ok':
        db = pymongo.MongoClient(host='localhost')['humanbrain']
        for record in result['data']:
            db.data.update({'_id': record['id']}, record, upsert=True)


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
