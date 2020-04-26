# import requests
# import pymongo
# from backend.lib.db import get_acquisition_ages
#
# ACQUISITION_AGES = get_acquisition_ages()
#
# def process():
#
#     uri = 'https://www.merriam-webster.com/thesaurus/{word}'
#     word =
#     uri_filled = uri.format(word=word)
#     request_obj = requests.get(uri_filled)
#
#     result = request_obj.json()
#
#     if result['status'] == 'ok':
#         db = pymongo.MongoClient(host='localhost')['merriam_webster']
#         for record in result['data']:
#             record['_id'] = record['id']
#             record.pop('id')
#             db.data.update({'_id': record['_id']}, record, upsert=True)
#
# def get_next_word():
#
#
# def setup():
#     # parser = argparse.ArgumentParser(description='Import various types of word data')
#     # parser.add_argument('--wordassoc', action='store_true', help='Small world of words data')
#     # parser.add_argument('--responses', action='store_true', help='Responses table refresh')
#     # parser.add_argument('--affix', action='store_true', help='Affixes')
#     # parser.add_argument('--norms', action='store_true', help='Word association norms')
#     # args = parser.parse_args()
#     process()
#
#
# if __name__ == "__main__":
#     setup()
