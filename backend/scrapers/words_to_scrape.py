import pymongo
from backend.db.layer import get_acquisition_ages

ACQUISITION_AGES = get_acquisition_ages()
DATABASE = 'words_to_scrape'

def main():
    words = ACQUISITION_AGES.keys()
    db = pymongo.MongoClient(host='localhost')[DATABASE]
    ids = db.words.insert_many([{'word': w, 'sources': {}} for w in words]).inserted_ids
    print('Saved {} words'.format(db.words.count_documents({})))

if __name__ == "__main__":
    main()
