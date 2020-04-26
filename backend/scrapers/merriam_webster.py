import requests
import pymongo
from backend.lib.db import get_acquisition_ages
from bs4 import BeautifulSoup
import re

ACQUISITION_AGES = get_acquisition_ages()
SOURCE = 'merriam_webster'


def process():
    uri = 'https://www.merriam-webster.com/dictionary/{word}'
    db = pymongo.MongoClient(host='localhost')['words_to_scrape']
    word_doc = db.words.find_one({"sources.{}".format(SOURCE): {'$exists': False}})
    print(word_doc)
    word = word_doc['word']

    uri_filled = uri.format(word=word)
    request_obj = requests.get(uri_filled)
    soup = BeautifulSoup(request_obj.content, 'html.parser')
    definition_wrapper = soup.find(id="definition-wrapper")

    definitions = []
    synonyms = {}
    antonyms = {}

    definitions_container = definition_wrapper.findAll('div', id=lambda x: x and x.startswith('dictionary-entry-'))

    for defition_div in definitions_container:
        definitions_text = defition_div.find('span', {'class': 'dtText'})
        definitions.append(re.sub(r'\s{2,}', ' ', definitions_text.get_text().strip().replace('\n', '')))

    syn_ant_wrapper = soup.find(id="synonyms-anchor")
    if syn_ant_wrapper:
        syn_ant_lists = syn_ant_wrapper.findAll('ul', {'class': 'mw-list'})

        if len(syn_ant_lists):
            for syn_ant_list in syn_ant_lists:
                previous = syn_ant_list.previous_sibling.lower()
                pos = previous.split(':')[1].strip()
                if 'synonym' in previous:
                    synonyms[pos] = syn_ant_list.get_text()
                if 'antonym' in previous:
                    antonyms[pos] = syn_ant_list.get_text()

    if definitions:
        result = {word: {'definitions': definitions, 'synonyms': synonyms, 'antonyms': antonyms}}

        db.mw.update({'word': word}, result, upsert=True)
        db.words.update({'word': word}, {"$set": {"sources.{}".format(SOURCE): True}})


def setup():
    process()


if __name__ == "__main__":
    setup()
