import requests
import pymongo
from backend.lib.db import get_acquisition_ages
from bs4 import BeautifulSoup
import re
import argparse
import random
from backend.scrapers.user_agents import user_agent_list

ACQUISITION_AGES = get_acquisition_ages()
SOURCE = 'merriam_webster'



def process(word=None):
    uri = 'https://www.merriam-webster.com/dictionary/{word}'
    db = pymongo.MongoClient(host='localhost')['words_to_scrape']
    word_doc = db.words.find_one({"sources.{}".format(SOURCE): {'$exists': False}})
    if not word:
        word = word_doc['word']

    uri_filled = uri.format(word=word)
    user_agent = random.choice(user_agent_list)
    headers = {'User-Agent': user_agent}
    request_obj = requests.get(uri_filled, allow_redirects=True, headers=headers)
    soup = BeautifulSoup(request_obj.content, 'html.parser')
    definition_wrapper = soup.find(id="definition-wrapper")

    definitions = []
    synonyms = {}
    antonyms = {}
    pos_list = []
    definitions_container = definition_wrapper.findAll('div', id=lambda x: x and x.startswith('dictionary-entry-'))

    for defition_div in definitions_container:
        definitions_text = defition_div.find('span', {'class': 'dtText'})
        if definitions_text:
            definitions.append(re.sub(r'\s{2,}', ' ', definitions_text.get_text().strip().replace('\n', '')))

    syn_ant_wrapper = soup.find(id="synonyms-anchor")
    if syn_ant_wrapper:
        syn_ant_lists = syn_ant_wrapper.findAll('ul', {'class': 'mw-list'})

        if len(syn_ant_lists):
            for syn_ant_list in syn_ant_lists:
                previous = syn_ant_list.previous_sibling.lower()
                pos = getPos(soup, previous=previous)
                if pos:
                    pos_list.append(pos)
                if 'synonym' in previous:
                    synonyms[pos] = syn_ant_list.get_text()
                if 'antonym' in previous:
                    antonyms[pos] = syn_ant_list.get_text()

    if not len(pos_list):
        pos_list = [getPos(soup)]

    if definitions:
        result = {
            word:
                {
                    'definitions': definitions,
                    'synonyms': synonyms,
                    'antonyms': antonyms,
                    'pos': pos_list
                }
        }
        db.mw.update({'word': word}, result, upsert=True)
        db.words.update({'word': word}, {"$set": {"sources.{}".format(SOURCE): True}})


def getPos(soup, previous=None):
    pos = None
    if previous:
        if ':' in previous:
            pos = previous.split(':')[1].strip()
    if not pos:
        header = soup.findAll('div', {'class': 'row entry-header'})[0]
        pos = header.find('a').get_text().lower()
    return pos


def setup():
    parser = argparse.ArgumentParser(description='Import dictionary data')
    parser.add_argument('--word', help='Download the data from AWS')
    args = parser.parse_args()
    process(args.word)


if __name__ == "__main__":
    setup()
