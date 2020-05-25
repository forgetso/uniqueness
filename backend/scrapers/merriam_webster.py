import requests
import pymongo
from backend.db.db import get_acquisition_ages
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
    definitions = None
    sub_category = None
    definitions_container = definition_wrapper.findAll('div', id=lambda x: x and x.startswith('dictionary-entry-'))
    # todo find out what all these different entries are
    if not definitions_container:
        definitions_container = definition_wrapper.findAll('div', id=lambda x: x and x.startswith('medical-entry-'))
        if definitions_container:
            sub_category = 'medical'

    if not definitions_container and definition_wrapper.find('h1', {'class': 'mispelled-word'}):
        alternate_spelling = definition_wrapper.find_next('p', {'class': 'spelling-suggestions'}).find_next('a').get_text()
    else:
        definitions, alternate_spelling = getDefinitions(definitions_container)

    synonyms, antonyms, pos_list = getRelations(soup)

    if not len(pos_list) and definitions:
        pos_list = [getPos(soup)]

    if definitions or alternate_spelling:
        storeWord(db, word, definitions, synonyms, antonyms, pos_list, alternate_spelling, sub_category)


def getDefinitions(definitions_container):
    definitions = []
    alternate_spelling = None
    for defition_div in definitions_container:
        definitions_texts = defition_div.findAll('span', {'class': 'dtText'})
        for definition_text in definitions_texts:
            definitions.append(re.sub(r'\s{2,}', ' ', definition_text.get_text().strip().replace('\n', '')))
        definitions_texts = defition_div.findAll('span', {'class': 'unText'})
        for definition_text in definitions_texts:
            definitions.append(re.sub(r'\s{2,}', ' ', definition_text.get_text().strip().replace('\n', '')))

    if not len(definitions):
        for defition_div in definitions_container:
            cxl = defition_div.find('span', {'class': 'cxl'})
            if cxl.get_text() == 'British spelling of':
                alternate_spelling = cxl.next_sibling.next_sibling['href']
    return definitions, alternate_spelling


def getRelations(soup):
    synonyms = {}
    antonyms = {}
    pos_list = []
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
    return synonyms, antonyms, pos_list


def getPos(soup, previous=None):
    pos = None
    if previous:
        if ':' in previous:
            pos = previous.split(':')[1].strip()
    if not pos:
        header = soup.findAll('div', {'class': 'row entry-header'}) + soup.findAll('div', {
            'class': 'row entry-header long-headword'})
        if header:
            header = header[0]
            pos = header.find('a').get_text().lower()

    return pos


def storeWord(db, word, definitions, synonyms, antonyms, pos_list, alternate_spelling, sub_category):
    result = {
        word:
            {
                'definitions': definitions,
                'synonyms': synonyms,
                'antonyms': antonyms,
                'pos': pos_list,
                'alternative_spelling': alternate_spelling,
                'sub_category': sub_category
            }
    }
    db.mw.update({'word': word}, result, upsert=True)
    db.words.update({'word': word}, {"$set": {"sources.{}".format(SOURCE): True}})


def setup():
    parser = argparse.ArgumentParser(description='Import dictionary data')
    parser.add_argument('--word', help='Download the data from AWS')
    args = parser.parse_args()
    process(args.word)


if __name__ == "__main__":
    setup()
