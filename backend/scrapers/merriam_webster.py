import requests
import pymongo
from bs4 import BeautifulSoup
import re
import argparse
import random
from backend.scrapers.user_agents import user_agent_list

SOURCE = 'merriam_webster'
REF_REGEX = re.compile(r'^cxl')
DATABASE = 'words_to_scrape'
SUB_CATEGORIES = ['legal', 'medical']


def process(word=None, mode='thesaurus'):
    uri = 'https://www.merriam-webster.com/{mode}/{word}'
    db = pymongo.MongoClient(host='localhost')[DATABASE]
    word_doc = db.words.find_one({"sources.{}".format(SOURCE): {'$exists': False}})
    if not word:
        if word_doc:
            word = word_doc['word']
        else:
            print('all words scraped!')
            exit(0)

    uri_filled = uri.format(word=word, mode=mode)
    user_agent = random.choice(user_agent_list)
    headers = {'User-Agent': user_agent}
    request_obj = requests.get(uri_filled, allow_redirects=True, headers=headers)
    soup = BeautifulSoup(request_obj.content, 'html.parser')

    no_spelling_suggestions = soup.find('div', {'class': 'no-spelling-suggestions'})
    if no_spelling_suggestions:
        print('word not found in {}'.format(mode))
        exit()

    definition_wrapper = soup.find(id="definition-wrapper".format(mode))
    definitions = None
    sub_category = None
    alternate_spelling = None
    entry_id = '{}-entry-'.format(mode)
    definitions_container = definition_wrapper.findAll('div', id=lambda x: x and x.startswith(entry_id))

    # todo find out what all these different entries are
    if mode == 'dictionary':
        if not definitions_container:
            for sub_category in SUB_CATEGORIES:
                definitions_container = definition_wrapper.findAll('div',
                                                                   id=lambda x: x and x.startswith(sub_category + '-'))
                if definitions_container:
                    break

        if not definitions_container:
            missing = definition_wrapper.find('p', {'class': 'missing-query'})
            if not missing and definition_wrapper.find('h1', {'class': 'mispelled-word'}):
                alternate_spelling = definition_wrapper.find_next('p', {'class': 'spelling-suggestions'}).find_next(
                    'a').get_text()
        else:
            definitions, alternate_spelling = getDefinitions(definitions_container)

        synonyms, antonyms, pos_list = getRelations(soup)

        if not len(pos_list) and definitions:
            pos = getPos(soup)
            if pos:
                pos_list = [pos]

        if definitions or alternate_spelling:
            storeWord(db, word, definitions, synonyms, antonyms, pos_list, alternate_spelling, sub_category)
    elif mode == 'thesaurus':
        synonyms = get_thesaurus_relations(definition_wrapper, 'syn')
        antonyms = get_thesaurus_relations(definition_wrapper, 'ant')
        storeRelations(db, word, synonyms, antonyms)


def get_thesaurus_relations(definition_wrapper, rtype):
    relations = []
    rels_list = [x.findAll('div', {'class': 'synonyms_list'}) for x in
                 definition_wrapper.findAll('span', {'class': '{}-list'.format(rtype)})]
    rels_list = [item for sublist in rels_list for item in sublist]
    rels_list = [item.strip() for sublist in [x.get_text().split(',') for x in rels_list] for item in sublist]
    if len(rels_list):
        relations.extend(rels_list)
        relations = [re.sub('\s\((or|also)\s(\w+)\)', ',\\2', y).split(',') for y in relations]
        relations = [re.sub('\[\w+\]', '', item) for sublist in relations for item in sublist]
    print(relations)
    return relations


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

    # do this as well instead of either/or
    for defition_div in definitions_container:
        cxl = defition_div.find('span', {'class': REF_REGEX})
        if cxl:
            next_sibling = cxl.next_sibling.next_sibling
            cxl_text = cxl.get_text()
            if hasattr(next_sibling, 'href'):
                alternate_spelling = next_sibling['href']
                if not cxl_text == 'British spelling of':
                    definitions.append(' '.join([cxl_text, next_sibling['href']]))
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
            div = header[0].find('div', {'class': 'col-12'})
            if div:
                span = div.find('span', {'class': 'fl'})
                if span:
                    pos = span.get_text()
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


def storeRelations(db, word, synonyms, antonyms):
    result = {
        'word': word,
        'synonyms': synonyms,
        'antonyms': antonyms,
    }
    db.relations.update({'word': word}, result, upsert=True)
    db.words.update({'word': word}, {"$set": {"sources.{}".format(SOURCE): True}})


def setup():
    parser = argparse.ArgumentParser(description='Import dictionary data')
    parser.add_argument('--word', help='Run for specific word')
    parser.add_argument('--mode', help='Dictionary or thesaurus', default='thesaurus')
    args = parser.parse_args()
    process(args.word, args.mode)


if __name__ == "__main__":
    setup()
