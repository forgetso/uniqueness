import requests
import bs4
import random
from backend.scrapers.user_agents import user_agent_list
import re
from urllib import parse
import pandas as pd
from backend.settings import DATA_FOLDER, LOG_LEVEL
import logging

logging.basicConfig(level=LOG_LEVEL)


def standard(soup, selector):
    items = soup.select(selector)
    return items


# proverbs and idioms scraped from the net

def main():
    SOURCES = [
        ('https://www.phrases.org.uk/meanings/proverbs.html',
         'p[class="phrase-list"]', None),
        ('https://en.wikiquote.org/wiki/English_proverbs_(alphabetically_by_proverb)',
         '#mw-content-text > div > ul > li', lambda x: x[:-5]),
        ('https://www.engvid.com/english-resource/50-common-proverbs-sayings/',
         'td[class="res_proverb_proverb"]', None),
        ('https://en.wikiquote.org/wiki/English_proverbs',
         '#mw-content-text > div > ul > li > i > b', None),
        ('https://lemongrad.com/proverbs-with-meanings-and-examples/',
         '#post-38199 > div.post-content > h3', None),
        ('https://lemongrad.com/idioms-with-meanings-and-examples/',
         '#post-39007 > div.post-content > h3', None),
        ('https://www.phrasemix.com/collections/the-50-most-important-english-proverbs',
         '#content > div.post > ol > li > h3', None),
        ('https://www.fluentu.com/blog/english/useful-english-proverbs/',
         '#main > div.post-2741.post.type-post.status-publish.format-standard.has-post-thumbnail.hentry.category-english-vocab-and-grammar.first-post > div.entry > h3',
         None),
        ('https://www.myenglishteacher.eu/blog/famous-proverbs-and-common-sayings/',
         'div.td-post-content.tagdiv-type > p > span > strong',
         None)
    ]

    proverbs = []
    headers = {'user-agent': random.choice(user_agent_list)}
    for url, selector, fn in SOURCES:
        r = requests.get(url, headers=headers)
        soup = bs4.BeautifulSoup(r.content, 'html.parser')
        source = parse.urlparse(r.url).netloc
        items = standard(soup, selector)
        if fn is not None:
            items = fn(items)
        if len(items):
            for item in items:
                proverb = clean_proverb(item)
                if len(proverb):
                    proverbs.append({'text': proverb, 'source': source})
        else:
            logging.debug('No items from {}'.format(source))
    df = pd.DataFrame(proverbs)
    filepath = ''.join([DATA_FOLDER, '/proverbs.csv'])
    logging.debug('saving to {}'.format(filepath))
    df.to_csv(filepath, index=False)


def clean_proverb(item):
    # remove useless characters
    text = re.sub(r'["]', '', item.get_text().lower().strip())
    # remove any numbered stuff
    text = re.sub(r'^\d+\.*\s', '', text)
    # split on full stop or newline
    result = re.split(r'[\.\n]', text)[0]
    return result


if __name__ == "__main__":
    main()
