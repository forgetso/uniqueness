import requests
import bs4
import random
from backend.scrapers.user_agents import user_agent_list
import re
from urllib import parse
import pandas as pd
from backend.settings import DATA_FOLDER, LOG_LEVEL
import logging
from urllib3.exceptions import ProtocolError, ConnectionError

logging.basicConfig(level=LOG_LEVEL)

PROVERB_MAX_LENGTH = 20
PROVERB_MIN_LENTH = 2
SOURCES = [
    # Config for a scraper is:
    # URL, SELECTOR, CUSTOM ITEM FILTER FUNCTION, CUSTOM TEXT SPLITTER TUPLE, NEXT_PAGE_SELECTOR_LIST
    ('https://www.phrases.org.uk/meanings/proverbs.html',
     'p[class="phrase-list"]', None, None, None),
    ('https://en.wikiquote.org/wiki/English_proverbs_(alphabetically_by_proverb)',
     '#mw-content-text > div > ul > li', lambda x: x[:-5], None, None),
    ('https://www.engvid.com/english-resource/50-common-proverbs-sayings/',
     'td[class="res_proverb_proverb"]', None, None, None),
    ('https://en.wikiquote.org/wiki/English_proverbs',
     '#mw-content-text > div > ul > li > i > b', None, None, None),
    ('https://lemongrad.com/proverbs-with-meanings-and-examples/',
     '#post-38199 > div.post-content > h3', None, None, None),
    ('https://lemongrad.com/idioms-with-meanings-and-examples/',
     '#post-39007 > div.post-content > h3', None, None, None),
    ('https://www.phrasemix.com/collections/the-50-most-important-english-proverbs',
     '#content > div.post > ol > li > h3', None, None, None),
    ('https://www.fluentu.com/blog/english/useful-english-proverbs/',
     '#main > div.post-2741.post.type-post.status-publish.format-standard.has-post-thumbnail.hentry.category-english-vocab-and-grammar.first-post > div.entry > h3',
     None, None, None),
    ('https://www.myenglishteacher.eu/blog/famous-proverbs-and-common-sayings/',
     'div.td-post-content > p > span > strong',
     None, None, None),
    ('http://sherwoodschool.ru/en/lessons1/english/english_proverbs_and_sayings/',
     'body > div.wrp-main > section > section > article > div > div.main-content > p', None,
     ('"&nbsp;"', 'meaning:'), None),
    ('http://www.inglesvillalba.com/english-proverbs-their-meanings-and-origins/',
     '#categoriaDescripcion > div.col-sm-9 > p > span > strong', None, None, None),
    ('https://preply.com/en/blog/2020/05/24/learn-30-of-the-most-popular-proverbs-in-english/',
     '#single__article > section.opacity__appearing.single__article__main-new > article > h3 > b', None, None,
     None),
    ('https://www.toppr.com/guides/business-communication-and-ethics/essential-of-good-english/proverbs/',
     '#post-66648 > div > p > em > strong', None, None, None),
    ('https://selffa.com/english-proverbs-with-meanings/',
     '#post-3376 > div > div.entry-content > h3', None, None, None),
    ('https://eltlearningjourneys.com/2016/05/10/35-common-english-proverbs/',
     '#post-2213 > div > p > strong', None, None, None),
    ('https://www.learn-english-today.com/proverbs/proverbs-A1.html', '#proverbs > ul > li > span', None, None,
     ('#content0 > p.ctr > a', '#content0 > p.ctr > strong > a')),
    ('https://www.learn-english-today.com/proverbs/proverbs-T2.html', '#proverbs > ul > li > span', None, None,
     ('#content0 > p.ctr > a', '#content0 > p.ctr > strong > a')),
    ('https://proverbicals.com/english-proverbs', '#post-262 > div.entry-content > ul > li', None, None, None),
    ('https://www.englishclub.com/ref/Sayings/A', '#ec-main > div > h3 > a', None, None, None),
    ('https://www.englishclub.com/ref/Sayings/B', '#ec-main > div > h3 > a', None, None, None),
    ('https://www.englishclub.com/ref/Sayings/C', '#ec-main > div > h3 > a', None, None, None),
    ('https://www.englishclub.com/ref/Sayings/D', '#ec-main > div > h3 > a', None, None, None),
    ('https://www.englishclub.com/ref/Sayings/E', '#ec-main > div > h3 > a', None, None, None),
    ('https://www.englishclub.com/ref/Sayings/F', '#ec-main > div > h3 > a', None, None, None),
    ('https://www.englishclub.com/ref/Sayings/G', '#ec-main > div > h3 > a', None, None, None),
    ('https://www.englishclub.com/ref/Sayings/H', '#ec-main > div > h3 > a', None, None, None),
    ('https://www.englishclub.com/ref/Sayings/I', '#ec-main > div > h3 > a', None, None, None),
    ('https://www.englishclub.com/ref/Sayings/J', '#ec-main > div > h3 > a', None, None, None),
    ('https://www.englishclub.com/ref/Sayings/K', '#ec-main > div > h3 > a', None, None, None),
    ('https://www.englishclub.com/ref/Sayings/L', '#ec-main > div > h3 > a', None, None, None),
    ('https://www.englishclub.com/ref/Sayings/M', '#ec-main > div > h3 > a', None, None, None),
    ('https://www.englishclub.com/ref/Sayings/N', '#ec-main > div > h3 > a', None, None, None),
    ('https://www.englishclub.com/ref/Sayings/O', '#ec-main > div > h3 > a', None, None, None),
    ('https://www.englishclub.com/ref/Sayings/P', '#ec-main > div > h3 > a', None, None, None),
    ('https://www.englishclub.com/ref/Sayings/Q', '#ec-main > div > h3 > a', None, None, None),
    ('https://www.englishclub.com/ref/Sayings/R', '#ec-main > div > h3 > a', None, None, None),
    ('https://www.englishclub.com/ref/Sayings/S', '#ec-main > div > h3 > a', None, None, None),
    ('https://www.englishclub.com/ref/Sayings/T', '#ec-main > div > h3 > a', None, None, None),
    ('https://www.englishclub.com/ref/Sayings/U', '#ec-main > div > h3 > a', None, None, None),
    ('https://www.englishclub.com/ref/Sayings/V', '#ec-main > div > h3 > a', None, None, None),
    ('https://www.englishclub.com/ref/Sayings/W', '#ec-main > div > h3 > a', None, None, None),
    ('https://www.englishclub.com/ref/Sayings/XYZ', '#ec-main > div > h3 > a', None, None, None),
]


def standard(soup, selector):
    items = soup.select(selector)
    return items


# proverbs and idioms scraped from the net
# TODO multiprocess scraper
def main():
    all_proverbs = []
    scraped = set()
    for url, selector, fn, split, next_page in SOURCES:

        while url is not None and url not in scraped:
            url_object = parse.urlparse(url)
            proverbs, soup = get_proverbs(url, selector, fn, split)
            if not len(proverbs):
                logging.debug('No items from {}'.format(url_object.netloc))
            else:
                all_proverbs.extend(proverbs)
            scraped.add(url)
            if next_page is not None:
                try:
                    url = get_next_page(soup, next_page, url_object)
                except (KeyError, IndexError, ProtocolError, ConnectionError):
                    url = None
            else:
                break

    df = pd.DataFrame(all_proverbs)
    filepath = ''.join([DATA_FOLDER, '/proverbs.csv'])
    logging.debug('saving to {}'.format(filepath))
    df.to_csv(filepath, index=False)


def get_next_page(soup, next_page, url_obj):
    url = None
    for nxt in next_page:
        path = ''.join(url_obj.path.split('/')[:-1])
        links = soup.select(nxt)
        if not links:
            continue
        next_link = links[len(links) - 1]['href']
        url = ''.join([url_obj.scheme, '://', url_obj.netloc, '/', path, '/', next_link])
    return url


def get_proverbs(url, selector, fn, split):
    proverbs = []
    headers = {'user-agent': random.choice(user_agent_list)}
    r = requests.get(url, headers=headers)
    soup = bs4.BeautifulSoup(r.content, 'html.parser')
    source = parse.urlparse(r.url).netloc
    items = standard(soup, selector)
    if fn is not None:
        items = fn(items)
    if len(items):
        for item in items:
            proverb = clean_proverb(item, custom_split=split)
            if len(proverb):
                p_split = proverb.split(' ')
                if PROVERB_MIN_LENTH <= len(p_split) < PROVERB_MAX_LENGTH:
                    proverbs.append({'text': proverb, 'source': source})
    return proverbs, soup


def clean_proverb(item, custom_split=None):
    text = item.get_text().lower().strip()
    # get rid of stuff with split
    if custom_split:
        for s in custom_split:
            text = text.split(s)[0].strip()
    # replace weird apostrophe
    text = text.replace('’', "'")
    # remove useless characters
    text = re.sub(r'["“]', '', text)
    # remove any numbered stuff
    text = re.sub(r'^\d+\.*\s', '', text)
    # split on full stop or newline
    result = re.split(r'[\.\n]', text)[0].strip()
    return result


def remove_boundary_chars(text):
    boundary_chars = ("'",)
    if len(text):
        # remove single quotes from start or finish
        if text[0] in boundary_chars:
            text = text[1:]
        if text[:len(text) - 1] in boundary_chars:
            text = text[:-1]
    return text


def find_urls(soup):
    soup.select('#content0 > p.ctr > a > strong')
    urls = []


if __name__ == "__main__":
    main()
