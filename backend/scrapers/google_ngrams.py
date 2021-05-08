import random
from backend.settings import LOG_LEVEL
from backend.scrapers.user_agents import user_agent_list
from backend.lib.scrapers.util import extract_json_from_script
from backend.lib.multiprocessor import job_pool
from bs4 import BeautifulSoup
import argparse
from backend.lib.scrapers.words import *
import logging
from time import sleep

logging.basicConfig(level=LOG_LEVEL)
BASE_URL = 'https://books.google.com'
USER_AGENT = random.choice(user_agent_list)
OUTPUTFILE = 'ngrams.csv'
JOB_NUMBER = 4


def process(**kwargs):
    words_to_scrape = []
    words = kwargs.get('words')
    input_file = kwargs.get('input_file')
    output_file = kwargs.get('output_file')
    if words is not None:
        words_to_scrape = pd.Series(words.split(','))
    elif input_file is not None:
        words_to_scrape = load_words_from_file(input_file)

    if output_file is None:
        output_file = OUTPUTFILE

    existing_data = get_progress(output_file)
    words_to_scrape = remove_downloaded_words_from_list(existing_data, words_to_scrape, col='ngram')
    jobs = job_pool(data=words_to_scrape, job_number=JOB_NUMBER,
                    job_to_do=iterate_words, user_agent=USER_AGENT,
                    base_url=BASE_URL)
    df = pd.concat(jobs)

    logging.info('Saving progress to {}'.format(output_file))
    save_data_to_file(output_file, existing_data, df)

    logging.debug('Got data:')
    if len(df) < len(words_to_scrape):
        logging.error(
            'Less words returned ({}) than requested ({}), job must have been cut short or failed!'.format(len(df), len(
                words_to_scrape)))

    return df


def iterate_words(words_to_scrape, user_agent, base_url):
    data = []
    if len(words_to_scrape):
        session, _initial_request = setup_session(user_agent, base_url)
        try:
            for words in batch(words_to_scrape.values, batch_size=12):
                # we get too many requests without waiting
                sleep(1)
                content_dict = download(session, words)
                data.extend(content_dict)
        except (Exception, KeyboardInterrupt) as e:
            logging.error('\nDownload failed: {}'.format(e))
            df = pd.DataFrame(data)
            return df

    df = pd.DataFrame(data)
    return df


def download(session, words):
    logging.debug('Getting ngrams for {}'.format(words))
    url = get_url(words)
    request_obj = session.get(url)
    if request_obj.status_code != 200:
        logging.debug(request_obj.headers)
        raise ConnectionError(
            'Cannot access {} \nServer returned {} - {}'.format(url, request_obj.status_code,
                                                                request_obj.content))

    soup = BeautifulSoup(request_obj.content, 'html.parser')
    all_scripts = soup.find_all('script')
    content_dict = extract_json_from_script(all_scripts, start_split='ngrams.data = ', end_split=';\n')
    return content_dict


def get_url(words):
    # https://books.google.com/ngrams/graph?
    # content=children%2Ccot%2Ccute%2Ccrying%2Ccry%2Cchild%2Cdiapers%2Cthoughtless%2Csweet%2Csmiles%2Csmall%2Cwet%2Cyucks%2Cshit%2Clife%2Cinfants%2Cinfant%2Cfamily%2Cmom%2C&
    # year_start=1800&year_end=2019&corpus=26&smoothing=3&
    # direct_url=t1%3B%2Cchildren%3B%2Cc0%3B.t1%3B%2Ccot%3B%2Cc0%3B.t1%3B%2Ccute%3B%2Cc0%3B.t1%3B%2Ccrying%3B%2Cc0%3B.t1%3B%2Ccry%3B%2Cc0%3B.t1%3B%2Cchild%3B%2Cc0%3B.t1%3B%2Cdiapers%3B%2Cc0%3B.t1%3B%2Cthoughtless%3B%2Cc0%3B.t1%3B%2Csweet%3B%2Cc0%3B.t1%3B%2Csmiles%3B%2Cc0%3B.t1%3B%2Csmall%3B%2Cc0%3B.t1%3B%2Cwet%3B%2Cc0
    content = words[0]
    direct_url = ''
    for word in words[1:]:
        content += '%2C{}%2C'.format(word.strip().replace(' ', '+'))
    content = content[:-3]

    for word in words:
        direct_url += 't1%3B%2C{word}%3B%2Cc0%3B.'.format(word=word)
    direct_url = direct_url[:-3]
    url = '{base_url}/ngrams/graph?content={content}&year_start=1800&year_end=2019&corpus=26&smoothing=3&direct_url={direct_url}'.format(
        base_url=BASE_URL, content=content, direct_url=direct_url)
    return url


def setup():
    parser = argparse.ArgumentParser(description='Download ngram word data')
    parser.add_argument('--input_file', default=None,
                        help='json or csv file containing words in first column or words  as keys')
    parser.add_argument('--words', default=None, help='comma separated list of words')
    parser.add_argument('--output_file', default=None, help='name of file to store ngram data')
    args = parser.parse_args()
    process(words=args.words, input_file=args.input_file, output_file=args.output_file)


if __name__ == "__main__":
    setup()
