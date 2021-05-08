import regex as re
import html
import argparse
from backend.lib.scrapers.words import *
import logging
from backend.settings import LOG_LEVEL
from backend.lib.multiprocessor import job_pool
from time import sleep
import chardet
import numpy as np
from urllib.error import HTTPError

logging.basicConfig(level=LOG_LEVEL)
logging.getLogger("chardet").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

BASE_URL = "https://google.com"
OUTPUTFILE = 'search_results.csv'
JOB_NUMBER = 1
USER_AGENT = "Mozilla/5.0 (Linux; Android 11) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.141 Mobile Safari/537.36"


def process(**kwargs):
    words_to_scrape = []
    words = kwargs.get('words')
    input_file = kwargs.get('input_file')
    output_file = kwargs.get('output_file')
    filter = kwargs.get('filter')

    if words is not None:
        words_to_scrape = pd.Series(words.split(','))
    elif input_file is not None:
        words_to_scrape = load_words_from_file(input_file, filter=filter)

    if output_file is None:
        output_file = OUTPUTFILE

    existing_data = get_progress(output_file)
    words_to_scrape = remove_downloaded_words_from_list(existing_data, words_to_scrape, col='word')
    jobs = job_pool(data=words_to_scrape, job_number=JOB_NUMBER,
                    job_to_do=iterate_words, user_agent=USER_AGENT,
                    base_url=BASE_URL, output_file=output_file)
    df = pd.concat([job[0] for job in jobs])
    errors = [job[1] for job in jobs]

    logging.info('Saving progress to {}'.format(output_file))
    save_data_to_file(output_file, existing_data, df)

    if len(errors):
        distinct_errors = list(set(errors))
        raise ConnectionError(distinct_errors[0].errno, distinct_errors[0].strerror)

    return df


def set_headers(user_agent):
    headers = {
        "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
        "accept-encoding": "identity",
        "accept-language": "en-US;q=0.9,en;q=0.8",
        "cache-control": "no-cache",
        "pragma": "no-cache",
        "referer": "https://www.google.com/",
        "sec-fetch-dest": "document",
        "sec-fetch-mode": "navigate",
        "sec-fetch-site": "same-origin",
        "sec-fetch-user": "?1",
        "sec-gpc": "1",
        "upgrade-insecure-requests": "1",
        "user-agent": user_agent,
    }
    headers = {k: str(v) for k, v in headers.items()}
    return headers


def iterate_words(words_to_scrape, user_agent, base_url, output_file):
    data = []
    error = None
    if len(words_to_scrape):
        cookies = {'CONSENT': 'YES+US.en-GB+20160412-16-0'}
        session, initial_request = setup_session(user_agent, base_url, headers=set_headers(user_agent), cookies=cookies)
        gs_lcp, ei = get_session_params(initial_request)
        rng = np.random.default_rng()
        words = words_to_scrape.values
        rng.shuffle(words)
        existing_data = get_progress(output_file)
        try:
            for word in words:
                print(word)
                # we get too many requests without waiting
                sleep(30)
                content_list = download(session, word, gs_lcp, ei)
                df = pd.DataFrame(content_list, columns=['word', 'result'])
                save_data_to_file(output_file, existing_data, df)
                data.extend(content_list)
        except (KeyboardInterrupt, ConnectionError) as e:
            error = e
            df = pd.DataFrame(data, columns=['word', 'result'])
            return df, error
        finally:
            df = pd.DataFrame(data, columns=['word', 'result'])
            return df, error


def get_session_params(session):
    encoding = chardet.detect(session.content)
    content = session.content.decode(encoding['encoding'])
    match = re.search('gs_lcp%3D[A-Za-z0-9]+', content)
    gs_lcp = ''
    ei = ''
    if match:
        gs_lcp = match[0]
    match = re.search('ei%3D[A-Za-z0-9]+', content)
    if match:
        ei = match[0]
    return gs_lcp, ei


def download(session, word, gs_lcp, ei):
    logging.debug('Getting search results for {}'.format(word))

    url = get_url(word, gs_lcp, ei)
    request_obj = session.get(url)
    r = request_obj

    encoding = chardet.detect(r.content)

    logging.debug(f'Status code {request_obj.status_code}')
    if request_obj.status_code not in (200, 302, 301):
        logging.warning(f'Cannot access URL {url}: {request_obj.status_code}')
        raise ConnectionError(request_obj.status_code, 'Cannot access URL {url}')

    content = html.unescape(request_obj.content.decode(encoding['encoding'])).replace('\\x3e', '>')
    content = content.replace('\\x3c', '<')
    content = content.replace('\\x3d', ':')
    content = content.replace('\\x22', "'")
    pattern = re.compile(''.join(['(', word, '(\s\w+){5,9}', ')']), flags=re.IGNORECASE + re.DOTALL)
    instances = [(word, x[0]) for x in re.findall(pattern, content)]
    logging.debug(instances)
    return instances


def get_url(word, ei, gs_lcp):
    query = f'"{word}+is+a+part+of"'
    if len(gs_lcp) and len(ei):
        url = f"https://google.com/search?source=hp&ei={ei}&q={query}&oq={query}&gs_lcp={gs_lcp}&sclient=mobile-gws-wiz-hp"
    else:
        url = f"https://google.com/search?source=hp&q={query}&oq={query}&sclient=mobile-gws-wiz-hp"
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
