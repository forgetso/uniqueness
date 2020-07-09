import requests
import bs4


def get_terms():
    r = requests.get('http://www.tysto.com/uk-us-spelling-list.html')
    soup = bs4.BeautifulSoup(r.content, 'html.parser')
    tables = soup.find_all('table')
    columns = tables[1].find_all('p')
    gb_terms = [x.strip() for x in columns[0].get_text().split('\n') if len(x.strip())]
    us_terms = [x.strip() for x in columns[1].get_text().split('\n') if len(x.strip())]
    assert (len(gb_terms) == len(us_terms))
    us2gb = dict(zip(us_terms, gb_terms))
    return us2gb
