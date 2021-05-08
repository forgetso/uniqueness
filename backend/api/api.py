from flask import Flask, make_response, request
from flask_restful import reqparse, Api, Resource
from flask_cors import CORS
import random
import json
import logging
import re
from backend.settings import LOG_LEVEL, DATA_FOLDER, GOOG_MERONYMS, SMALLWORLDOFWORDS
from backend.db.layer import get_word_filter, save_relations_initial, save_relations_response, \
    get_concretenesses
import spacy
import pandas as pd
from pathlib import Path
import numpy as np
from datetime import datetime
from backend.lib.measures import get_lemma, most_similar
import cupy

MODEL = 'attractrepel'
anlp = spacy.load(Path(DATA_FOLDER).joinpath(MODEL))
# to get spacy to work on the GPU we need to put this array on the GPU first
cupy_model = cupy.asarray(anlp.vocab.vectors.data.base, dtype=anlp.vocab.vectors.data.base.dtype)
en_words = [x for x in anlp.vocab.strings if x.startswith('en_')]
vectors = np.array([anlp(word).vector for word in en_words], dtype=anlp.vocab.vectors_dtype)

app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'
cors = CORS(app, resources={r"*": {"origins": "*"}})
api = Api(app)

parser = reqparse.RequestParser()
parser.add_argument('relationgroupid')
parser.add_argument('relationids')
parser.add_argument('fingerprint')
parser.add_argument('timedurations')
parser.add_argument('relationtype')

logging.info('getting cues...')
word_filter = pd.Series(get_word_filter(bands='all', pos='NoC')).str.decode('utf8')
word_filter = word_filter[~word_filter.isin(anlp.Defaults.stop_words)]
concretenesses = pd.DataFrame(pd.Series(get_concretenesses()), columns=['concreteness'])
concrete_words = concretenesses[concretenesses['concreteness'] > 4]
goog_meronyms = pd.read_csv(GOOG_MERONYMS)
smallworld = pd.read_csv(SMALLWORLDOFWORDS)

verb_adj = pd.Series(get_word_filter(bands='all', pos=['Verb', 'Adj']))
CUES = random.choice(word_filter.values)
logging.basicConfig(level=LOG_LEVEL)
HEADERS = {'Access-Control-Allow-Origin': '*', 'Access-Control-Allow-Headers': '*'}
RELATIONTYPES = ['meronym', 'hyponym']


class RelationQuery(Resource):
    def get(self):
        response = get_words()
        return output_json(response, 200)

    def post(self):
        response = None
        # try:
        args = request.get_json()
        print(args)
        cue_relations = {
            'relationgroupid': int(args['relationgroupid']),
            'relationids': [int(x) for x in args['relationids']],
            'fingerprint': re.sub('[^0-9a-zA-Z]+', '', str(args['fingerprint'])),
            'timedurations': [int(x) for x in args['timedurations']],
            'relationtype': re.sub('[^a-z]+', '', str(args['relationtype'])),
            'ipaddress': request.remote_addr,
            'locale': request.headers.get('Accept-Language'),
            'useragent': request.headers.get('User-Agent'),
            'source': 'ui'
        }
        print(cue_relations)
        assert (cue_relations['relationtype'] in RELATIONTYPES)
        # logging.debug(msg='{},{}'.format(response_id, correction))
        if cue_relations is not None:
            save_relations_response(**cue_relations)
            response = output_json({'msg': 'ok'}, 201, HEADERS)
        # except Exception as e:
        #     print(e)
        #     response = output_json({'msg': str(e)}, 500, HEADERS)
        return response


def get_random_cue():
    sample = None
    while not sample:

        potential = concrete_words.sample(n=1, random_state=datetime.utcnow().microsecond).index[0]
        lemma = get_lemma(potential)[0]
        if isinstance(lemma, str):
            if len(lemma) == len(potential) - 1:
                potential = lemma
        if potential.endswith('tion'):
            continue
        if potential.endswith('sion'):
            continue
        if potential.endswith('ies'):
            continue

        sample = potential
    return sample


def get_random_potential_meronyms(df, cola, colb):
    # get a dataframe in form cue word | potential meronym | similarity score
    # based on results from scraping google
    word = df.sample(n=1)[cola].values[0]
    df = df[df[cola] == word][[cola, colb]]
    df.rename({cola: 'cue', colb: 'relation'}, inplace=1, axis=1)
    # TODO config for prefixes for models
    df['score'] = df.apply(lambda x: anlp('en_' + x['cue']).similarity(anlp('en_' + x['relation'])), axis=1)
    df = remove_stopwords(df, 'cue')
    df = remove_stopwords(df, 'relation')
    df = df[~(df['relation'] == 'Unknown word')]
    df = df.drop_duplicates()
    return df


def get_most_similar(model, word, n):
    # result_keys, _, scores = model.vocab.vectors.most_similar(np.array([word.vector]), n=n)
    cupy_vector = cupy.asarray(np.array([word.vector]), dtype=anlp.vocab.vectors.data.base.dtype)
    result_keys, _, scores = most_similar(model=anlp, queries=cupy_vector, n=n, data=cupy_model)
    newstrs = [[model.vocab.strings.__getitem__(key) for key in keygroup] for keygroup in result_keys]
    df = pd.DataFrame(np.concatenate([np.array(newstrs), scores], axis=0)).T
    return df


def remove_stopwords(df, col):
    df = df[~df[col].isin(anlp.Defaults.stop_words)]
    return df


def output_json(data, code, headers=None):
    """Makes a Flask response with a JSON encoded body"""
    if isinstance(data, pd.DataFrame):
        jsondata = data.to_json(index=False, orient='table')
    else:
        jsondata = json.dumps(data)
    resp = make_response(jsondata, code)
    resp.headers.extend(headers or {})
    return resp


def get_words():
    grouplength = 5
    response = pd.DataFrame()
    while len(response) == 0:
        # TODO Config for the potential meronyms source
        potential_meronyms = get_random_potential_meronyms(smallworld, 'cue', 'R1')
        if len(potential_meronyms):
            cue = potential_meronyms['cue'].values[0]
            if len(potential_meronyms) < grouplength:
                additonal_pms = get_additional_potentials(cue)
                missing = grouplength - len(potential_meronyms)
                if len(additonal_pms) >= missing:
                    # TODO encrypt the scores
                    sample = additonal_pms.sample(n=missing, random_state=datetime.utcnow().microsecond)
                    final = pd.concat([potential_meronyms, sample])
                else:
                    final = potential_meronyms
            else:
                final = potential_meronyms[:grouplength]

            if len(final) == grouplength:
                print('saving relations')
                final['cue'] = cue
                final['modelname'] = MODEL
                print(final)
                response = save_relations_initial(final)
    return response


def get_additional_potentials(sample_word):
    # TODO tidy up into smaller functions and replace arbitrary numbers with variables
    pluralised = '%ss' % sample_word
    df = get_most_similar(anlp, anlp('en_' + sample_word), n=10000)
    df.rename({0: 'relation', 1: 'score'}, inplace=True, axis=1)
    index_plural = df[df['relation'] == pluralised].index
    if len(index_plural):
        df.drop(index_plural, inplace=True)
    df.drop(index_plural, inplace=True)
    df['relation'] = df['relation'].str.replace('en_', '')
    df['lemma'] = df['relation'].apply(lambda x: pd.Series(get_lemma(x)[0], dtype=str))
    df['lemma'].replace({None: ''}, inplace=True)
    df['lemma'] = df['lemma'].astype(str)
    df['relation'] = df.apply(
        lambda x: x['lemma'] if len(x['relation']) >= len(x['lemma']) - 1 else x['relation'],
        axis=1)
    df = df.drop(df[df['relation'] == sample_word].index)
    df = df.drop(df[df['relation'] == ''].index)
    df['score'] = df['score'].astype(float)
    df = df[~df['relation'].isin(verb_adj)]
    df = remove_stopwords(df, 'relation')
    df = df[df['relation'].isin(word_filter)]
    df = df[df['score'] <= 0.8]
    df.drop(columns=['lemma'], inplace=True)
    df = df.groupby('relation').agg({'score': ['mean']})
    df.columns = df.columns.droplevel(1)
    df.reset_index(inplace=True)
    df = df.sort_values(by='score', ascending=False)
    df['cue'] = sample_word
    return df


##
## Actually setup the Api resource routing here
##
api.add_resource(RelationQuery, '/relationquery')

if __name__ == '__main__':
    print('in main thread')
    app.run(debug=False, host='192.168.1.28')
