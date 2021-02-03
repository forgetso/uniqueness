from flask import Flask, make_response
from flask_restful import reqparse, Api, Resource
from flask_cors import CORS
import random
import json
import logging
import re
from backend.settings import LOG_LEVEL, DATA_FOLDER
from backend.db.layer import get_word_filter, get_meronyms, save_cue_relations, get_concretenesses
import spacy
import pandas as pd
from pathlib import Path
import numpy as np
from datetime import datetime
from backend.lib.measures import get_lemma, most_similar
import cupy

anlp = spacy.load(Path(DATA_FOLDER).joinpath('attractrepel'))
# to get spacy to work on the GPU we need to put this array on the GPU first
cupy_model = cupy.asarray(anlp.vocab.vectors.data.base, dtype=anlp.vocab.vectors.data.base.dtype)
en_words = [x for x in anlp.vocab.strings if x.startswith('en_')]
vectors = np.array([anlp(word).vector for word in en_words], dtype=anlp.vocab.vectors_dtype)
# cupy_vectors = cupy.asarray(vectors, dtype=anlp.vocab.vectors.data.base.dtype)

app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'
cors = CORS(app, resources={r"*": {"origins": "*"}})
api = Api(app)

# def abort_if_todo_doesnt_exist(todo_id):
#     if todo_id not in TODOS:
#         abort(404, message="Todo {} doesn't exist".format(todo_id))

parser = reqparse.RequestParser()
parser.add_argument('relations')
parser.add_argument('cue')
parser.add_argument('relation_type')
parser.add_argument('time_duration')
parser.add_argument('fingerprint')
parser.add_argument('response_id')

logging.info('getting cues...')
word_filter = pd.Series(get_word_filter(bands='all', pos='NoC')).str.decode('utf8')
word_filter = word_filter[~word_filter.isin(anlp.Defaults.stop_words)]
concretenesses = pd.DataFrame(pd.Series(get_concretenesses()), columns=['concreteness'])
concrete_words = concretenesses[concretenesses['concreteness'] > 4]
verb_adj = pd.Series(get_word_filter(bands='all', pos=['Verb', 'Adj']))
CUES = random.choice(word_filter.values)
logging.basicConfig(level=LOG_LEVEL)
HEADERS = {'Access-Control-Allow-Origin': '*', 'Access-Control-Allow-Headers': '*'}


class RelationQuery(Resource):
    def get(self):
        response = None
        # TODO if word ends in ..ing should the examples also end in ing, ation, etc.
        while response is None:

            # lemmas = getAllLemmas(sample_word)
            # if 'NOUN' in lemmas:
            # sample_word = lemmas['NOUN'][0]
            # TODO remove words with same lemma
            sample_word = get_random_cue()
            pluralised = '%ss' % sample_word
            df = get_most_similar(anlp, anlp('en_' + sample_word), n=10000)
            df.rename({0: 'word', 1: 'sim1'}, inplace=True, axis=1)
            index_plural = df[df['word'] == pluralised].index
            if len(index_plural):
                df.drop(index_plural, inplace=True)
            df.drop(index_plural, inplace=True)
            df['word'] = df['word'].str.replace('en_', '')
            df['lemma'] = df['word'].apply(lambda x: pd.Series(get_lemma(x)[0]))
            df['lemma'].replace({None: ''}, inplace=True)
            df['lemma'] = df['lemma'].astype(str)
            df['word'] = df.apply(
                lambda x: x['lemma'] if len(x['word']) >= len(x['lemma']) - 1 else x['word'],
                axis=1)
            df = df.drop(df[df['word'] == sample_word].index)
            df = df.drop(df[df['word'] == ''].index)
            df['sim1'] = df['sim1'].astype(float)
            df = df[~df['word'].isin(verb_adj)]
            df = df[~df['word'].isin(anlp.Defaults.stop_words)]
            df = df[df['word'].isin(word_filter)]
            df = df[df['sim1'] <= 0.8]
            df.drop(columns=['lemma'], inplace=True)
            df = df.groupby('word').agg({'sim1': ['mean']})
            df.columns = df.columns.droplevel(1)
            df.reset_index(inplace=True)
            df = df.sort_values(by='sim1', ascending=False)
            if len(df) >= 5:
                # TODO encrypt the scores
                sam = df.sample(n=5, random_state=datetime.utcnow().microsecond)
                id = save_cue_relations(cue=sample_word, relations=response['relations'])
                response = {
                    'word': sample_word,
                    'relations': list(
                        zip(sam['word'].values.tolist(), sam['sim1'].values.tolist()))
                }

        return output_json(response, 200)

    def post(self):
        response = None
        try:
            args = parser.parse_args()
            cue_relations = {'cue': args['cue'],
                             'relation_type': args['relation_type'],
                             'relations': json.loads(args['relations'].replace("'", '"')),
                             'created': str(datetime.utcnow()),
                             'time_duration': args['time_duration'],
                             'fingerprint': args['fingerprint'], }
            print(cue_relations)
            cue_relations = sanitise_response(cue_relations)
            print(cue_relations)
            # logging.debug(msg='{},{}'.format(response_id, correction))
            if cue_relations:
                save_cue_relations(**cue_relations)
                response = output_json({'msg': 'ok'}, 201, HEADERS)
        except Exception as e:
            response = output_json({'msg': e}, 500, HEADERS)
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


def get_most_similar(model, word, n):
    # result_keys, _, scores = model.vocab.vectors.most_similar(np.array([word.vector]), n=n)
    cupy_vector = cupy.asarray(np.array([word.vector]), dtype=anlp.vocab.vectors.data.base.dtype)
    result_keys, _, scores = most_similar(model=anlp, queries=cupy_vector, n=n, data=cupy_model)
    newstrs = [[model.vocab.strings.__getitem__(key) for key in keygroup] for keygroup in result_keys]
    df = pd.DataFrame(np.concatenate([np.array(newstrs), scores], axis=0)).T
    return df


def output_json(data, code, headers=None):
    """Makes a Flask response with a JSON encoded body"""
    resp = make_response(json.dumps(data), code)
    resp.headers.extend(headers or {})
    return resp


def sanitise_response(cue_relations):
    sanitised_cue_relations = {}
    sanitised_cue_relations['relations'] = {re.sub('[^a-zA-Z\s]+', '', word): float(score) for word, score in
                                            cue_relations['relations'].items()}
    sanitised_cue_relations['cue'] = re.sub('[^a-zA-Z]+', '', cue_relations['cue'])
    sanitised_cue_relations['relation_type'] = re.sub('[^a-zA-Z]+', '', cue_relations['relation_type'])
    sanitised_cue_relations['time_duration'] = int(re.sub('[^0-9]+', '', cue_relations['time_duration']))
    sanitised_cue_relations['fingerprint'] = re.sub('[^0-9a-zA-Z]+', '', cue_relations['fingerprint'])
    sanitised_cue_relations['created'] = cue_relations['created']
    return sanitised_cue_relations


##
## Actually setup the Api resource routing here
##
api.add_resource(RelationQuery, '/relationquery')

if __name__ == '__main__':
    app.run(debug=True)
