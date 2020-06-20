from flask import Flask, make_response
from flask_restful import reqparse, Api, Resource
from flask_cors import CORS
import random
import json
from backend.db.db import get_cues, save_cue_response_pair, update_correction
from backend.db.mongo import connect
import logging
import re
from datetime import datetime as dt
from backend.lib.helpers.string import check_spelling
from backend.settings import LOG_LEVEL
from backend.settings import AWS_USERNAME, AWS_SERVER
from backend.scrapers.merriam_webster import DATABASE as MERRIAM_WEBSTER_DB
from backend.scrapers.human_brain_cloud import DATABASE as HUMAN_BRAIN_DB
import subprocess
import bson

app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'
cors = CORS(app, resources={r"*": {"origins": "*"}})
api = Api(app)

# def abort_if_todo_doesnt_exist(todo_id):
#     if todo_id not in TODOS:
#         abort(404, message="Todo {} doesn't exist".format(todo_id))

parser = reqparse.RequestParser()
parser.add_argument('response')
parser.add_argument('cue')
parser.add_argument('time_duration')
parser.add_argument('user_id')
parser.add_argument('response_id')
parser.add_argument('original_word')
parser.add_argument('correction')

logging.info('getting cues...')
CUES = ['blahb']
logging.info('done')
# logging.getLogger('flask_cors').level = logging.DEBUG
logging.basicConfig(level=LOG_LEVEL)
HEADERS = {'Access-Control-Allow-Origin': '*', 'Access-Control-Allow-Headers': '*'}


# CueResponse
# returns a single cue or collects a response
class CueResponse(Resource):
    def get(self):
        # abort_if_todo_doesnt_exist(todo_id)
        data = {'cue': random.choice(CUES)}
        response = output_json(data, 200, HEADERS)
        return response

    def post(self):
        args = parser.parse_args()
        cue_response_pair = sanitise_response(
            {'response': args['response'],
             'created': dt.utcnow(),
             'cue': args['cue'],
             'time_duration': args['time_duration'],
             'user_id': args['user_id'],
             'corrections': []}
        )
        cue = cue_response_pair['cue'].replace(' ', '_')
        response = cue_response_pair['response'].replace(' ', '_')
        incorrect_spelling, corrections_list, corrections_dict = check_spelling(cue, response)
        cue_response_pair['corrections'] = corrections_dict
        cue_response_pair['source'] = 'ui'
        response_id = save_cue_response_pair(cue_response_pair)
        if len(corrections_list):
            response = output_json({'response_id': response_id, 'msg': 'ok', 'corrections': corrections_dict}, 201,
                                   HEADERS)
        else:
            response = output_json({'response_id': response_id, 'msg': 'ok'}, 201, HEADERS)
        return response


class CorrectSpelling(Resource):
    def post(self):
        response = None
        try:
            args = parser.parse_args()
            response_id = args.get('response_id')
            correction = args.get('correction')
            original_word = args.get('original_word')
            # logging.debug(msg='{},{}'.format(response_id, correction))
            if response_id and correction:
                update_correction(response_id, original_word, correction)
                response = output_json({'msg': 'ok'}, 201, HEADERS)
        except Exception as e:
            response = output_json({'msg': e}, 500, HEADERS)
        return response


class Stats(Resource):

    def __init__(self):
        self.sshtunnel_port = get_ssh_server()

    def get(self):
        mw_data = self.merriam_webster_data()
        hb_data = self.human_brain_cloud_data()
        result = [mw_data, hb_data]
        return output_json(code=200, data=result)

    def merriam_webster_data(self):
        # this data is stored on an AWS instance
        last_word = None
        db = connect(database=MERRIAM_WEBSTER_DB, port=self.sshtunnel_port)
        cur = db['words'].find({})
        mw_data = [removeObjectId(x) for x in cur]
        count = 0
        for m in mw_data:
            if m['sources'].get('merriam_webster'):
                count += 1
                last_word = m.get('word')

        return {'source': 'merriam_webster', 'count': count, 'last_word': last_word}

    def human_brain_cloud_data(self):
        # this stuff is stored locally
        db = connect(database=HUMAN_BRAIN_DB)
        cur = db['data'].find({})
        hb_data = [removeObjectId(x) for x in cur]
        count = len(hb_data)
        return {'source': 'humanbraincloud', 'count': count}


def output_json(data, code, headers=None):
    """Makes a Flask response with a JSON encoded body"""
    resp = make_response(json.dumps(data), code)
    resp.headers.extend(headers or {})
    return resp


def sanitise_response(cue_response_pair):
    cue_response_pair['response'] = re.sub('[^a-zA-Z\s]+', '', cue_response_pair['response'])
    cue_response_pair['cue'] = re.sub('[^a-zA-Z]+', '', cue_response_pair['cue'])
    cue_response_pair['time_duration'] = int(re.sub('[^0-9]+', '', cue_response_pair['time_duration']))
    return cue_response_pair


def get_ssh_server():
    db = connect(database='uniqueness')
    query = {'sshtunnel.server': AWS_SERVER, 'sshtunnel.username': AWS_USERNAME}
    doc = [x for x in db['config'].find(query)]

    if not len(doc):
        cmd = ['python', '-m', 'backend.db.sshtunnel', AWS_SERVER, AWS_USERNAME]
        sshtunnel = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        query = {'sshtunnel': {'server': AWS_SERVER, 'username': AWS_USERNAME, 'pid': sshtunnel.pid}}
        doc = [x for x in db['config'].find(query)]

    doc = doc[0]
    sshtunnel_port = doc['sshtunnel']['local_port']
    return sshtunnel_port


def removeObjectId(doc):
    doc.pop('_id')
    return doc


##
## Actually setup the Api resource routing here
##
api.add_resource(CueResponse, '/cueresponse')
api.add_resource(CorrectSpelling, '/correctspelling')
api.add_resource(Stats, '/stats')

if __name__ == '__main__':
    app.run(debug=True)
