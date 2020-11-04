from flask import Flask, make_response
from flask_restful import reqparse, Api, Resource
from flask_cors import CORS
import random
import json
from backend.db.layer import get_cues, save_cue_response_pair, update_correction
import logging
import re
from datetime import datetime as dt
from backend.lib.helpers.string import check_spelling
from backend.settings import LOG_LEVEL

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



##
## Actually setup the Api resource routing here
##
api.add_resource(CueResponse, '/cueresponse')
api.add_resource(CorrectSpelling, '/correctspelling')

if __name__ == '__main__':
    app.run(debug=True)
