from flask import Flask, make_response
from flask_restful import reqparse, abort, Api, Resource
from flask_cors import CORS
import random
import json
from backend.lib.db import get_cues, save_cue_response_pair, correct_cue_response_pair
import logging
from backend.lib.spelling import check_spelling
import re
from datetime import datetime as dt

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
parser.add_argument('correction')

print('getting cues...')
CUES = get_cues()
print('done')
#logging.getLogger('flask_cors').level = logging.DEBUG
logging.basicConfig(level=logging.DEBUG)
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

        # just need to make this an iterable for the next line to be ok
        corrections = check_spelling(cue_response_pair['response']) or set()
        #TODO restrict the number of spelling corrections returned to the page (perhaps phonetically?)
        corrections_list = list(corrections)
        cue_response_pair['corrections'] = corrections_list
        response_id = save_cue_response_pair(cue_response_pair)
        if len(corrections_list):
            response = output_json({'response_id': response_id, 'msg': 'ok', 'corrections': corrections_list}, 201,
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
            #logging.debug(msg='{},{}'.format(response_id, correction))
            if response_id and correction:
                correct_cue_response_pair(response_id, correction)
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
    cue_response_pair['response'] = re.sub('[^a-zA-Z]+', '', cue_response_pair['response'])
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
