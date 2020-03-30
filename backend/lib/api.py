from flask import Flask, make_response
from flask_restful import reqparse, abort, Api, Resource
from flask_cors import CORS
import random
import json
from backend.lib.db import get_cue_response_pairs
import logging

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

CUE_RESPONSE_PAIRS = [x for x in get_cue_response_pairs()]
print(CUE_RESPONSE_PAIRS[0])
CUES = [x['cue'] for x in CUE_RESPONSE_PAIRS]
logging.getLogger('flask_cors').level = logging.DEBUG
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
        print(args)
        cue_response_pair = {'response': args['response'], 'cue': args['cue'], 'time_duration': args['time_duration']}
        print(cue_response_pair)
        response = output_json(201, HEADERS)
        return response


def output_json(data, code, headers=None):
    """Makes a Flask response with a JSON encoded body"""
    resp = make_response(json.dumps(data), code)
    resp.headers.extend(headers or {})
    return resp


##
## Actually setup the Api resource routing here
##
api.add_resource(CueResponse, '/cueresponse')

if __name__ == '__main__':
    app.run(debug=True)
