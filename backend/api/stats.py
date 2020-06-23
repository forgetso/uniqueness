from flask import Flask, make_response
from flask_restful import reqparse, Api, Resource
from flask_cors import CORS
import json
from backend.db.mongo import connect
import logging
import re
from backend.settings import LOG_LEVEL
from backend.settings import AWS_USERNAME, AWS_SERVER
from backend.scrapers.merriam_webster import DATABASE as MERRIAM_WEBSTER_DB
from backend.scrapers.human_brain_cloud import DATABASE as HUMAN_BRAIN_DB
import subprocess
from backend.settings import PIHOLE_SERVER

app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'
cors = CORS(app, resources={r"*": {"origins": "*"}})
api = Api(app)

parser = reqparse.RequestParser()

logging.info('getting cues...')
CUES = ['blahb']
logging.info('done')
# logging.getLogger('flask_cors').level = logging.DEBUG
logging.basicConfig(level=LOG_LEVEL)
HEADERS = {'Access-Control-Allow-Origin': '*', 'Access-Control-Allow-Headers': '*'}


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
        cmd = ['python', '-m', 'backend.db.ssh', AWS_SERVER, AWS_USERNAME]
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
api.add_resource(Stats, '/stats')

if __name__ == '__main__':
    app.run(debug=False, host=PIHOLE_SERVER)
