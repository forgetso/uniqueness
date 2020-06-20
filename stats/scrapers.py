import sys, os
# os.path.abspath(os.path.join('..', ''))
from backend.settings import PIHOLE_SERVER, AWS_SERVER, PIHOLE_USERNAME, AWS_USERNAME
from backend.scrapers.merriam_webster import DATABASE as merriam_db
from backend.scrapers.human_brain_cloud import DATABASE as humanbrain_db
from backend.importers.common import make_external_backup, get_external_backup
from pathlib import Path
from sshtunnel import SSHTunnelForwarder
import pymongo
from http.server import BaseHTTPRequestHandler, HTTPServer
import logging

PORT = 7338




class ExternalMongoSSHServer:
    def __init__(self, server, username):
        home = str(Path.home())
        pkey = '{}/.ssh/id_rsa'.format(home)
        self.server = SSHTunnelForwarder(
            server,
            ssh_username=username,
            ssh_password=pkey,
            remote_bind_address=('127.0.0.1', 27017)
        )
        # start ssh tunnel
        self.server.start()

    def stop(self):
        self.server.stop()

