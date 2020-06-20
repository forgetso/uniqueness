from pathlib import Path
from sshtunnel import SSHTunnelForwarder
import argparse
import logging
import time
from backend.db.mongo import connect
import os


class SSHTunnel:
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


def main(server, username):
    config_id = None
    tunnel = None
    try:
        tunnel = SSHTunnel(server=server, username=username)
        # this is the LOCAL HOST port that has been bound to the ssh tunnel and is needed to connect to mongo
        port = tunnel.server.local_bind_port
        config_id = update_config(server, username, port)
        while True:
            time.sleep(1)
    finally:
        logging.debug('Shutting down ssh server')
        remove_config(config_id)
        tunnel.stop()


def update_config(server, username, port):
    db = connect('uniqueness')
    db['config'].insert({'sshtunnel': {'server': server, 'username': username, 'local_port': port, 'pid': os.getpid()}})


def remove_config(_id):
    try:
        db = connect('uniqueness')
        db['config'].remove({'_id': _id})
    except DeprecationWarning:
        pass


def setup():
    parser = argparse.ArgumentParser(description='Spawn an SSH tunnel to connect to external services')
    parser.add_argument('server', help='Server to connect to (ip address)')
    parser.add_argument('username', help='Username for server')
    args = parser.parse_args()
    main(args.server, args.username)


if __name__ == '__main__':
    setup()
