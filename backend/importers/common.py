from backend.db.db import execute_values_wrap, save_cue_response_pairs
from backend.settings import LOG_LEVEL
import logging
import subprocess, time
import paramiko
from pathlib import Path

logging.basicConfig(level=LOG_LEVEL)


def save_responses_and_users(responses, users, user_fields):
    # users is a list of dicts per user
    # responses is a list of dicts per response
    response_ids = []
    user_ids = []
    if responses:
        # assumes the same order of ids as was sent in the original list. is this wise?
        user_ids = execute_values_wrap('users', user_fields, users, returning=('external_user_id',))
        print(user_ids)
        user_ids_dict = {}
        for user_id in user_ids:
            user_ids_dict[user_id[1]] = user_id[0]
        for r in responses:
            try:
                # logging.debug('looking for external user id {}'.format(r['user_id']))
                # TODO allow for alphanumeric user ids
                r['user_id'] = user_ids_dict[int(r['user_id'])]
            except KeyError:
                raise KeyError(
                    'Error matching the newly created userIds to the external_user_ids {}, {}'.format(r['user_id'],
                                                                                                      user_ids_dict))
        response_ids = save_cue_response_pairs(responses)

    return len(response_ids), len(user_ids)


def make_external_backup(server, username, database, collection):
    home = str(Path.home())
    pkey = '{}/.ssh/id_rsa'.format(home)
    ssh = paramiko.SSHClient()
    ssh.connect(server, username=username, pkey=pkey)
    cmd_to_execute = '/usr/bin/mongoexport --collection={} --db={} --out=/tmp/{}.json --jsonArray'.format(collection,
                                                                                                          database,
                                                                                                          collection)
    ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(cmd_to_execute)
    return ssh_stdout, ssh_stderr


def get_external_backup(id_rsa, username, server, location, target):
    external_location = ''.join(['@'.join([username, server]), ":", location])
    p = subprocess.Popen(
        ["scp", id_rsa, external_location, target])
    while p.poll() is None:
        print('\nWaiting for data...\n')
        time.sleep(5)
