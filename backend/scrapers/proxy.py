import argparse
import subprocess
import os
import random
from backend.settings_private import PROXY_LOGIN, PROXY_PASSWORD
from backend.settings import CWD
import logging
from backend.scrapers.human_brain_cloud import process as human_brain_cloud
import time
from requests import get
import signal
import psutil

logging.basicConfig(level=logging.DEBUG)

FNULL = open(os.devnull, 'w')


def process(humanbrain=None):
    ## getting the IP address using socket.gethostbyname() method
    ip_address_initial = get('https://api.ipify.org').text

    files_path = [''.join([CWD, '/vpn/ovpn_udp/', x]) for x in os.listdir('vpn/ovpn_udp') if
                  x.startswith('us') and x != 'userpass.txt']
    # make sure the configs are set to look for a credentials file
    for file in files_path:
        write = False
        with open(file, 'r') as f:
            lines = f.read()
            lines_split = lines.split('auth-user-pass userpass.txt')
            if len(lines_split) != 2:
                lines_split = lines.split('auth-user-pass')
                lines = ''.join([lines_split[0], '\nauth-user-pass userpass.txt\n', lines_split[1]])
                write = True
            f.close()
        if write:
            with open(file, 'w') as f:
                assert ('auth-user-pass userpass.txt' in lines)
                f.write(lines)
                f.close()
    proxy_server_config = random.choice(files_path)
    if not os.path.exists('vpn/ovpn_udp/userpass.txt'):
        logging.debug('writing userpass file')
        userpassfile = open('vpn/ovpn_udp/userpass.txt', 'w')
        userpassfile.write(PROXY_LOGIN)
        userpassfile.write("\n")
        userpassfile.write(PROXY_PASSWORD)
        userpassfile.close()
    logging.debug(proxy_server_config)
    command = ["sudo", "/usr/sbin/openvpn", proxy_server_config]
    proxy_process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        start_time = time.time()
        while True:
            line = proxy_process.stdout.readline().decode('utf8')
            logging.debug(line)
            try:
                if line.index('Initialization Sequence Completed'):
                    break
            except ValueError:
                time_elapsed = time.time() - start_time
                time.sleep(1)
                if time_elapsed > 30:
                    kill(proxy_process)
                    raise Exception('Could not connect to proxy server')
                continue

        ## getting the IP address using socket.gethostbyname() method
        ip_address_subsequent = get('https://api.ipify.org').text
        logging.debug('ip address was {} ip address now {}'.format(ip_address_initial, ip_address_subsequent))
        if ip_address_subsequent == ip_address_initial:
            kill(proxy_process)
            raise Exception('IP Address not changed')
        # various jobs to be run behind proxy
        if humanbrain:
            human_brain_cloud()

        kill(proxy_process)
    except Exception as e:
        logging.debug(e)
        logging.debug('attempting to kill open vpn processes')
        kill(proxy_process)


def kill(process):
    parent = psutil.Process(process.pid)
    for child in parent.children(recursive=True):  # or parent.children() for recursive=False
        child.kill()
    parent.kill()


def setup():
    parser = argparse.ArgumentParser(description='Import various types of word data')
    parser.add_argument('--humanbrain', action='store_true', help='Get human brain data via proxy')
    args = parser.parse_args()
    process(humanbrain=args.humanbrain)


if __name__ == "__main__":
    setup()
