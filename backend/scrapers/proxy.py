import argparse
import subprocess
import os
import random
from backend.settings_private import PROXY_LOGIN, PROXY_PASSWORD
from backend.settings import PROJECT_ROOT, LOG_LEVEL
from backend.settings_private import IP
import logging
import time
from requests import get
import psutil
import signal
import json
from importlib import import_module

logging.basicConfig(level=LOG_LEVEL)
# this cannot be worked out using os.getcwd as the proxy script needs to be run from sudo so the CWD would be
# wrong in this case (root/..etc)
CWD = PROJECT_ROOT
FNULL = open(os.devnull, 'w')
VPN_CONFIG_FOLDER = '/vpn/ovpn_udp/'
VPN_CONFIG_FOLDER_FULL = ''.join([CWD, VPN_CONFIG_FOLDER])
USERPASSFILE = ''.join([CWD, VPN_CONFIG_FOLDER, 'userpass.txt'])
USERPASSCONFIG = 'auth-user-pass {}'.format(USERPASSFILE)
OPENVPN_PID_FILE = "/tmp/openvpn_pid"


def process(job=None, **kwargs):
    ## getting the IP address using socket.gethostbyname() method
    ip_address_initial = IP

    files_path = [''.join([VPN_CONFIG_FOLDER_FULL, x]) for x in os.listdir(VPN_CONFIG_FOLDER_FULL) if
                  x.startswith('us') and x != 'userpass.txt']
    # make sure the configs are set to look for a credentials file
    for file in files_path:
        write = False
        with open(file, 'r') as f:
            lines = f.read()
            lines_split = lines.split(USERPASSCONFIG)
            if len(lines_split) != 2:
                lines_split = lines.split('auth-user-pass')
                lines = ''.join([lines_split[0], '\n', USERPASSCONFIG, '\n', lines_split[1]])
                write = True
            f.close()
        if write:
            with open(file, 'w') as f:
                assert (USERPASSCONFIG in lines)
                f.write(lines)
                f.close()
    proxy_server_config = random.choice(files_path)
    if not os.path.exists(USERPASSFILE):
        logging.debug('writing userpass file')
        userpassfile = open(USERPASSFILE, 'w')
        userpassfile.write(PROXY_LOGIN)
        userpassfile.write("\n")
        userpassfile.write(PROXY_PASSWORD)
        userpassfile.close()
    logging.debug(proxy_server_config)
    command = ["sudo", "/usr/sbin/openvpn", "--writepid", OPENVPN_PID_FILE, "--config", proxy_server_config]
    proxy_process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.PIPE)
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
                time.sleep(0.1)
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
        if job:
            module = import_module(job)
            process = getattr(module, 'process')
            process(**kwargs)
        stop_openvpn()
        kill(proxy_process)
    except (Exception, KeyboardInterrupt) as e:
        logging.debug(e)
        if proxy_process is not None:
            stop_openvpn()
        logging.debug('attempting to kill open vpn processes')
        kill(proxy_process)
        time.sleep(1)
        if checkIfProcessRunning('openvpn'):
            processes = findProcessIdByName('openvpn')
            logging.debug('had to kill the processes with the kernel')
            for p in processes:
                os.kill(p['pid'], signal.SIGSTOP)


def process_output(proxy_process, max_time=30):
    start_time = time.time()
    while True:
        time_elapsed = time.time() - start_time
        line = proxy_process.stdout.readline().decode('utf8')
        logging.debug(line)
        if time_elapsed > max_time:
            break


def stop_openvpn():
    pid_file = open(OPENVPN_PID_FILE, mode='r')
    pid = int(pid_file.read())
    logging.debug('sending stop command to proxy process {}'.format(pid))
    os.kill(pid, signal.SIGTERM)


def kill(process):
    parent = psutil.Process(process.pid)
    for child in parent.children(recursive=True):  # or parent.children() for recursive=False
        child.kill()
    parent.kill()


def checkIfProcessRunning(processName):
    '''
    Check if there is any running process that contains the given name processName.
    '''
    # Iterate over the all the running process
    for proc in psutil.process_iter():
        try:
            # Check if process name contains the given name string.
            if processName.lower() in proc.name().lower():
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return False


def findProcessIdByName(processName):
    '''
    Get a list of all the PIDs of a all the running process whose name contains
    the given string processName
    '''

    listOfProcessObjects = []

    # Iterate over the all the running process
    for proc in psutil.process_iter():
        try:
            pinfo = proc.as_dict(attrs=['pid', 'name', 'create_time'])
            # Check if process name contains the given name string.
            if processName.lower() in pinfo['name'].lower():
                listOfProcessObjects.append(pinfo)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

    return listOfProcessObjects


def setup():
    parser = argparse.ArgumentParser(description='Import various types of word data')
    parser.add_argument('job', help='Get job name to run via proxy')
    parser.add_argument('args', help='JSON of arguments for the job')
    args = parser.parse_args()
    process(args.job, **json.loads(args.args))


if __name__ == "__main__":
    setup()
