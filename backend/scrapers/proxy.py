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
from pathlib import Path
import pandas as pd

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
IP_SERVICE = 'https://api.ipify.org'
LOCK_PREFIX = 'uniqueness_lock'
TMP_FOLDER = '/tmp'


def process(job=None, **kwargs):
    bad_configs = get_excluded_configs()
    files_path = [''.join([VPN_CONFIG_FOLDER_FULL, x]) for x in os.listdir(VPN_CONFIG_FOLDER_FULL) if
                  x.startswith('us') and x != 'userpass.txt' and x not in bad_configs['exclude']]
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
    proxy_server_config = choose_proxy_config(files_path)
    if not os.path.exists(USERPASSFILE):
        logging.debug('writing userpass file')
        userpassfile = open(USERPASSFILE, 'w')
        userpassfile.write(PROXY_LOGIN)
        userpassfile.write("\n")
        userpassfile.write(PROXY_PASSWORD)
        userpassfile.close()
    logging.debug(proxy_server_config)

    ## getting the IP address using socket.gethostbyname() method
    ip_address_initial = get(IP_SERVICE).text
    proxy_process = None
    if ip_address_initial == IP:
        attempts = 0
        while attempts < 3:
            try:
                proxy_process = setup_proxy(proxy_server_config)
                break
            except ConnectionError:
                stop_openvpn()
                kill(proxy_process)
                exclude_proxy_config(proxy_server_config)
                attempts += 1
                continue
    try:
        if ip_changed(ip_address_initial):
            set_lock(job)
            run_process(job, **kwargs)
    finally:
        release_lock(job)
        logging.debug('Finally trying to kill openvpn if no other processes are using it')
        final_process_check()


def set_lock(job):
    open(Path(TMP_FOLDER).joinpath('{}_{}'.format(LOCK_PREFIX, job)).absolute(), 'a').close()


def release_lock(job):
    path = Path(TMP_FOLDER).joinpath('{}_{}'.format(LOCK_PREFIX, job))
    if path.exists():
        os.remove(path.absolute())


def ip_changed(ip_address_initial):
    ## getting the IP address using socket.gethostbyname() method
    ip_address_subsequent = get(IP_SERVICE).text
    logging.debug('ip address was {} ip address now {}'.format(ip_address_initial, ip_address_subsequent))
    return ip_address_subsequent != IP


def choose_proxy_config(files_path):
    return random.choice(files_path)


def get_excluded_configs():
    result = pd.DataFrame([], columns=['exclude'])
    path = config_path()
    if path.exists():
        result = pd.read_csv(path)
    return result


def config_path():
    path = Path(VPN_CONFIG_FOLDER_FULL)
    path = path.joinpath('exclude.csv')
    return path


def exclude_proxy_config(bad_config):
    logging.debug('excluding bad config: {}'.format(bad_config))
    path = config_path()
    excludes = get_excluded_configs()
    excludes = excludes.append(pd.DataFrame([[bad_config]], columns=['exclude']))
    excludes = excludes.drop_duplicates()
    logging.debug('writing {} to {}'.format(excludes, path))
    excludes.to_csv(path, index=False)


def setup_proxy(proxy_server_config):
    command = ["sudo", "/usr/sbin/openvpn", "--writepid", OPENVPN_PID_FILE, "--config", proxy_server_config]
    proxy_process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     stdin=subprocess.PIPE)
    start_time = time.time()
    while True:
        line = proxy_process.stdout.readline().decode('utf8')
        logging.debug(line)
        try:
            if line.index('Initialization Sequence Completed'):
                break
            if line.index('AUTH_FAILED'):
                raise ConnectionError('Authentication error')
        except ValueError as e:
            time_elapsed = time.time() - start_time
            time.sleep(0.1)
            if time_elapsed > 10:
                kill(proxy_process)
                raise ConnectionError('Could not connect to proxy server: {}'.format(e))
            continue

    return proxy_process


def run_process(job, **kwargs):
    # any job that should be run behind proxy
    if job:
        module = import_module(job)
        process = getattr(module, 'process')
        process(**kwargs)


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
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass


def kill(process):
    # os.killpg(os.getpgid(pro.pid), signal.SIGTERM)
    if process is not None:
        parent = psutil.Process(process.pid)
        for child in parent.children(recursive=True):  # or parent.children() for recursive=False
            child.kill()
        parent.kill()


def final_process_check():
    # make sure openvpn is killed if there are no other scrapers running
    running_jobs = [x for x in os.listdir(TMP_FOLDER) if x.startswith(LOCK_PREFIX)]
    logging.debug('There are {} running jobs')
    if len(running_jobs) <= 1:
        if checkIfProcessRunning('openvpn'):
            logging.debug('openvpn is still runnning, using ifconfig to stop tunnel')
            tunnel_down()
            processes = findProcessIdByName('openvpn')
            logging.debug('had to kill the processes with the kernel')
            for p in processes:
                logging.debug('sending SIGSTOP to process {}'.format(p['pid']))
                os.kill(p['pid'], signal.SIGSTOP)


def tunnel_down():
    # TODO check how many tun interfaces are actually running and kill them all
    command = ["sudo", "ip", "link", "delete", "tun0"]
    subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                     stdin=subprocess.PIPE)
    command = ["sudo", "ifconfig", "tun0", "down"]
    subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                     stdin=subprocess.PIPE)


def checkIfProxyProcessRunning():
    x = subprocess.Popen(['ps', '-eo', 'args'], stdout=subprocess.PIPE)

    def check(x):
        count = 0
        for y in x.stdout:
            if 'backend.scrapers.proxy' in y:
                count += 1

    return check(x)


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
