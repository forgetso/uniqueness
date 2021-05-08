import requests
import pandas as pd
from backend.settings import VPN_URLS
from pathlib import Path

urls = pd.read_csv(VPN_URLS, names=['url'])
for url in urls['url']:
    if '.com.udp' in url:
        r = requests.get(url)
        file = url.split('/')[-1]
        print(Path('./vpn/ovpn_udp').joinpath(file).absolute())
        with Path('./vpn/ovpn_udp').joinpath(file).open(mode='w', encoding='utf8') as f:
            f.write(r.content.decode('utf8'))
