import urllib.request
import re

url = 'https://www.ncei.noaa.gov/pub/data/cirs/climdiv/'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
try:
    with urllib.request.urlopen(req, timeout=15) as resp:
        html = resp.read().decode('utf-8', errors='ignore')
        matches = re.findall(r'href=["\'](climdiv-[^"\']+)["\']', html)
        print('NOAA climdiv files found:')
        for m in sorted(set(matches)):
            if any(k in m for k in ['pcpncy', 'tmpccy', 'tmincy', 'tmaxcy']):
                print(' ', m)
except Exception as e:
    print('NOAA Error:', e)

