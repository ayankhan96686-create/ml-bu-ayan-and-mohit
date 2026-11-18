import urllib.request
import re

url = 'https://www.ers.usda.gov/data-products/fertilizer-use-and-price/'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
try:
    with urllib.request.urlopen(req, timeout=15) as resp:
        html = resp.read().decode('utf-8', errors='ignore')
        matches = re.findall(r'href=["\']([^"\']+)["\']', html)
        for m in sorted(set(matches)):
            if any(k in m.lower() for k in ['fertilizer', 'download', 'table', 'datafile']):
                print(' ', m)
except Exception as e:
    print('ERS Error:', e)

