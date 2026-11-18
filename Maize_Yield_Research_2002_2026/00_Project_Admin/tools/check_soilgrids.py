import urllib.request
import re

url = 'https://files.isric.org/soilgrids/latest/data/'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
try:
    with urllib.request.urlopen(req, timeout=15) as resp:
        html = resp.read().decode('utf-8', errors='ignore')
        matches = re.findall(r'href=["\']([^"\']+)["\']', html)
        print('SoilGrids folders/files:')
        for m in sorted(set(matches)):
            if not m.startswith('?') and not m.startswith('/'):
                print(' ', m)
except Exception as e:
    print('SoilGrids error:', e)

