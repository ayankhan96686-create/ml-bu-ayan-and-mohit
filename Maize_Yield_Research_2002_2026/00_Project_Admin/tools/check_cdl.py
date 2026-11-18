import urllib.request
import re

url = 'https://www.nass.usda.gov/Research_and_Science/Cropland/Release/index.php'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
try:
    with urllib.request.urlopen(req, timeout=15) as resp:
        html = resp.read().decode('utf-8', errors='ignore')
        matches = re.findall(r'href=["\']([^"\']+)["\']', html)
        print('CDL release links found:')
        for m in sorted(set(matches)):
            if any(k in m.lower() for k in ['cdl', 'cropland', 'zip', 'tar', 'geotiff', 'historical', 'release']):
                print(' ', m)
except Exception as e:
    print('CDL Error:', e)

