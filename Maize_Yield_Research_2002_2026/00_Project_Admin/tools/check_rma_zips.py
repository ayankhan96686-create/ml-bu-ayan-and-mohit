import urllib.request
import re

url = 'https://www.rma.usda.gov/tools-reports/summary-of-business/state-county-crop-summary-business'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
try:
    with urllib.request.urlopen(req, timeout=15) as resp:
        html = resp.read().decode('utf-8', errors='ignore')
        matches = re.findall(r'href=["\']([^"\']+\.zip)["\']', html)
        print('RMA files found:', len(matches))
        for m in sorted(set(matches)):
            print(' ', m)
except Exception as e:
    print('RMA Error:', e)

