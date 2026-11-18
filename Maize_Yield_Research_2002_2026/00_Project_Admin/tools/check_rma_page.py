import urllib.request
import re

url = 'https://www.rma.usda.gov/en/Tools-and-Reports/Summary-of-Business/StateCountyCrop'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
try:
    with urllib.request.urlopen(req, timeout=15) as resp:
        html = resp.read().decode('utf-8', errors='ignore')
        matches = re.findall(r'href=["\']([^"\']+\.zip)["\']', html)
        print('RMA zip links found:', len(matches))
        for m in sorted(set(matches))[:15]:
            print(' ', m)
except Exception as e:
    print('RMA Error:', e)

