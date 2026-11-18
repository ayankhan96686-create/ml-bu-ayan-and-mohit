import urllib.request
import re

url = 'https://www.nass.usda.gov/Data_and_Statistics/index.php'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
try:
    html = urllib.request.urlopen(req, timeout=15).read().decode('utf-8', errors='ignore')
    matches = re.findall(r'href=["\']([^"\']+)["\']', html)
    print("Found links:")
    for m in sorted(set(matches)):
        if any(k in m.lower() for k in ['quickstats', 'bulk', 'data', 'download', 'dataset', 'crop']):
            print(m)
except Exception as e:
    print("Error:", e)

