import urllib.request
import re

url = 'https://www.rma.usda.gov/Information-Tools/Summary-of-Business/State-County-Crop-Coverage'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
try:
    with urllib.request.urlopen(req, timeout=15) as resp:
        html = resp.read().decode('utf-8', errors='ignore')
        matches = re.findall(r'href=["\']([^"\']+\.(?:zip|txt|csv|pdf))["\']', html)
        print('RMA links:')
        for m in sorted(set(matches)):
            print(' ', m)
except Exception as e:
    print('RMA Error:', e)

