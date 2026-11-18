import urllib.request
import re

url = 'https://www.nass.usda.gov/Research_and_Science/Cropland/sarsfaqs2.php'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
try:
    with urllib.request.urlopen(req, timeout=15) as resp:
        html = resp.read().decode('utf-8', errors='ignore')
        # Find FAQ questions or table of state coverage
        print('FAQs page length:', len(html))
        # Look for corn code or state history
        for line in html.splitlines():
            if any(k in line.lower() for k in ['corn', 'coverage', '1997', '1999', '2002', 'legend', 'code']):
                if len(line.strip()) < 150:
                    print(line.strip())
except Exception as e:
    print('FAQ error:', e)

