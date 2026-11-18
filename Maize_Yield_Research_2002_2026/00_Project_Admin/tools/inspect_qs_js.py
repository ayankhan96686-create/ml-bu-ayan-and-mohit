import urllib.request
import re

url = 'https://quickstats.nass.usda.gov/'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
html = urllib.request.urlopen(req, timeout=15).read().decode('utf-8', errors='ignore')

# find script tags or inline js
scripts = re.findall(r'<script[^>]*>(.*?)</script>', html, re.DOTALL)
print(f"Total script blocks: {len(scripts)}")
for i, s in enumerate(scripts):
    if any(k in s for k in ['saveFilters', 'onSubmit', 'results', '_uuid', 'xhrPost', 'fetch', 'ajax']):
        print(f"--- Script {i} ---")
        for line in s.splitlines():
            if any(k in line for k in ['onSubmit', 'results', 'uuid', 'post', 'action', 'location', 'csv']):
                print(line.strip())

