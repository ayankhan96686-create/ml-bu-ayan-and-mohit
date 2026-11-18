import urllib.request
import re

url = 'https://quickstats.nass.usda.gov/'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
html = urllib.request.urlopen(req, timeout=15).read().decode('utf-8', errors='ignore')

scripts = re.findall(r'<script[^>]*>(.*?)</script>', html, re.DOTALL)
for i, s in enumerate(scripts):
    if 'loadSubmit' in s or 'results' in s:
        print(f"--- Script {i} ---")
        for line in s.splitlines():
            if any(k in line.lower() for k in ['loadsubmit', 'results', 'download', 'csv', 'spreadsheet']):
                print(line.strip())

