import urllib.request
import re

url = 'https://quickstats.nass.usda.gov/'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
html = urllib.request.urlopen(req, timeout=15).read().decode('utf-8', errors='ignore')

scripts = re.findall(r'<script[^>]*>(.*?)</script>', html, re.DOTALL)
s = scripts[3]
idx = s.find('function onSubmit')
if idx != -1:
    print(s[idx:idx+1500])

