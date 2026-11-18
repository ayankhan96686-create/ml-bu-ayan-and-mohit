import urllib.request
import urllib.parse
import json

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
    'Content-Type': 'application/x-www-form-urlencoded',
    'Accept': 'application/json',
    'X-Requested-With': 'XMLHttpRequest'
}

form_data = {
    'source_desc': 'SURVEY',
    'sector_desc': 'ENVIRONMENTAL',
    'commodity_desc': 'CORN',
}

encoded = urllib.parse.urlencode(form_data).encode('utf-8')
req = urllib.request.Request('https://quickstats.nass.usda.gov/row_counts', data=encoded, headers=headers)
try:
    with urllib.request.urlopen(req, timeout=15) as resp:
        res = json.loads(resp.read().decode())
        print("QuickStats Environmental Corn records count:", res)
except Exception as e:
    print("Error:", e)

