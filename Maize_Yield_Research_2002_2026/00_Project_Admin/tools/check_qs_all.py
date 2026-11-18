import urllib.request
import urllib.parse
import json

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
    'Content-Type': 'application/x-www-form-urlencoded',
    'Accept': 'application/json, text/javascript, */*; q=0.01',
    'X-Requested-With': 'XMLHttpRequest'
}

states = [
    'IOWA', 'ILLINOIS', 'INDIANA', 'MINNESOTA', 
    'MISSOURI', 'NEBRASKA', 'OHIO', 'SOUTH DAKOTA', 'WISCONSIN'
]
years = [str(y) for y in range(2002, 2027)]

form_data = [
    ('source_desc', 'SURVEY'),
    ('sector_desc', 'CROPS'),
    ('group_desc', 'FIELD CROPS'),
    ('commodity_desc', 'CORN'),
    ('statisticcat_desc', 'YIELD'),
    ('short_desc', 'CORN, GRAIN - YIELD, MEASURED IN BU / ACRE'),
    ('agg_level_desc', 'COUNTY'),
]
for st in states:
    form_data.append(('state_name', st))
for y in years:
    form_data.append(('year', y))

encoded = urllib.parse.urlencode(form_data).encode('utf-8')
req = urllib.request.Request('https://quickstats.nass.usda.gov/row_counts', data=encoded, headers=headers)
try:
    with urllib.request.urlopen(req, timeout=15) as resp:
        res = json.loads(resp.read().decode())
        print(f"Total row count for 9 states (2002-2026): {res.get('rowcnt')}")
except Exception as e:
    print(f"Error: {e}")

