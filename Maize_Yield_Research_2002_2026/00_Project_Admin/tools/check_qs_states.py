import urllib.request
import urllib.parse
import json

states = [
    'IOWA', 'ILLINOIS', 'INDIANA', 'MINNESOTA', 
    'MISSOURI', 'NEBRASKA', 'OHIO', 'SOUTH DAKOTA', 'WISCONSIN'
]

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
    'Content-Type': 'application/x-www-form-urlencoded',
    'Accept': 'application/json, text/javascript, */*; q=0.01',
    'X-Requested-With': 'XMLHttpRequest'
}

print("Checking QuickStats row counts by state for Corn Grain Yield (all years 2002-2026)...")

for st in states:
    data = {
        'source_desc': 'SURVEY',
        'sector_desc': 'CROPS',
        'group_desc': 'FIELD CROPS',
        'commodity_desc': 'CORN',
        'statisticcat_desc': 'YIELD',
        'short_desc': 'CORN, GRAIN - YIELD, MEASURED IN BU / ACRE',
        'agg_level_desc': 'COUNTY',
        'state_name': st,
    }
    encoded = urllib.parse.urlencode(data).encode('utf-8')
    req = urllib.request.Request('https://quickstats.nass.usda.gov/row_counts', data=encoded, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            res = json.loads(resp.read().decode())
            print(f"State {st}: {res.get('rowcnt')} records across all years")
    except Exception as e:
        print(f"State {st} error: {e}")

