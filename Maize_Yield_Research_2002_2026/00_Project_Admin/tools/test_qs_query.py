import urllib.request
import urllib.parse
import json

# Let's test a simple query:
# Commodity: CORN
# Sector: CROPS
# Group: FIELD CROPS
# Statistic Category: YIELD
# State: IOWA
# Agg level: COUNTY
# Year: 2020

data = {
    'source_desc': 'SURVEY',
    'sector_desc': 'CROPS',
    'group_desc': 'FIELD CROPS',
    'commodity_desc': 'CORN',
    'statisticcat_desc': 'YIELD',
    'short_desc': 'CORN, GRAIN - YIELD, MEASURED IN BU / ACRE',
    'agg_level_desc': 'COUNTY',
    'state_name': 'IOWA',
    'year': '2020'
}

encoded_data = urllib.parse.urlencode(data).encode('utf-8')
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
    'Content-Type': 'application/x-www-form-urlencoded',
    'Accept': 'application/json, text/javascript, */*; q=0.01',
    'X-Requested-With': 'XMLHttpRequest'
}

req = urllib.request.Request('https://quickstats.nass.usda.gov/row_counts', data=encoded_data, headers=headers)
try:
    with urllib.request.urlopen(req, timeout=15) as resp:
        print("row_counts response:", resp.read().decode())
except Exception as e:
    print("row_counts error:", e)

req2 = urllib.request.Request('https://quickstats.nass.usda.gov/uuid/encode', data=encoded_data, headers=headers)
try:
    with urllib.request.urlopen(req2, timeout=15) as resp:
        uuid_resp = resp.read().decode()
        print("uuid/encode response:", uuid_resp)
        uuid_clean = uuid_resp.strip('"\r\n ')
        # Now let's see what /results/{uuid} looks like
        res_url = f"https://quickstats.nass.usda.gov/results/{uuid_clean}"
        req3 = urllib.request.Request(res_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req3, timeout=15) as resp3:
            html = resp3.read().decode('utf-8', errors='ignore')
            print(f"Results page length: {len(html)}")
            for line in html.splitlines():
                if any(k in line.lower() for k in ['csv', 'download', 'spreadsheet', 'data_item']):
                    print(line[:120].strip())
except Exception as e:
    print("uuid/encode error:", e)

