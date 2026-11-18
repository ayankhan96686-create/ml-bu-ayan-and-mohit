import urllib.request
import urllib.parse
import json
import os
import time

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
    'Content-Type': 'application/x-www-form-urlencoded',
    'Accept': 'application/json, text/javascript, */*; q=0.01',
    'X-Requested-With': 'XMLHttpRequest'
}

states = [
    ('IOWA', 'IA', '19'),
    ('ILLINOIS', 'IL', '17'),
    ('INDIANA', 'IN', '18'),
    ('MINNESOTA', 'MN', '27'),
    ('MISSOURI', 'MO', '29'),
    ('NEBRASKA', 'NE', '31'),
    ('OHIO', 'OH', '39'),
    ('SOUTH DAKOTA', 'SD', '46'),
    ('WISCONSIN', 'WI', '55')
]
years = [str(y) for y in range(2002, 2027)]
out_dir = r"E:\ml do not open\Maize_Yield_Research_2002_2026\01_Raw_Data\01_USDA_NASS_Yield"

for st_name, st_abbr, fips in states:
    print(f"Processing {st_name} ({st_abbr})...")
    form_data = [
        ('source_desc', 'SURVEY'),
        ('sector_desc', 'CROPS'),
        ('group_desc', 'FIELD CROPS'),
        ('commodity_desc', 'CORN'),
        ('statisticcat_desc', 'YIELD'),
        ('short_desc', 'CORN, GRAIN - YIELD, MEASURED IN BU / ACRE'),
        ('agg_level_desc', 'COUNTY'),
        ('state_name', st_name)
    ]
    for y in years:
        form_data.append(('year', y))
    
    encoded = urllib.parse.urlencode(form_data).encode('utf-8')
    req = urllib.request.Request('https://quickstats.nass.usda.gov/uuid/encode', data=encoded, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as resp:
        uuid_str = resp.read().decode().strip('"\r\n ')
    
    csv_url = f"https://quickstats.nass.usda.gov/data/spreadsheet/{uuid_str}.csv"
    filename = f"USDA_NASS_Corn_Yield_{st_abbr}_2002_2026_{uuid_str}.csv"
    dest_path = os.path.join(out_dir, filename)
    
    req_csv = urllib.request.Request(csv_url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req_csv, timeout=60) as resp_csv, open(dest_path, 'wb') as f:
        f.write(resp_csv.read())
        
    sz = os.path.getsize(dest_path)
    print(f"  Saved {filename} ({sz} bytes)")
    time.sleep(1) # respectful rate limiting

print("All state yield files downloaded.")

