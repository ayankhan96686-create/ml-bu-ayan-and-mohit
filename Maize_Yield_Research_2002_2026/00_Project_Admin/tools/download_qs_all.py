import urllib.request
import urllib.parse
import json
import csv
import os

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
print("Encoding query on QuickStats...")
req = urllib.request.Request('https://quickstats.nass.usda.gov/uuid/encode', data=encoded, headers=headers)
with urllib.request.urlopen(req, timeout=30) as resp:
    uuid_str = resp.read().decode().strip('"\r\n ')
    print("Obtained UUID:", uuid_str)

csv_url = f"https://quickstats.nass.usda.gov/data/spreadsheet/{uuid_str}.csv"
out_dir = r"E:\ml do not open\Maize_Yield_Research_2002_2026\01_Raw_Data\01_USDA_NASS_Yield"
out_file = os.path.join(out_dir, f"USDA_NASS_Corn_Grain_Yield_9States_2002_2026_{uuid_str}.csv")

print(f"Downloading CSV from {csv_url} ...")
req_csv = urllib.request.Request(csv_url, headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req_csv, timeout=60) as resp_csv, open(out_file, 'wb') as f:
    f.write(resp_csv.read())

file_size = os.path.getsize(out_file)
print(f"Downloaded successfully: {out_file} ({file_size} bytes)")

# Analyze year coverage:
year_counts = {}
with open(out_file, mode='r', encoding='utf-8', errors='ignore') as f:
    reader = csv.DictReader(f)
    for row in reader:
        yr = row.get('Year')
        year_counts[yr] = year_counts.get(yr, 0) + 1

print("\nYear breakdown:")
for yr in sorted(year_counts.keys()):
    print(f"  {yr}: {year_counts[yr]} county records")

missing_years = set(years) - set(year_counts.keys())
print("\nRequested years missing in QuickStats:", sorted(missing_years))

