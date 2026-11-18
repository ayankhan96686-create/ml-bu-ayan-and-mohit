import urllib.request
import urllib.parse
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

form_data = [
    ('source_desc', 'SURVEY'),
    ('sector_desc', 'CROPS'),
    ('group_desc', 'FIELD CROPS'),
    ('commodity_desc', 'CORN'),
    ('statisticcat_desc', 'PROGRESS'),
]
for st in states:
    form_data.append(('state_name', st))

encoded = urllib.parse.urlencode(form_data).encode('utf-8')
print("Encoding query for Crop Progress data...")
req = urllib.request.Request('https://quickstats.nass.usda.gov/uuid/encode', data=encoded, headers=headers)
with urllib.request.urlopen(req, timeout=30) as resp:
    uuid_str = resp.read().decode().strip('"\r\n ')
    print("Obtained UUID:", uuid_str)

csv_url = f"https://quickstats.nass.usda.gov/data/spreadsheet/{uuid_str}.csv"
out_dir = r"E:\ml do not open\Maize_Yield_Research_2002_2026\01_Raw_Data\12_Planting_and_Crop_Management"
out_file = os.path.join(out_dir, f"USDA_NASS_Corn_Crop_Progress_9States_{uuid_str}.csv")

print(f"Downloading CSV from {csv_url} ...")
req_csv = urllib.request.Request(csv_url, headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req_csv, timeout=60) as resp_csv, open(out_file, 'wb') as f:
    f.write(resp_csv.read())

sz = os.path.getsize(out_file)
print(f"Downloaded successfully: {out_file} ({sz} bytes)")

