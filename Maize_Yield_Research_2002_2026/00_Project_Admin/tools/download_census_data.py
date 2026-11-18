import urllib.request
import os
import zipfile

raw_dir = r"E:\ml do not open\Maize_Yield_Research_2002_2026\01_Raw_Data\14_County_and_State_Boundaries"
ext_dir = r"E:\ml do not open\Maize_Yield_Research_2002_2026\02_Extracted_Data\14_County_and_State_Boundaries"
os.makedirs(raw_dir, exist_ok=True)
os.makedirs(ext_dir, exist_ok=True)

files = [
    ("national_county2020.txt", "https://www2.census.gov/geo/docs/reference/codes2020/national_county2020.txt", False),
    ("tl_2024_us_state.zip", "https://www2.census.gov/geo/tiger/TIGER2024/STATE/tl_2024_us_state.zip", True),
    ("tl_2024_us_county.zip", "https://www2.census.gov/geo/tiger/TIGER2024/COUNTY/tl_2024_us_county.zip", True),
]

for fname, url, is_zip in files:
    dest_path = os.path.join(raw_dir, fname)
    if not os.path.exists(dest_path):
        print(f"Downloading {fname} from {url}...")
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=120) as resp, open(dest_path, 'wb') as f:
            f.write(resp.read())
        print(f"  Downloaded {fname} ({os.path.getsize(dest_path)} bytes)")
    else:
        print(f"  Already exists: {fname} ({os.path.getsize(dest_path)} bytes)")

    if is_zip:
        sub_ext = os.path.join(ext_dir, os.path.splitext(fname)[0])
        os.makedirs(sub_ext, exist_ok=True)
        print(f"Extracting {fname} to {sub_ext}...")
        with zipfile.ZipFile(dest_path, 'r') as z:
            z.extractall(sub_ext)
            print(f"  Extracted {len(z.namelist())} files.")

# Generate official 9-state study area county list CSV with FIPS
print("Generating official study-area county manifest...")
census_txt = os.path.join(raw_dir, "national_county2020.txt")
study_counties_csv = r"E:\ml do not open\Maize_Yield_Research_2002_2026\00_Project_Admin\Study_Area_Counties_FIPS.csv"
target_fips = {
    '19': ('Iowa', 'IA'),
    '17': ('Illinois', 'IL'),
    '18': ('Indiana', 'IN'),
    '27': ('Minnesota', 'MN'),
    '29': ('Missouri', 'MO'),
    '31': ('Nebraska', 'NE'),
    '39': ('Ohio', 'OH'),
    '46': ('South Dakota', 'SD'),
    '55': ('Wisconsin', 'WI')
}

import csv
records = []
with open(census_txt, 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='|')
    header = next(reader)
    for row in reader:
        if len(row) >= 7 and row[1] in target_fips:
            st_name, st_abbr = target_fips[row[1]]
            full_fips = row[1] + row[2]
            records.append({
                'State_FIPS': row[1],
                'State_Abbr': st_abbr,
                'State_Name': st_name,
                'County_FIPS': row[2],
                'Full_FIPS': full_fips,
                'County_Name': row[4],
                'County_NS': row[3],
                'Class_FIPS': row[5],
                'Func_Stat': row[6]
            })

with open(study_counties_csv, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=[
        'State_FIPS', 'State_Abbr', 'State_Name', 'County_FIPS', 'Full_FIPS',
        'County_Name', 'County_NS', 'Class_FIPS', 'Func_Stat'
    ])
    writer.writeheader()
    writer.writerows(records)

print(f"Generated {study_counties_csv} with {len(records)} official counties across 9 states.")

