import urllib.request
import os
import csv

raw_dir = r"E:\ml do not open\Maize_Yield_Research_2002_2026\01_Raw_Data\05_NOAA_Weather"
ext_dir = r"E:\ml do not open\Maize_Yield_Research_2002_2026\02_Extracted_Data\05_NOAA_Weather"
os.makedirs(raw_dir, exist_ok=True)
os.makedirs(ext_dir, exist_ok=True)

# Study area state FIPS
target_state_fips = {'19', '17', '18', '27', '29', '31', '39', '46', '55'}

files = [
    ("climdiv-pcpncy-v1.0.0-20261006", "precipitation"),
    ("climdiv-tmpccy-v1.0.0-20261006", "temp_mean"),
    ("climdiv-tmaxcy-v1.0.0-20261006", "temp_max"),
    ("climdiv-tmincy-v1.0.0-20261006", "temp_min"),
]

base_url = "https://www.ncei.noaa.gov/pub/data/cirs/climdiv/"

months = [f"Month_{m:02d}" for m in range(1, 13)]
csv_header = ["State_FIPS", "County_FIPS", "Element_Code", "Year"] + months

for fname, var_name in files:
    raw_path = os.path.join(raw_dir, fname)
    url = base_url + fname
    if not os.path.exists(raw_path):
        print(f"Downloading {fname} from {url}...")
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=120) as resp, open(raw_path, 'wb') as f:
            f.write(resp.read())
        print(f"  Downloaded {fname} ({os.path.getsize(raw_path)} bytes)")
    else:
        print(f"  Already exists: {fname} ({os.path.getsize(raw_path)} bytes)")

    # Extract study area 2002-2026 rows
    out_csv = os.path.join(ext_dir, f"NOAA_County_{var_name}_CornBelt_2002_2026.csv")
    print(f"Extracting 2002-2026 study area records to {out_csv}...")
    extracted_rows = []
    with open(raw_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if len(line) < 11:
                continue
            code = line[:11]
            st_fips = code[:2]
            cty_fips = code[2:5]
            elem = code[5:7]
            year = int(code[7:11])
            
            if st_fips in target_state_fips and 2002 <= year <= 2026:
                vals = line[11:].split()
                # Pad to 12 months if needed
                while len(vals) < 12:
                    vals.append("-99.99")
                extracted_rows.append([st_fips, cty_fips, elem, year] + vals)
                
    with open(out_csv, 'w', newline='', encoding='utf-8') as out_f:
        writer = csv.writer(out_f)
        writer.writerow(csv_header)
        writer.writerows(extracted_rows)
    print(f"  Extracted {len(extracted_rows)} records for {var_name} (2002-2026).")

print("NOAA weather data processing complete.")

