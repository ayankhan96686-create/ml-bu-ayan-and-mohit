import urllib.request
import os
import zipfile
import time

raw_dir = r"E:\ml do not open\Maize_Yield_Research_2002_2026\01_Raw_Data\03_PRISM_Weather"
ext_dir = r"E:\ml do not open\Maize_Yield_Research_2002_2026\02_Extracted_Data\03_PRISM_Weather"
os.makedirs(raw_dir, exist_ok=True)
os.makedirs(ext_dir, exist_ok=True)

elements = ['ppt', 'tmean', 'tmin', 'tmax']
# Let's download a systematic sequence: annual grids for 2002 through 2025, and check 2026
years = [str(y) for y in range(2002, 2027)]

print(f"Starting PRISM gridded data collection for elements {elements} across {len(years)} years...")

results = {}
for elem in elements:
    elem_raw = os.path.join(raw_dir, elem)
    elem_ext = os.path.join(ext_dir, elem)
    os.makedirs(elem_raw, exist_ok=True)
    os.makedirs(elem_ext, exist_ok=True)
    
    for yr in years:
        url = f"https://services.nacse.org/prism/data/get/us/4km/{elem}/{yr}"
        dest_zip = os.path.join(elem_raw, f"PRISM_{elem}_us_4km_{yr}.zip")
        
        if os.path.exists(dest_zip) and os.path.getsize(dest_zip) > 10000:
            print(f"  Already exists: {elem} {yr} ({os.path.getsize(dest_zip)} bytes)")
            continue
            
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                content_type = resp.headers.get('Content-Type', '')
                if 'zip' in content_type:
                    data = resp.read()
                    with open(dest_zip, 'wb') as f:
                        f.write(data)
                    print(f"  Downloaded: {elem} {yr} ({len(data)} bytes)")
                    
                    # Extract
                    sub_ext = os.path.join(elem_ext, yr)
                    os.makedirs(sub_ext, exist_ok=True)
                    with zipfile.ZipFile(dest_zip, 'r') as z:
                        z.extractall(sub_ext)
                else:
                    msg = resp.read().decode('utf-8', errors='ignore')[:100]
                    print(f"  Not available / provisional message for {elem} {yr}: {msg}")
        except Exception as e:
            print(f"  Error {elem} {yr}: {e}")
        time.sleep(0.5)

print("PRISM download batch complete.")

