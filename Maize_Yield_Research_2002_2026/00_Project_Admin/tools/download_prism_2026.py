import urllib.request
import os
import zipfile
import time

raw_dir = r"E:\ml do not open\Maize_Yield_Research_2002_2026\01_Raw_Data\03_PRISM_Weather"
ext_dir = r"E:\ml do not open\Maize_Yield_Research_2002_2026\02_Extracted_Data\03_PRISM_Weather"

elements = ['ppt', 'tmean', 'tmin', 'tmax']
months_2026 = [f"2026{m:02d}" for m in range(5, 11)] # May to October 2026 (Corn growing season)

print("Downloading available provisional 2026 monthly PRISM grids (May-Oct 2026)...")

for elem in elements:
    elem_raw = os.path.join(raw_dir, f"{elem}_2026_provisional")
    elem_ext = os.path.join(ext_dir, f"{elem}_2026_provisional")
    os.makedirs(elem_raw, exist_ok=True)
    os.makedirs(elem_ext, exist_ok=True)
    
    for ym in months_2026:
        url = f"https://services.nacse.org/prism/data/get/us/4km/{elem}/{ym}"
        dest_zip = os.path.join(elem_raw, f"PRISM_{elem}_us_4km_{ym}_provisional.zip")
        
        if os.path.exists(dest_zip):
            continue
            
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                if 'zip' in resp.headers.get('Content-Type', ''):
                    data = resp.read()
                    with open(dest_zip, 'wb') as f:
                        f.write(data)
                    print(f"  Downloaded 2026 provisional: {elem} {ym} ({len(data)} bytes)")
                    
                    sub_ext = os.path.join(elem_ext, ym)
                    os.makedirs(sub_ext, exist_ok=True)
                    with zipfile.ZipFile(dest_zip, 'r') as z:
                        z.extractall(sub_ext)
                else:
                    print(f"  Message for {elem} {ym}: {resp.read().decode('utf-8', errors='ignore')[:80]}")
        except Exception as e:
            print(f"  Error {elem} {ym}: {e}")
        time.sleep(0.5)

print("2026 provisional PRISM collection complete.")

