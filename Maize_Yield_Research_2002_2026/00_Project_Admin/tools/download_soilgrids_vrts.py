import urllib.request
import os

raw_dir = r"E:\ml do not open\Maize_Yield_Research_2002_2026\01_Raw_Data\11_Soil_SoilGrids"
os.makedirs(raw_dir, exist_ok=True)

properties = ['clay', 'sand', 'silt', 'soc', 'phh2o', 'bdod']
depths = ['0-5cm', '5-15cm', '15-30cm']

for prop in properties:
    prop_dir = os.path.join(raw_dir, prop)
    os.makedirs(prop_dir, exist_ok=True)
    for d in depths:
        fn = f"{prop}_{d}_mean.vrt"
        url = f"https://files.isric.org/soilgrids/latest/data/{prop}/{fn}"
        dest = os.path.join(prop_dir, fn)
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = resp.read()
                with open(dest, 'wb') as f:
                    f.write(data)
                print(f"Downloaded SoilGrids master VRT: {prop} {d} ({len(data)} bytes)")
        except Exception as e:
            print(f"Error {prop} {d}: {e}")

print("SoilGrids master VRT collection complete.")

