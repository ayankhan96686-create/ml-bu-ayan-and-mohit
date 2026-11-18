import urllib.request
import re
import os

raw_dir = r"E:\ml do not open\Maize_Yield_Research_2002_2026\01_Raw_Data\11_Soil_SoilGrids"
os.makedirs(raw_dir, exist_ok=True)

# Download README.md and checksum.sha256.txt
for fn in ["README.md", "checksum.sha256.txt"]:
    url = f"https://files.isric.org/soilgrids/latest/data/{fn}"
    dest = os.path.join(raw_dir, fn)
    if not os.path.exists(dest):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=15) as resp, open(dest, 'wb') as f:
                f.write(resp.read())
            print(f"Downloaded SoilGrids {fn} ({os.path.getsize(dest)} bytes)")
        except Exception as e:
            print(f"Error {fn}: {e}")

# Check clay folder
url = "https://files.isric.org/soilgrids/latest/data/clay/"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
try:
    with urllib.request.urlopen(req, timeout=15) as resp:
        html = resp.read().decode('utf-8', errors='ignore')
        matches = re.findall(r'href=["\']([^"\']+\.tif)["\']', html)
        print("Sample SoilGrids clay rasters:")
        for m in sorted(set(matches))[:6]:
            print(" ", m)
except Exception as e:
    print("Clay folder error:", e)

