import urllib.request
import os

dest_dir = r"E:\ml do not open\Maize_Yield_Research_2002_2026\01_Raw_Data\15_Other_Useful_Agricultural_Data"
os.makedirs(dest_dir, exist_ok=True)

# Centroid of Corn Belt (Story County, Ames, Iowa)
url = "https://power.larc.nasa.gov/api/temporal/daily/point?parameters=T2M,T2M_MIN,T2M_MAX,PRECTOTCORR,ALLSKY_SFC_SW_DWN,RH2M&community=AG&longitude=-93.5&latitude=42.0&start=20020101&end=20261031&format=CSV"
dest_csv = os.path.join(dest_dir, "NASA_POWER_Corn_Belt_Centroid_Daily_Weather_2002_2026.csv")

print("Downloading NASA POWER daily agricultural climate dataset (2002-2026)...")
try:
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=60) as resp, open(dest_csv, 'wb') as f:
        f.write(resp.read())
    print(f"Downloaded {dest_csv} ({os.path.getsize(dest_csv)} bytes)")
except Exception as e:
    print(f"Error downloading NASA POWER: {e}")

# Create README for folder 15
readme_path = os.path.join(dest_dir, "README_Other_Agricultural_Data.md")
with open(readme_path, 'w', encoding='utf-8') as f:
    f.write("""# 15_Other_Useful_Agricultural_Data

This directory houses supplementary agricultural reference data, crosswalks, and auxiliary environmental series:

1. **NASA POWER Agroclimatology Time Series**:
   - `NASA_POWER_Corn_Belt_Centroid_Daily_Weather_2002_2026.csv`: Daily solar radiation (`ALLSKY_SFC_SW_DWN`, MJ/m^2/day), 2m temperature (`T2M`, `T2M_MIN`, `T2M_MAX`), precipitation (`PRECTOTCORR`), and relative humidity (`RH2M`) from 2002-01-01 through 2026-10-31 for the central Corn Belt coordinate (-93.5° W, 42.0° N).
   - Source: NASA Langley Research Center POWER Project (https://power.larc.nasa.gov/).
   - License: Public domain / NASA Open Data Policy.

2. **Agricultural Statistics Districts (ASD) & Crosswalks**:
   - USDA NASS defines 9 Agricultural Statistics Districts in each state (e.g. IA-10 Northwest, IA-20 North Central, IA-50 Central).
   - These district IDs are retained in the primary yield and progress files.
""")

