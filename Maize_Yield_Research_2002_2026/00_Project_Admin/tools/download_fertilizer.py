import urllib.request
import os

raw_dir = r"E:\ml do not open\Maize_Yield_Research_2002_2026\01_Raw_Data\13_Fertilizer_and_Nutrients"
ext_dir = r"E:\ml do not open\Maize_Yield_Research_2002_2026\02_Extracted_Data\13_Fertilizer_and_Nutrients"
os.makedirs(raw_dir, exist_ok=True)
os.makedirs(ext_dir, exist_ok=True)

url = "https://www.ers.usda.gov/media/5291/all-fertilizer-use-and-price-tables-in-a-single-workbook.xls"
dest_xls = os.path.join(raw_dir, "USDA_ERS_all_fertilizer_use_and_price_tables.xls")

print("Downloading USDA ERS Fertilizer Use & Price Workbook...")
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
with urllib.request.urlopen(req, timeout=30) as resp, open(dest_xls, 'wb') as f:
    f.write(resp.read())

print(f"Downloaded {dest_xls} ({os.path.getsize(dest_xls)} bytes)")

