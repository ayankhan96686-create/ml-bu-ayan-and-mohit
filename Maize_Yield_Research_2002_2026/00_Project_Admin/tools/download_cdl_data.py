import urllib.request
import os

raw_dir = r"E:\ml do not open\Maize_Yield_Research_2002_2026\01_Raw_Data\02_USDA_CDL_Crop_Classification"
os.makedirs(raw_dir, exist_ok=True)

downloads = [
    ("US_CDL_Legend.png", "https://www.nass.usda.gov/sites/default/files/Research_and_Science/Cropland/docs/US_CDL_Legend.png"),
    ("US_2024_CDL_legend.jpg", "https://www.nass.usda.gov/sites/default/files/Research_and_Science/Cropland/docs/US_2024_CDL_legend.jpg"),
    ("US_2023_CDL_legend.jpg", "https://www.nass.usda.gov/sites/default/files/Research_and_Science/Cropland/docs/US_2023_CDL_legend.jpg"),
    ("CDL_2020_19_Iowa.tif", "https://nassgeodata.gmu.edu/webservice/nass_data_cache/byfips/CDL_2020_19.tif")
]

for fname, url in downloads:
    dest = os.path.join(raw_dir, fname)
    if os.path.exists(dest):
        print(f"Already exists: {fname}")
        continue
    print(f"Downloading {fname} from {url}...")
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=120) as resp, open(dest, 'wb') as f:
            f.write(resp.read())
        print(f"  Downloaded {fname} ({os.path.getsize(dest)} bytes)")
    except Exception as e:
        print(f"  Error {fname}: {e}")

# Also create official CDL class definition CSV
cdl_classes_csv = os.path.join(raw_dir, "USDA_CDL_Crop_Classification_Codes.csv")
import csv
cdl_codes = [
    (1, "Corn", "Field Crops", "#FFD300"),
    (2, "Cotton", "Field Crops", "#FF2626"),
    (3, "Rice", "Field Crops", "#00A8E2"),
    (4, "Sorghum", "Field Crops", "#FF9E0A"),
    (5, "Soybeans", "Field Crops", "#267000"),
    (6, "Sunflower", "Field Crops", "#FFFF00"),
    (12, "Sweet Corn", "Vegetables", "#DDAA00"),
    (13, "Pop or Orn Corn", "Field Crops", "#DDAA55"),
    (21, "Barley", "Field Crops", "#E2007C"),
    (22, "Durum Wheat", "Field Crops", "#896E43"),
    (23, "Spring Wheat", "Field Crops", "#DAB680"),
    (24, "Winter Wheat", "Field Crops", "#A57000"),
    (25, "Other Small Grains", "Field Crops", "#D69EBC"),
    (26, "Dbl Crop WinWht/Soybeans", "Double Crops", "#707000"),
    (27, "Rye", "Field Crops", "#AB0000"),
    (28, "Oats", "Field Crops", "#A05A32"),
    (225, "Dbl Crop WinWht/Corn", "Double Crops", "#FFD300"),
    (226, "Dbl Crop Oats/Corn", "Double Crops", "#FFD300"),
    (228, "Dbl Crop Triticale/Corn", "Double Crops", "#FFD300"),
    (237, "Dbl Crop Barley/Corn", "Double Crops", "#FFD300"),
    (241, "Dbl Crop Corn/Soybeans", "Double Crops", "#FFD300"),
]

with open(cdl_classes_csv, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["CDL_Code", "Class_Name", "Category", "Hex_Color"])
    for row in cdl_codes:
        writer.writerow(row)

print("CDL downloads and reference specifications complete.")

