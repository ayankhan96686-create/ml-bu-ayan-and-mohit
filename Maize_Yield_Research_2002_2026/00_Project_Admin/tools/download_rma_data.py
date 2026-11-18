import urllib.request
import os
import zipfile
import time
import csv

raw_dir = r"E:\ml do not open\Maize_Yield_Research_2002_2026\01_Raw_Data\12_Planting_and_Crop_Management"
ext_dir = r"E:\ml do not open\Maize_Yield_Research_2002_2026\02_Extracted_Data\12_Planting_and_Crop_Management"
os.makedirs(raw_dir, exist_ok=True)
os.makedirs(ext_dir, exist_ok=True)

target_state_fips = {'19', '17', '18', '27', '29', '31', '39', '46', '55'}
years = [str(y) for y in range(2002, 2027)]

print(f"Downloading USDA RMA Summary of Business (sobcov) files for {len(years)} years (2002-2026)...")

corn_records_all = []

for yr in years:
    fname = f"sobcov_{yr}.zip"
    dest_path = os.path.join(raw_dir, fname)
    url = f"https://pubfs-rma.fpac.usda.gov/pub/Web_Data_Files/Summary_of_Business/state_county_crop/sobcov_{yr}.zip"
    
    if not os.path.exists(dest_path):
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp, open(dest_path, 'wb') as f:
                f.write(resp.read())
            print(f"  Downloaded {fname} ({os.path.getsize(dest_path)} bytes)")
        except Exception as e:
            print(f"  Error downloading {fname}: {e}")
            continue
    else:
        print(f"  Already exists: {fname}")

    # Extract archive
    sub_ext = os.path.join(ext_dir, f"sobcov_{yr}")
    os.makedirs(sub_ext, exist_ok=True)
    with zipfile.ZipFile(dest_path, 'r') as z:
        z.extractall(sub_ext)
        txt_files = [f for f in z.namelist() if f.endswith('.txt')]
        if txt_files:
            txt_file = os.path.join(sub_ext, txt_files[0])
            # Filter corn records in our study area states
            # RMA layout: commodity name or code (Corn is 0041 or 'CORN')
            with open(txt_file, 'r', encoding='latin1') as tf:
                for line in tf:
                    parts = line.strip().split('|')
                    # check if state FIPS matches and commodity is corn
                    if len(parts) >= 6:
                        # RMA typically has: Year|State_FIPS|State_Abbr|County_FIPS|County_Name|Commodity_Code|Commodity_Name...
                        st_fips = parts[1].zfill(2)
                        comm = parts[5].upper() if len(parts) > 5 else ''
                        comm_name = parts[6].upper() if len(parts) > 6 else ''
                        if st_fips in target_state_fips and ('0041' in comm or 'CORN' in comm or 'CORN' in comm_name):
                            corn_records_all.append(parts)

    time.sleep(0.3)

out_summary = os.path.join(ext_dir, "USDA_RMA_Corn_Belt_Corn_Coverage_2002_2026.csv")
with open(out_summary, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow([
        "Year", "State_FIPS", "State_Abbr", "County_FIPS", "County_Name",
        "Commodity_Code", "Commodity_Name", "Insurance_Plan", "Coverage_Type",
        "Coverage_Level", "Delivery_Type", "Policies_Sold", "Policies_Earning_Prem",
        "Policies_Indemnified", "Net_Planted_Acres", "Endorsed_Acres", "Liability",
        "Total_Premium", "Subsidy", "Indemnity_Amount", "Loss_Ratio"
    ])
    for r in corn_records_all:
        # truncate or pad to 21 columns
        row_clean = r[:21] + [''] * max(0, 21 - len(r))
        writer.writerow(row_clean)

print(f"RMA processing complete. Extracted {len(corn_records_all)} county-level corn insurance and acreage records to {out_summary}.")

