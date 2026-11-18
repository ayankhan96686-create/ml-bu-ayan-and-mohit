import os
import glob
import struct
import csv
import pandas as pd
import numpy as np
from PIL import Image

base_dir = r"E:\ml do not open\Maize_Yield_Research_2002_2026"
prism_dir = os.path.join(base_dir, "02_Extracted_Data", "03_PRISM_Weather")
counties_ref_csv = os.path.join(base_dir, "00_Project_Admin", "Study_Area_Counties_FIPS.csv")
county_dbf_path = os.path.join(base_dir, "02_Extracted_Data", "14_County_and_State_Boundaries", "tl_2024_us_county", "tl_2024_us_county.dbf")
out_dir = os.path.join(base_dir, "03_Cleaned_Data")
os.makedirs(out_dir, exist_ok=True)
out_file = os.path.join(out_dir, "PRISM_Climate_Cleaned.csv")

print("Executing Phase 3 (Part B): PRISM Gridded Climate Zonal Statistics & Cleaning...")

# 1. Load official 814 study counties
df_counties = pd.read_csv(counties_ref_csv, dtype=str)
valid_fips = set(df_counties['Full_FIPS'].str.strip())
fips_to_meta = df_counties.set_index('Full_FIPS')[['State_Name', 'County_Name']].to_dict('index')

# 2. Parse county centroids (INTPTLAT, INTPTLON) from official DBF
print("Reading county centroids from US Census DBF...")
county_coords = {}
with open(county_dbf_path, 'rb') as f:
    header = f.read(32)
    num_records, header_len, record_len = struct.unpack('<IHH', header[4:12])
    fields = []
    while True:
        field_desc = f.read(32)
        if field_desc[0] == 0x0D:
            break
        name = field_desc[:11].replace(b'\x00', b'').decode('ascii')
        typ = chr(field_desc[11])
        length = field_desc[16]
        fields.append((name, typ, length))
        
    f.seek(header_len)
    for _ in range(num_records):
        rec_bytes = f.read(record_len)
        if len(rec_bytes) < record_len:
            break
        offset = 1 # skip deletion flag
        rec_dict = {}
        for fname, ftyp, flen in fields:
            val = rec_bytes[offset:offset+flen].decode('latin1', errors='ignore').strip()
            rec_dict[fname] = val
            offset += flen
            
        st = rec_dict.get('STATEFP', '').zfill(2)
        cty = rec_dict.get('COUNTYFP', '').zfill(3)
        geoid = st + cty
        if geoid in valid_fips:
            try:
                lat = float(rec_dict.get('INTPTLAT', '0'))
                lon = float(rec_dict.get('INTPTLON', '0'))
                county_coords[geoid] = (lat, lon)
            except:
                pass

print(f"Mapped {len(county_coords)} study county geographic centroids.")

# PRISM Grid Parameters
# Ulxmap = -125.0, Ulymp = 49.916666666667, cellsize = 0.0416666666667
ULX = -125.0
ULY = 49.916666666667
CELL = 0.0416666666667
NROWS = 621
NCOLS = 1405
NODATA = -9999.0

# Precompute county pixel coordinates and 3x3 window slices
county_pixels = {}
for fips, (lat, lon) in county_coords.items():
    c = int((lon - ULX) / CELL)
    r = int((ULY - lat) / CELL)
    county_pixels[fips] = (r, c)

def get_zonal_stat(arr, r, c, window=1):
    r_min = max(0, r - window)
    r_max = min(NROWS, r + window + 1)
    c_min = max(0, c - window)
    c_max = min(NCOLS, c + window + 1)
    sub = arr[r_min:r_max, c_min:c_max]
    valid = sub[(sub > -9000.0) & (~np.isnan(sub))]
    total_pixels = sub.size
    coverage_pct = round(len(valid) / total_pixels * 100.0, 1) if total_pixels > 0 else 0.0
    val_mean = float(valid.mean()) if len(valid) > 0 else np.nan
    return val_mean, coverage_pct

# Process each year (2002 to 2025 annual grids)
elements = ['ppt', 'tmean', 'tmin', 'tmax']
years = [str(y) for y in range(2002, 2026)]

prism_data_by_year = {}

for yr in years:
    print(f"Processing PRISM rasters for year {yr}...")
    yr_grids = {}
    for elem in elements:
        tif_pattern = os.path.join(prism_dir, elem, yr, f"prism_{elem}_us_25m_{yr}.tif")
        matched = glob.glob(tif_pattern)
        if matched:
            im = Image.open(matched[0])
            yr_grids[elem] = np.array(im, dtype=np.float32)
        else:
            yr_grids[elem] = None
            
    for fips in sorted(county_coords.keys()):
        r, c = county_pixels[fips]
        st_name = fips_to_meta[fips]['State_Name']
        cty_name = fips_to_meta[fips]['County_Name']
        
        row_dict = {
            'county_fips': fips,
            'state': st_name,
            'county': cty_name,
            'year': int(yr)
        }
        
        coverages = []
        for elem in elements:
            grid = yr_grids.get(elem)
            if grid is not None:
                m_val, cov = get_zonal_stat(grid, r, c, window=1)
                row_dict[f'prism_{elem}'] = round(m_val, 2) if not np.isnan(m_val) else np.nan
                coverages.append(cov)
            else:
                row_dict[f'prism_{elem}'] = np.nan
                coverages.append(0.0)
                
        row_dict['prism_valid_coverage_pct'] = round(np.mean(coverages), 1) if coverages else 0.0
        prism_data_by_year[(fips, int(yr))] = row_dict

# Process 2026 provisional growing season (months 202605 to 202610)
print("Processing 2026 provisional growing season PRISM monthly grids...")
months_2026 = [f"2026{m:02d}" for m in range(5, 11)]

grids_2026 = {elem: [] for elem in elements}
for elem in elements:
    for ym in months_2026:
        pattern = os.path.join(prism_dir, f"{elem}_2026_provisional", ym, f"prism_{elem}_us_25m_{ym}.tif")
        matched = glob.glob(pattern)
        if matched:
            im = Image.open(matched[0])
            grids_2026[elem].append(np.array(im, dtype=np.float32))

for fips in sorted(county_coords.keys()):
    r, c = county_pixels[fips]
    st_name = fips_to_meta[fips]['State_Name']
    cty_name = fips_to_meta[fips]['County_Name']
    
    # 2026 PPT: sum of May to October monthly grids
    ppt_monthly_means = [get_zonal_stat(g, r, c, window=1)[0] for g in grids_2026['ppt']]
    ppt_sum = sum(v for v in ppt_monthly_means if not np.isnan(v)) if ppt_monthly_means else np.nan
    
    tmean_monthly_means = [get_zonal_stat(g, r, c, window=1)[0] for g in grids_2026['tmean']]
    tmean_avg = np.mean([v for v in tmean_monthly_means if not np.isnan(v)]) if tmean_monthly_means else np.nan
    
    tmin_monthly_means = [get_zonal_stat(g, r, c, window=1)[0] for g in grids_2026['tmin']]
    tmin_avg = np.mean([v for v in tmin_monthly_means if not np.isnan(v)]) if tmin_monthly_means else np.nan
    
    tmax_monthly_means = [get_zonal_stat(g, r, c, window=1)[0] for g in grids_2026['tmax']]
    tmax_avg = np.mean([v for v in tmax_monthly_means if not np.isnan(v)]) if tmax_monthly_means else np.nan
    
    prism_data_by_year[(fips, 2026)] = {
        'county_fips': fips,
        'state': st_name,
        'county': cty_name,
        'year': 2026,
        'prism_ppt': round(ppt_sum, 2) if not np.isnan(ppt_sum) else np.nan, # provisional growing season sum
        'prism_tmean': round(tmean_avg, 2) if not np.isnan(tmean_avg) else np.nan,
        'prism_tmin': round(tmin_avg, 2) if not np.isnan(tmin_avg) else np.nan,
        'prism_tmax': round(tmax_avg, 2) if not np.isnan(tmax_avg) else np.nan,
        'prism_valid_coverage_pct': 100.0
    }

# Assemble into DataFrame
df_prism_clean = pd.DataFrame(list(prism_data_by_year.values())).sort_values(['county_fips', 'year']).reset_index(drop=True)
# Rename columns to clearly state units
df_prism_clean.rename(columns={
    'prism_ppt': 'prism_ppt_mm',
    'prism_tmean': 'prism_tmean_c',
    'prism_tmin': 'prism_tmin_c',
    'prism_tmax': 'prism_tmax_c'
}, inplace=True)

df_prism_clean.to_csv(out_file, index=False)
print(f"Saved clean PRISM dataset to: {out_file}")
print(f"Total rows: {len(df_prism_clean):,}")
print(f"Unique counties: {df_prism_clean['county_fips'].nunique()}")
print(f"Years: {df_prism_clean['year'].min()} to {df_prism_clean['year'].max()} ({df_prism_clean['year'].nunique()} years)")
print("PRISM summary statistics:")
print(df_prism_clean[['prism_ppt_mm', 'prism_tmean_c', 'prism_tmin_c', 'prism_tmax_c', 'prism_valid_coverage_pct']].describe().round(2))

