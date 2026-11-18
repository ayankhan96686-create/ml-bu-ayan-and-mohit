import os
import csv
import pandas as pd
import numpy as np

base_dir = r"E:\ml do not open\Maize_Yield_Research_2002_2026"
raw_noaa_dir = os.path.join(base_dir, "01_Raw_Data", "05_NOAA_Weather")
counties_ref_csv = os.path.join(base_dir, "00_Project_Admin", "Study_Area_Counties_FIPS.csv")
out_dir = os.path.join(base_dir, "03_Cleaned_Data")
os.makedirs(out_dir, exist_ok=True)
out_file = os.path.join(out_dir, "NOAA_Climate_Cleaned.csv")

print("Executing Phase 3: NOAA Climate Data Cleaning & Feature Engineering...")

# 1. Load official 814 study counties
df_counties = pd.read_csv(counties_ref_csv, dtype=str)
valid_fips = set(df_counties['Full_FIPS'].str.strip())
fips_to_names = df_counties.set_index('Full_FIPS')[['State_Name', 'County_Name']].to_dict('index')
print(f"Loaded {len(valid_fips)} official Corn Belt study counties from reference registry.")

# NOAA CIRS to State FIPS crosswalk
noaa_to_fips = {
    '11': '17', # Illinois
    '12': '18', # Indiana
    '13': '19', # Iowa
    '21': '27', # Minnesota
    '23': '29', # Missouri
    '25': '31', # Nebraska
    '33': '39', # Ohio
    '39': '46', # South Dakota
    '47': '55'  # Wisconsin
}

# Elements: 01=pcpn, 02=tmpc, 27=tmax, 28=tmin
elements = {
    'climdiv-pcpncy-v1.0.0-20261006': 'pcpn',
    'climdiv-tmpccy-v1.0.0-20261006': 'tmpc',
    'climdiv-tmaxcy-v1.0.0-20261006': 'tmax',
    'climdiv-tmincy-v1.0.0-20261006': 'tmin'
}

data_by_element = {}

for fname, elem in elements.items():
    fpath = os.path.join(raw_noaa_dir, fname)
    print(f"Parsing raw NOAA {elem} from {fname}...")
    records = {}
    with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if len(line) < 11:
                continue
            noaa_st = line[:2]
            if noaa_st not in noaa_to_fips:
                continue
            st_fips = noaa_to_fips[noaa_st]
            cty_fips = line[2:5]
            full_fips = st_fips + cty_fips
            if full_fips not in valid_fips:
                continue
            year = int(line[7:11])
            if year < 2002 or year > 2026:
                continue
            
            vals = line[11:].split()
            # Month values: 1 to 12
            # Replace sentinels (-9.99, -99.99) with np.nan
            m_vals = []
            for v in vals[:12]:
                try:
                    fv = float(v)
                    if fv <= -9.0:
                        m_vals.append(np.nan)
                    else:
                        m_vals.append(fv)
                except:
                    m_vals.append(np.nan)
            while len(m_vals) < 12:
                m_vals.append(np.nan)
                
            records[(full_fips, year)] = m_vals
    print(f"  Extracted {len(records):,} county-year series for {elem}.")
    data_by_element[elem] = records

# Combine into master climate records
combined_rows = []
all_keys = set()
for elem in elements.values():
    all_keys.update(data_by_element[elem].keys())

print(f"Merging elements across {len(all_keys):,} total (county_fips, year) pairs...")

for (fips, yr) in sorted(all_keys):
    st_name = fips_to_names[fips]['State_Name']
    cty_name = fips_to_names[fips]['County_Name']
    
    # Defaults
    pcpn_m = data_by_element.get('pcpn', {}).get((fips, yr), [np.nan]*12)
    tmpc_m = data_by_element.get('tmpc', {}).get((fips, yr), [np.nan]*12)
    tmax_m = data_by_element.get('tmax', {}).get((fips, yr), [np.nan]*12)
    tmin_m = data_by_element.get('tmin', {}).get((fips, yr), [np.nan]*12)
    
    # Engineered features
    # May=index 4, Jun=5, Jul=6, Aug=7, Sep=8, Oct=9
    prcp_may = pcpn_m[4]
    prcp_jun = pcpn_m[5]
    prcp_jul = pcpn_m[6]
    prcp_aug = pcpn_m[7]
    prcp_sep = pcpn_m[8]
    
    # Growing season precipitation
    may_aug_prcp_list = [pcpn_m[i] for i in [4, 5, 6, 7] if not np.isnan(pcpn_m[i])]
    prcp_may_aug = sum(may_aug_prcp_list) if len(may_aug_prcp_list) == 4 else np.nan
    
    may_sep_prcp_list = [pcpn_m[i] for i in [4, 5, 6, 7, 8] if not np.isnan(pcpn_m[i])]
    prcp_may_sep = sum(may_sep_prcp_list) if len(may_sep_prcp_list) == 5 else np.nan
    
    annual_prcp_list = [p for p in pcpn_m if not np.isnan(p)]
    prcp_annual = sum(annual_prcp_list) if len(annual_prcp_list) == 12 else np.nan
    
    # Temperatures
    tmean_may_aug_list = [tmpc_m[i] for i in [4, 5, 6, 7] if not np.isnan(tmpc_m[i])]
    tmean_may_aug = np.mean(tmean_may_aug_list) if len(tmean_may_aug_list) == 4 else np.nan
    
    tmax_jul = tmax_m[6]
    tmax_aug = tmax_m[7]
    tmax_may_aug_list = [tmax_m[i] for i in [4, 5, 6, 7] if not np.isnan(tmax_m[i])]
    tmax_may_aug = np.mean(tmax_may_aug_list) if len(tmax_may_aug_list) == 4 else np.nan
    
    tmin_may_aug_list = [tmin_m[i] for i in [4, 5, 6, 7] if not np.isnan(tmin_m[i])]
    tmin_may_aug = np.mean(tmin_may_aug_list) if len(tmin_may_aug_list) == 4 else np.nan
    
    # Heat stress indicators
    # High July-August max temperature > 90°F
    heat_stress_jul_aug = max(tmax_jul if not np.isnan(tmax_jul) else 0, tmax_aug if not np.isnan(tmax_aug) else 0)
    
    row = {
        'county_fips': fips,
        'state': st_name,
        'county': cty_name,
        'year': yr,
        'noaa_prcp_may_in': round(prcp_may, 2) if not np.isnan(prcp_may) else np.nan,
        'noaa_prcp_jun_in': round(prcp_jun, 2) if not np.isnan(prcp_jun) else np.nan,
        'noaa_prcp_jul_in': round(prcp_jul, 2) if not np.isnan(prcp_jul) else np.nan,
        'noaa_prcp_aug_in': round(prcp_aug, 2) if not np.isnan(prcp_aug) else np.nan,
        'noaa_prcp_sep_in': round(prcp_sep, 2) if not np.isnan(prcp_sep) else np.nan,
        'noaa_prcp_may_aug_in': round(prcp_may_aug, 2) if not np.isnan(prcp_may_aug) else np.nan,
        'noaa_prcp_may_sep_in': round(prcp_may_sep, 2) if not np.isnan(prcp_may_sep) else np.nan,
        'noaa_prcp_annual_in': round(prcp_annual, 2) if not np.isnan(prcp_annual) else np.nan,
        'noaa_tmean_may_aug_f': round(tmean_may_aug, 2) if not np.isnan(tmean_may_aug) else np.nan,
        'noaa_tmax_jul_f': round(tmax_jul, 2) if not np.isnan(tmax_jul) else np.nan,
        'noaa_tmax_aug_f': round(tmax_aug, 2) if not np.isnan(tmax_aug) else np.nan,
        'noaa_tmax_may_aug_f': round(tmax_may_aug, 2) if not np.isnan(tmax_may_aug) else np.nan,
        'noaa_tmin_may_aug_f': round(tmin_may_aug, 2) if not np.isnan(tmin_may_aug) else np.nan,
        'noaa_heat_stress_max_f': round(heat_stress_jul_aug, 2) if heat_stress_jul_aug > 0 else np.nan
    }
    combined_rows.append(row)

df_clean_noaa = pd.DataFrame(combined_rows).sort_values(['county_fips', 'year']).reset_index(drop=True)
df_clean_noaa.to_csv(out_file, index=False)

print(f"Successfully saved clean NOAA climate dataset to: {out_file}")
print(f"Total cleaned rows: {len(df_clean_noaa):,}")
print(f"Unique counties: {df_clean_noaa['county_fips'].nunique()}")
print(f"Years: {df_clean_noaa['year'].min()} to {df_clean_noaa['year'].max()}")
print(f"Duplicate (county_fips, year) check: {df_clean_noaa.duplicated(subset=['county_fips', 'year']).sum()}")
print("Feature summary:")
print(df_clean_noaa[['noaa_prcp_may_aug_in', 'noaa_tmean_may_aug_f', 'noaa_tmax_jul_f', 'noaa_heat_stress_max_f']].describe().round(2))

