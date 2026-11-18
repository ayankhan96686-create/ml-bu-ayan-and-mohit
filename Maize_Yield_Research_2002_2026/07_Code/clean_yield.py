import os
import pandas as pd
import numpy as np

base_dir = r"E:\ml do not open\Maize_Yield_Research_2002_2026"
raw_yield_file = os.path.join(base_dir, "01_Raw_Data", "01_USDA_NASS_Yield", "USDA_NASS_Corn_Grain_Yield_9States_2002_2026_723B6328-B866-3F54-AB6D-07F62EE064E9.csv")
out_dir = os.path.join(base_dir, "03_Cleaned_Data")
os.makedirs(out_dir, exist_ok=True)
out_file = os.path.join(out_dir, "USDA_Yield_Cleaned.csv")

print("Executing Phase 2: Primary Target Dataset Cleaning (USDA NASS Corn Grain Yield)...")

# 1. Load raw data
df = pd.read_csv(raw_yield_file, dtype=str)
n_raw = len(df)
print(f"Loaded raw records: {n_raw:,}")

# 2. Filter strictly for corn grain yield
df = df[df['Commodity'].str.upper() == 'CORN'].copy()
df = df[df['Data Item'].str.upper() == 'CORN, GRAIN - YIELD, MEASURED IN BU / ACRE'].copy()
print(f"Records matching 'CORN, GRAIN - YIELD, MEASURED IN BU / ACRE': {len(df):,}")

# 3. Filter genuine county observations (exclude 'OTHER COUNTIES' residuals)
df_genuine = df[df['County ANSI'].notnull() & (df['County ANSI'].str.strip() != '')].copy()
n_excluded_residuals = len(df) - len(df_genuine)
print(f"Excluded residual 'OTHER COUNTIES' state balance records: {n_excluded_residuals:,}")
print(f"Genuine county records retained: {len(df_genuine):,}")

# 4. Standardize geographic identifiers (construct 5-digit county FIPS)
df_genuine['state_fips'] = df_genuine['State ANSI'].str.strip().str.zfill(2)
df_genuine['county_fips_3'] = df_genuine['County ANSI'].str.strip().str.zfill(3)
df_genuine['county_fips'] = df_genuine['state_fips'] + df_genuine['county_fips_3']
df_genuine['state'] = df_genuine['State'].str.strip().str.title()
df_genuine['county'] = df_genuine['County'].str.strip().str.title()

# 5. Standardize temporal identifiers
df_genuine['year'] = pd.to_numeric(df_genuine['Year'].str.strip(), errors='raise').astype(int)

# 6. Standardize target numeric yield
df_genuine['corn_yield_bu_acre'] = pd.to_numeric(df_genuine['Value'].str.replace(',', '', regex=False).str.strip(), errors='raise')

# 7. Standardize CV (%)
df_genuine['cv_percent'] = pd.to_numeric(df_genuine['CV (%)'].str.replace(',', '', regex=False).str.strip(), errors='coerce')

# 8. Check for duplicates
dup_mask = df_genuine.duplicated(subset=['county_fips', 'year'], keep=False)
if dup_mask.any():
    print(f"WARNING: Found {dup_mask.sum()} duplicate county-year records! Resolving...")
    df_genuine = df_genuine.drop_duplicates(subset=['county_fips', 'year'], keep='first')
else:
    print("Verification: Zero duplicate (county_fips, year) records found.")

# 9. Format final clean schema
final_cols = ['county_fips', 'state', 'county', 'year', 'corn_yield_bu_acre', 'cv_percent']
df_clean = df_genuine[final_cols].sort_values(['county_fips', 'year']).reset_index(drop=True)

# 10. Save cleaned table
df_clean.to_csv(out_file, index=False)
print(f"Saved cleaned target dataset to: {out_file}")
print(f"Final usable records: {len(df_clean):,}")
print(f"Unique counties: {df_clean['county_fips'].nunique()}")
print(f"Years: {df_clean['year'].min()} to {df_clean['year'].max()} ({df_clean['year'].nunique()} years)")
print("Summary statistics of corn_yield_bu_acre:")
print(df_clean['corn_yield_bu_acre'].describe().round(2))

