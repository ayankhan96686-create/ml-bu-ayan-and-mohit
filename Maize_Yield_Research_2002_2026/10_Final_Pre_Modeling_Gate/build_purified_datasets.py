import os
import pandas as pd
import numpy as np

gate_dir = r"E:\ml do not open\Maize_Yield_Research_2002_2026\10_Final_Pre_Modeling_Gate"
base_dir = r"E:\ml do not open\Maize_Yield_Research_2002_2026"

print("Building Purified Feature Tables with Clean Feature Engineering...")

# Load historical labeled feature table
p_ft = os.path.join(base_dir, "04_Integrated_Data", "County_Year_Feature_Table.csv")
df_ft = pd.read_csv(p_ft)
print(f"Loaded Feature Table: {df_ft.shape}")

# Load future 2026 feature table
p_fut = os.path.join(base_dir, "04_Integrated_Data", "Future_2026_Feature_Table.csv")
df_fut = pd.read_csv(p_fut)
print(f"Loaded Future 2026 Table: {df_fut.shape}")

# Load eligibility manifest
p_elig = os.path.join(gate_dir, "COUNTY_PREDICTION_ELIGIBILITY.csv")
df_elig = pd.read_csv(p_elig, dtype={'county_fips': str})
eligible_fips = set(df_elig[df_elig['eligible_for_2026_forecast'] == True]['county_fips'])
print(f"Eligible agricultural counties count: {len(eligible_fips)}")

# 1. Compute Strict Calendar-Aligned Historical Yield Lags
# Load USDA Cleaned Yield for all years 2002-2025
p_yield = os.path.join(base_dir, "03_Cleaned_Data", "USDA_Yield_Cleaned.csv")
df_y = pd.read_csv(p_yield, dtype={'county_fips': str})

all_fips = sorted(df_ft['county_fips'].astype(str).unique())
years = np.arange(2002, 2027) # 2002 through 2026

spine = pd.MultiIndex.from_product([all_fips, years], names=['county_fips', 'year']).to_frame().reset_index(drop=True)
spine = spine.merge(df_y[['county_fips', 'year', 'corn_yield_bu_acre']], on=['county_fips', 'year'], how='left')
spine = spine.sort_values(['county_fips', 'year']).reset_index(drop=True)

# Strict Calendar Lags
spine['yield_lag_1yr'] = spine.groupby('county_fips')['corn_yield_bu_acre'].shift(1)
spine['yield_lag_2yr'] = spine.groupby('county_fips')['corn_yield_bu_acre'].shift(2)
# Rolling 5-year historical mean (excluding current year, requiring >=3 observed years)
spine['yield_roll_mean_5yr'] = spine.groupby('county_fips')['corn_yield_bu_acre'].transform(
    lambda s: s.shift(1).rolling(5, min_periods=3).mean()
)

# Merge lags onto historical feature table
df_ft['county_fips_str'] = df_ft['county_fips'].astype(str).str.zfill(5)
spine['county_fips_str'] = spine['county_fips'].astype(str).str.zfill(5)

df_ft_clean = df_ft.merge(
    spine[['county_fips_str', 'year', 'yield_lag_1yr', 'yield_lag_2yr', 'yield_roll_mean_5yr']],
    on=['county_fips_str', 'year'],
    how='left'
).drop(columns=['county_fips_str'])

# Merge lags onto Future 2026 table
df_fut['county_fips_str'] = df_fut['county_fips'].astype(str).str.zfill(5)
df_fut_clean = df_fut.merge(
    spine[['county_fips_str', 'year', 'yield_lag_1yr', 'yield_lag_2yr', 'yield_roll_mean_5yr']],
    on=['county_fips_str', 'year'],
    how='left'
).drop(columns=['county_fips_str'])

# 2. Add RMA Missing Indicator
df_ft_clean['rma_missing_indicator'] = df_ft_clean['rma_policies_sold'].isna().astype(int)
df_fut_clean['rma_missing_indicator'] = df_fut_clean['rma_policies_sold'].isna().astype(int)

# 3. Add County Prediction Eligibility Flag
df_ft_clean['eligible_for_2026_forecast'] = True # All labeled historical rows are in eligible counties
df_fut_clean['eligible_for_2026_forecast'] = df_fut_clean['county_fips'].astype(str).str.zfill(5).isin(eligible_fips)

# 4. Strictly EXCLUDE cv_percent from feature tables
if 'cv_percent' in df_ft_clean.columns:
    df_ft_clean = df_ft_clean.drop(columns=['cv_percent'])
if 'cv_percent' in df_fut_clean.columns:
    df_fut_clean = df_fut_clean.drop(columns=['cv_percent'])

# Reorder columns logically:
# 1. Administrative / Keys
# 2. Target (for historical)
# 3. Eligibility Flag
# 4. Strict Environmental Predictors (Soil, Climate NOAA, Climate PRISM)
# 5. In-Season Crop Condition & Progress
# 6. Pre-Harvest RMA Management
# 7. Historical Yield Lags
id_cols = ['county_fips', 'state_fips', 'state', 'state_name', 'county_name', 'year', 'eligible_for_2026_forecast']
target_col = ['corn_yield_bu_acre']
flag_cols = ['prism_valid_coverage_pct', 'rma_missing_indicator']

feature_cols_only = [c for c in df_ft_clean.columns if c not in id_cols + target_col + flag_cols]

final_train_cols = id_cols[:6] + target_col + ['eligible_for_2026_forecast'] + flag_cols + sorted(feature_cols_only)
final_fut_cols = id_cols[:6] + ['eligible_for_2026_forecast'] + flag_cols + sorted(feature_cols_only)

df_ft_clean = df_ft_clean[final_train_cols]
df_fut_clean = df_fut_clean[final_fut_cols]

p_train_out = os.path.join(gate_dir, "Purified_Corn_Belt_Historical_Dataset.csv")
p_fut_out = os.path.join(gate_dir, "Purified_Corn_Belt_Future_2026_Dataset.csv")

df_ft_clean.to_csv(p_train_out, index=False)
df_fut_clean.to_csv(p_fut_out, index=False)

print(f"\nSaved Purified Historical Dataset: {df_ft_clean.shape} -> {p_train_out}")
print(f"Saved Purified Future 2026 Dataset: {df_fut_clean.shape} -> {p_fut_out}")
print(f"Historical 'cv_percent' present? {'cv_percent' in df_ft_clean.columns} (Must be False)")
print(f"Future 'cv_percent' present? {'cv_percent' in df_fut_clean.columns} (Must be False)")
print(f"Future eligible counties count: {df_fut_clean['eligible_for_2026_forecast'].sum()} of {len(df_fut_clean)}")

print("\nPurified dataset engineering complete.")
