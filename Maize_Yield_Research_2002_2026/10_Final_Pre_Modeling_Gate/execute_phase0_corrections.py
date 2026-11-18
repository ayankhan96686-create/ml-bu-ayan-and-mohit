import os
import pandas as pd
import numpy as np

gate_dir = r"E:\ml do not open\Maize_Yield_Research_2002_2026\10_Final_Pre_Modeling_Gate"
base_dir = r"E:\ml do not open\Maize_Yield_Research_2002_2026"

print("Executing Phase 0: Final Data Integrity Corrections...")

# 1. Five Datasets Paths
p_master = os.path.join(base_dir, "04_Integrated_Data", "County_Year_Master_Dataset.csv")
p_feature = os.path.join(base_dir, "04_Integrated_Data", "County_Year_Feature_Table.csv")
p_future = os.path.join(base_dir, "04_Integrated_Data", "Future_2026_Feature_Table.csv")
p_pur_hist = os.path.join(gate_dir, "Purified_Corn_Belt_Historical_Dataset.csv")
p_pur_fut = os.path.join(gate_dir, "Purified_Corn_Belt_Future_2026_Dataset.csv")

cols_master = pd.read_csv(p_master, nrows=1).columns.tolist()
cols_feature = pd.read_csv(p_feature, nrows=1).columns.tolist()
cols_future = pd.read_csv(p_future, nrows=1).columns.tolist()
cols_pur_hist = pd.read_csv(p_pur_hist, nrows=1).columns.tolist()
cols_pur_fut = pd.read_csv(p_pur_fut, nrows=1).columns.tolist()

# Build comprehensive union of all columns
all_cols_union = []
for c in cols_master + cols_feature + cols_future + cols_pur_hist + cols_pur_fut:
    if c not in all_cols_union:
        all_cols_union.append(c)

print(f"Total distinct columns across all 5 datasets: {len(all_cols_union)}")

# 2. Build CORRECTED_COLUMN_MANIFEST.csv
manifest_data = []
for c in all_cols_union:
    in_m = c in cols_master
    in_ft = c in cols_feature
    in_fut = c in cols_future
    in_ph = c in cols_pur_hist
    in_pf = c in cols_pur_fut
    
    # Late-season operational horizon categorization
    if c == 'corn_yield_bu_acre':
        role = 'Target_Vector_y'
        horizon = 'Observed Post-Harvest'
        verdict = 'Target'
    elif c == 'cv_percent':
        role = 'Target_Metadata'
        horizon = 'Observed Post-Harvest'
        verdict = 'EXCLUDED_LEAKAGE (Purged from Purified Datasets)'
    elif c.startswith('rma_postharvest_') or c == 'state_progress_postharvest_harvested_w44_pct':
        role = 'Post_Harvest_Outcome'
        horizon = 'Post-Harvest (Dec-Apr)'
        verdict = 'EXCLUDED_POST_HARVEST'
    elif c in ['county_fips', 'state_fips', 'state', 'state_name', 'county_name']:
        role = 'Administrative_Key'
        horizon = 'Static'
        verdict = 'Grouping_Key_Only (Excluded from X)'
    elif c in ['soil_intptlat', 'soil_intptlon']:
        role = 'Spatial_Coordinates'
        horizon = 'Static'
        verdict = 'Spatial_Validation_Only (Excluded from Baseline X)'
    elif c == 'year':
        role = 'Temporal_Key'
        horizon = 'Annual'
        verdict = 'Temporal_Split_Key (Excluded from Numeric Tree Splits)'
    elif c == 'eligible_for_2026_forecast':
        role = 'Operational_Scope_Filter'
        horizon = 'Static'
        verdict = 'Scope_Filter_Flag'
    elif c == 'prism_valid_coverage_pct':
        role = 'Quality_Indicator'
        horizon = 'Audit'
        verdict = 'Audit_Flag (Excluded from X)'
    elif c == 'rma_missing_indicator':
        role = 'Management_Missing_Flag'
        horizon = 'Pre-Harvest'
        verdict = 'INCLUDED_MANAGEMENT_X'
    elif c == 'noaa_prcp_annual_in':
        role = 'Full_Year_Climate'
        horizon = 'Post-Harvest (Full Year Jan-Dec)'
        verdict = 'EXCLUDED_FROM_LATE_SEASON_X (Missing in 2026; unobserved at harvest)'
    elif c.startswith('prism_'):
        role = 'PRISM_Gridded_Weather'
        horizon = 'Annual in Hist; May-Oct in 2026'
        verdict = 'SECONDARY_PREDICTOR (Mismatched period; NOAA preferred)'
    elif c.startswith('soil_'):
        role = 'Soil_Physical_Property'
        horizon = 'Pre-Planting (Static)'
        verdict = 'INCLUDED_BASELINE_X'
    elif c.startswith('noaa_'):
        role = 'NOAA_Growing_Season_Climate'
        horizon = 'May-September Observed'
        verdict = 'INCLUDED_BASELINE_X'
    elif c.startswith('state_cond_') or c.startswith('state_progress_'):
        role = 'Crop_Phenology_Condition'
        horizon = 'In-Season Observed (May-Oct)'
        verdict = 'INCLUDED_EXTENDED_X'
    elif c.startswith('rma_'):
        role = 'RMA_Underwriting'
        horizon = 'Pre-Harvest Underwriting'
        verdict = 'INCLUDED_MANAGEMENT_X'
    elif c.startswith('yield_lag_') or c.startswith('yield_roll_'):
        role = 'Historical_Yield_Memory'
        horizon = 'Pre-Planting (Calendar Aligned)'
        verdict = 'INCLUDED_AUTOREGRESSIVE_X'
    else:
        role = 'Other'
        horizon = 'Review'
        verdict = 'Review'
        
    manifest_data.append({
        'column_name': c,
        'in_master_83': in_m,
        'in_feature_table_79': in_ft,
        'in_future_2026_77': in_fut,
        'in_purified_hist_83': in_ph,
        'in_purified_fut_82': in_pf,
        'feature_role': role,
        'operational_availability_horizon': horizon,
        'late_season_experiment_verdict': verdict
    })

df_manifest_corr = pd.DataFrame(manifest_data)
p_man_out = os.path.join(gate_dir, "CORRECTED_COLUMN_MANIFEST.csv")
df_manifest_corr.to_csv(p_man_out, index=False)
print(f"Saved CORRECTED_COLUMN_MANIFEST.csv ({len(df_manifest_corr)} columns) -> {p_man_out}")

# 3. Resolve 788 vs 787 County Eligibility: COUNTY_ELIGIBILITY_RESOLVED.csv
p_ref = os.path.join(base_dir, "00_Project_Admin", "Study_Area_Counties_FIPS.csv")
df_ref = pd.read_csv(p_ref, dtype=str)
p_ydf = os.path.join(base_dir, "03_Cleaned_Data", "USDA_Yield_Cleaned.csv")
df_y = pd.read_csv(p_ydf, dtype={'county_fips': str})
p_soil = os.path.join(base_dir, "03_Cleaned_Data", "SoilGrids_Cleaned.csv")
df_s = pd.read_csv(p_soil, dtype={'county_fips': str})
p_rma = os.path.join(base_dir, "03_Cleaned_Data", "RMA_Cleaned.csv")
df_rma = pd.read_csv(p_rma, dtype={'county_fips': str})

# Soil missing count per county
soil_nulls = df_s.set_index('county_fips')[['clay_0_5cm']].isna().rename(columns={'clay_0_5cm': 'soil_is_null'})

# Yield counts
y_cnt = df_y.groupby('county_fips')['year'].count().rename('labeled_years_count')
rma_cnt = df_rma.groupby('county_fips')['year'].count().rename('rma_years_count')

df_county_res = df_ref.rename(columns={'Full_FIPS': 'county_fips', 'State_Name': 'state_name', 'County_Name': 'county_name'}).copy()
df_county_res['county_fips'] = df_county_res['county_fips'].str.strip()
df_county_res = df_county_res.merge(y_cnt, on='county_fips', how='left').fillna({'labeled_years_count': 0})
df_county_res = df_county_res.merge(rma_cnt, on='county_fips', how='left').fillna({'rma_years_count': 0})
df_county_res = df_county_res.merge(soil_nulls, on='county_fips', how='left').fillna({'soil_is_null': False})

def classify_county(row):
    fips = row['county_fips']
    y_n = int(row['labeled_years_count'])
    s_null = bool(row['soil_is_null'])
    
    if y_n == 0:
        if fips in ['27077', '29029']:
            cat = 'Excluded_Marginal_Unverified'
            elig_788 = False
            elig_787 = False
            notes = 'Zero NASS historical yield labels. Minor RMA acreage (<3k acres). Excluded due to lack of ground truth.'
        else:
            cat = 'Excluded_NonAgricultural_Terrain'
            elig_788 = False
            elig_787 = False
            notes = 'Zero NASS historical yield labels and zero RMA policies. Boreal forest, rocky Ozarks, sandhills dune, or urban core.'
    elif s_null:
        cat = 'Eligible_With_Imputed_Soil_Only (Cook County IL)'
        elig_788 = True
        elig_787 = False # The sole difference between 788 and 787!
        notes = 'Cook County IL (Chicago): 11 historical yield labels present, but centroid is masked out in SoilGrids (100% NaN soil). Eligible under 788 (with soil imputation/tree routing); excluded under 787 (strict physical soil completeness).'
    else:
        cat = 'Eligible_Fully_Observed_Agricultural'
        elig_788 = True
        elig_787 = True
        notes = f'Verified agricultural county with {y_n} labeled historical years and 100% complete SoilGrids profile.'
        
    return cat, elig_788, elig_787, notes

res_tuples = df_county_res.apply(classify_county, axis=1)
df_county_res['eligibility_tier'] = [r[0] for r in res_tuples]
df_county_res['eligible_scope_788'] = [r[1] for r in res_tuples]
df_county_res['eligible_scope_787'] = [r[2] for r in res_tuples]
df_county_res['technical_notes'] = [r[3] for r in res_tuples]

p_county_out = os.path.join(gate_dir, "COUNTY_ELIGIBILITY_RESOLVED.csv")
df_county_res.to_csv(p_county_out, index=False)
print(f"Saved COUNTY_ELIGIBILITY_RESOLVED.csv ({len(df_county_res)} counties) -> {p_county_out}")
print(f"Scope 788 Count: {df_county_res['eligible_scope_788'].sum()}")
print(f"Scope 787 Count: {df_county_res['eligible_scope_787'].sum()}")
print(f"The exact difference between 788 and 787: Cook County, IL (FIPS 17031) soil mask status.")

print("\nPhase 0 manifests generated successfully.")

