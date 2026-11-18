"""
07_Code/integrate_datasets.py
Integrates all cleaned domain datasets into unified analysis tables.
Produces:
1. 04_Integrated_Data/County_Year_Master_Dataset.csv (Full 2002-2026 master dataset with diagnostic & leakage variables)
2. 04_Integrated_Data/County_Year_Feature_Table.csv (2002-2025 ML-ready supervised learning table with genuine yield labels)
3. 04_Integrated_Data/Future_2026_Feature_Table.csv (2026 prospective unlabeled prediction table)
"""

import os
import pandas as pd
import numpy as np

PROJECT_ROOT = r"E:\ml do not open\Maize_Yield_Research_2002_2026"
CLEAN_DIR = os.path.join(PROJECT_ROOT, "03_Cleaned_Data")
OUT_DIR = os.path.join(PROJECT_ROOT, "04_Integrated_Data")
os.makedirs(OUT_DIR, exist_ok=True)

# 1. Base Spine: 814 counties x 25 years (2002-2026) = 20,350 county-years
fips_path = os.path.join(PROJECT_ROOT, "00_Project_Admin", "Study_Area_Counties_FIPS.csv")
df_counties = pd.read_csv(fips_path, dtype=str)

df_counties_clean = pd.DataFrame({
    "county_fips": df_counties["Full_FIPS"].str.zfill(5),
    "state_fips": df_counties["State_FIPS"].str.zfill(2),
    "state": df_counties["State_Abbr"].str.strip(),
    "state_name": df_counties["State_Name"].str.strip(),
    "county_name": df_counties["County_Name"].str.strip()
})

years = list(range(2002, 2027))
records_spine = []
for yr in years:
    c_copy = df_counties_clean.copy()
    c_copy["year"] = yr
    records_spine.append(c_copy)

spine = pd.concat(records_spine, ignore_index=True)
print(f"Base spine created: {spine.shape[0]} county-years (814 counties x 25 years).")

# 2. Merge USDA Corn Grain Yield
yield_path = os.path.join(CLEAN_DIR, "USDA_Yield_Cleaned.csv")
df_yield = pd.read_csv(yield_path, dtype={"county_fips": str, "year": int})
df_yield["county_fips"] = df_yield["county_fips"].str.zfill(5)
df_yield_sub = df_yield[["county_fips", "year", "corn_yield_bu_acre", "cv_percent"]]

merged = pd.merge(spine, df_yield_sub, on=["county_fips", "year"], how="left")
print(f"After Yield merge: {merged.shape[0]} rows. Labeled yield count: {merged['corn_yield_bu_acre'].notna().sum()}")

# 3. Merge NOAA Climate
noaa_path = os.path.join(CLEAN_DIR, "NOAA_Climate_Cleaned.csv")
df_noaa = pd.read_csv(noaa_path, dtype={"county_fips": str, "year": int})
df_noaa["county_fips"] = df_noaa["county_fips"].str.zfill(5)
noaa_feature_cols = [c for c in df_noaa.columns if c.startswith("noaa_")]
df_noaa_sub = df_noaa[["county_fips", "year"] + noaa_feature_cols]

merged = pd.merge(merged, df_noaa_sub, on=["county_fips", "year"], how="left")
print(f"After NOAA merge: {merged.shape[0]} rows. NOAA features: {len(noaa_feature_cols)}")

# 4. Merge PRISM Climate
prism_path = os.path.join(CLEAN_DIR, "PRISM_Climate_Cleaned.csv")
df_prism = pd.read_csv(prism_path, dtype={"county_fips": str, "year": int})
df_prism["county_fips"] = df_prism["county_fips"].str.zfill(5)
prism_feature_cols = [c for c in df_prism.columns if c.startswith("prism_")]
df_prism_sub = df_prism[["county_fips", "year"] + prism_feature_cols]

merged = pd.merge(merged, df_prism_sub, on=["county_fips", "year"], how="left")
print(f"After PRISM merge: {merged.shape[0]} rows. PRISM features: {len(prism_feature_cols)}")

# 5. Merge SoilGrids (Static soil profile broadcast per county)
soil_path = os.path.join(CLEAN_DIR, "SoilGrids_Cleaned.csv")
df_soil = pd.read_csv(soil_path, dtype={"county_fips": str})
df_soil["county_fips"] = df_soil["county_fips"].str.zfill(5)

# Calculate weighted 0-30cm profile means (depths: 0-5cm wt=5/30, 5-15cm wt=10/30, 15-30cm wt=15/30)
df_soil["soil_clay_mean_0_30cm"] = (df_soil["clay_0_5cm"]*5 + df_soil["clay_5_15cm"]*10 + df_soil["clay_15_30cm"]*15)/30.0
df_soil["soil_sand_mean_0_30cm"] = (df_soil["sand_0_5cm"]*5 + df_soil["sand_5_15cm"]*10 + df_soil["sand_15_30cm"]*15)/30.0
df_soil["soil_silt_mean_0_30cm"] = (df_soil["silt_0_5cm"]*5 + df_soil["silt_5_15cm"]*10 + df_soil["silt_15_30cm"]*15)/30.0
df_soil["soil_soc_mean_0_30cm"]  = (df_soil["soc_0_5cm"]*5 + df_soil["soc_5_15cm"]*10 + df_soil["soc_15_30cm"]*15)/30.0
df_soil["soil_ph_mean_0_30cm"]   = (df_soil["phh2o_0_5cm"]*5 + df_soil["phh2o_5_15cm"]*10 + df_soil["phh2o_15_30cm"]*15)/30.0
df_soil["soil_bdod_mean_0_30cm"] = (df_soil["bdod_0_5cm"]*5 + df_soil["bdod_5_15cm"]*10 + df_soil["bdod_15_30cm"]*15)/30.0

soil_cols_all = [c for c in df_soil.columns if c not in ["county_fips", "state", "county"]]
# Rename with soil_ prefix for clarity
soil_rename = {c: f"soil_{c}" if not c.startswith("soil_") else c for c in soil_cols_all}
df_soil_renamed = df_soil[["county_fips"] + soil_cols_all].rename(columns=soil_rename)

merged = pd.merge(merged, df_soil_renamed, on=["county_fips"], how="left")
print(f"After SoilGrids merge: {merged.shape[0]} rows. Soil features: {len(soil_cols_all)}")

# 6. Merge RMA Insurance
rma_path = os.path.join(CLEAN_DIR, "RMA_Cleaned.csv")
df_rma = pd.read_csv(rma_path, dtype={"county_fips": str, "year": int})
df_rma["county_fips"] = df_rma["county_fips"].str.zfill(5)
rma_cols = [c for c in df_rma.columns if c.startswith("rma_")]
df_rma_sub = df_rma[["county_fips", "year"] + rma_cols]

merged = pd.merge(merged, df_rma_sub, on=["county_fips", "year"], how="left")
print(f"After RMA merge: {merged.shape[0]} rows. RMA features: {len(rma_cols)}")

# 7. Merge Crop Condition (State-level broadcast)
cond_path = os.path.join(CLEAN_DIR, "Crop_Condition_Cleaned.csv")
df_cond = pd.read_csv(cond_path, dtype={"state_fips": str, "year": int})
df_cond["state_fips"] = df_cond["state_fips"].str.zfill(2)
cond_cols = [c for c in df_cond.columns if c.startswith("state_cond_")]
df_cond_sub = df_cond[["state_fips", "year"] + cond_cols]

merged = pd.merge(merged, df_cond_sub, on=["state_fips", "year"], how="left")
print(f"After Crop Condition merge: {merged.shape[0]} rows. Condition features: {len(cond_cols)}")

# 8. Merge Crop Progress (State-level broadcast)
prog_path = os.path.join(CLEAN_DIR, "Crop_Progress_Cleaned.csv")
df_prog = pd.read_csv(prog_path, dtype={"state_fips": str, "year": int})
df_prog["state_fips"] = df_prog["state_fips"].str.zfill(2)
prog_cols = [c for c in df_prog.columns if c.startswith("state_progress_")]
df_prog_sub = df_prog[["state_fips", "year"] + prog_cols]

merged = pd.merge(merged, df_prog_sub, on=["state_fips", "year"], how="left")
print(f"After Crop Progress merge: {merged.shape[0]} rows. Progress features: {len(prog_cols)}")

# Sort comprehensively
merged = merged.sort_values(["state", "county_name", "year"]).reset_index(drop=True)

# -------------------------------------------------------------
# SAVE 1: Full Master Dataset (with all diagnostics & leakage vars)
# -------------------------------------------------------------
master_path = os.path.join(OUT_DIR, "County_Year_Master_Dataset.csv")
merged.round(4).to_csv(master_path, index=False)
print(f"\n[1] Saved County_Year_Master_Dataset.csv: {merged.shape[0]} rows, {merged.shape[1]} columns.")

# -------------------------------------------------------------
# SAVE 2: ML-Ready Feature Table (2002-2025, genuine yield labels, NO LEAKAGE)
# -------------------------------------------------------------
# Identify leakage columns to exclude from predictive modeling
leakage_cols = [
    "rma_postharvest_policies_indemnified",
    "rma_postharvest_indemnity_amount",
    "rma_postharvest_loss_ratio",
    "state_progress_postharvest_harvested_w44_pct"
]

# Historical labeled observations
ml_df = merged[(merged["year"] <= 2025) & (merged["corn_yield_bu_acre"].notna())].copy()
ml_feature_cols = [c for c in ml_df.columns if c not in leakage_cols]
ml_df = ml_df[ml_feature_cols].copy()

feature_path = os.path.join(OUT_DIR, "County_Year_Feature_Table.csv")
ml_df.round(4).to_csv(feature_path, index=False)
print(f"[2] Saved County_Year_Feature_Table.csv: {ml_df.shape[0]} rows, {ml_df.shape[1]} columns.")
print(f"    Years: {sorted(ml_df['year'].unique())}")
print(f"    Target column: 'corn_yield_bu_acre' (Min: {ml_df['corn_yield_bu_acre'].min():.1f}, Max: {ml_df['corn_yield_bu_acre'].max():.1f}, Mean: {ml_df['corn_yield_bu_acre'].mean():.1f} bu/acre)")

# -------------------------------------------------------------
# SAVE 3: Future 2026 Feature Table (Unlabeled prospective set)
# -------------------------------------------------------------
future_df = merged[merged["year"] == 2026].copy()
# Strictly exclude target yield, cv_percent, and post-harvest leakage variables
future_feature_cols = [c for c in future_df.columns if c not in leakage_cols + ["corn_yield_bu_acre", "cv_percent"]]
future_df = future_df[future_feature_cols].copy()

future_path = os.path.join(OUT_DIR, "Future_2026_Feature_Table.csv")
future_df.round(4).to_csv(future_path, index=False)
print(f"[3] Saved Future_2026_Feature_Table.csv: {future_df.shape[0]} rows, {future_df.shape[1]} columns.")
print(f"    Coverage: {future_df['county_fips'].nunique()} counties for 2026.")

print("\n--- Phase 7 Dataset Integration Complete ---")

