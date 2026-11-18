"""
Year 2012 Removal Impact Analysis & Ablation Audit
Script: audit_2012_removal_impact.py
Description: Evaluates the scientific, biophysical, and statistical consequences of removing
             the historic 2012 drought year from the dataset.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

PROJECT_ROOT = r"E:\ml do not open\Maize_Yield_Research_2002_2026"
DATA_PATH = os.path.join(PROJECT_ROOT, "14_Final_Data_Model_Optimization", "08_Final_Model", "Historical_Features_Corrected_v2.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "14_Final_Data_Model_Optimization", "04_Model_Optimization", "2012_Ablation_Study")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading dataset for 2012 ablation analysis...")
df = pd.read_csv(DATA_PATH)

# Feature definitions (69 strict in-season predictors)
lag_cols = ['yield_lag_1yr', 'yield_lag_2yr', 'yield_roll_mean_5yr']
soil_cols = [c for c in df.columns if c.startswith('soil_') and not c.endswith('_index') and not c.endswith('_contrast') and c != 'county_fips_5digit']
raw_climate_cols = [
    'noaa_heat_stress_max_f', 'noaa_prcp_aug_in', 'noaa_prcp_jul_in',
    'noaa_prcp_jun_in', 'noaa_prcp_may_aug_in', 'noaa_prcp_may_in', 'noaa_prcp_may_sep_in',
    'noaa_prcp_sep_in', 'noaa_tmax_aug_f', 'noaa_tmax_jul_f', 'noaa_tmax_may_aug_f',
    'noaa_tmean_may_aug_f', 'noaa_tmin_may_aug_f', 'prism_ppt_mm', 'prism_tmax_c',
    'prism_tmean_c', 'prism_tmin_c'
]
adv_climate_cols = [
    'gdd_may_aug_c', 'gdd_july_c', 'edd_july_f', 'edd_aug_f', 'vpd_july_kpa', 'vpd_aug_kpa',
    'compound_heat_drought_july', 'noaa_prcp_jul_in_zscore', 'noaa_tmax_jul_f_zscore'
]
irrigation_cols = ['irrigation_regime_proxy', 'irrigation_drought_buffer', 'rma_liability_per_planted_acre']
phenology_cols = [
    'silking_thermal_interaction', 'delayed_planting_penalty',
    'state_progress_planted_w20_pct', 'state_progress_emerged_w22_pct',
    'state_progress_silking_w30_pct', 'state_progress_dough_w34_pct',
    'state_cond_excellent_mean', 'state_cond_good_mean', 'state_cond_poor_mean'
]
adv_soil_cols = ['soil_awc_index', 'subsoil_clay_contrast']
predictors = lag_cols + soil_cols + raw_climate_cols + adv_climate_cols + irrigation_cols + phenology_cols + adv_soil_cols

def calc_metrics(y_true, y_pred):
    sse = np.sum((y_true - y_pred)**2)
    sst = np.sum((y_true - np.mean(y_true))**2)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = 1.0 - (sse / sst) if sst > 0 else 0.0
    bias = np.mean(y_pred - y_true)
    return round(rmse, 4), round(mae, 4), round(r2, 4), round(bias, 4)

# -------------------------------------------------------------------------
# Cohorts
# -------------------------------------------------------------------------
# Test cohort (2023-2025)
test_mask = df['year'] >= 2023
X_te = df.loc[test_mask, predictors].copy()
y_te = df.loc[test_mask, 'corn_yield_bu_acre'].values

# Baseline Training Cohort WITH 2012 (2002-2022)
tr_with_mask = df['year'] <= 2022
X_tr_with = df.loc[tr_with_mask, predictors].copy()
y_tr_with = df.loc[tr_with_mask, 'corn_yield_bu_acre'].values

# Ablated Training Cohort WITHOUT 2012 (2002-2022 excluding 2012)
tr_no_mask = (df['year'] <= 2022) & (df['year'] != 2012)
X_tr_no = df.loc[tr_no_mask, predictors].copy()
y_tr_no = df.loc[tr_no_mask, 'corn_yield_bu_acre'].values

print(f"Test cohort (2023-2025): {len(y_te)} obs")
print(f"Train cohort WITH 2012: {len(y_tr_with)} obs")
print(f"Train cohort WITHOUT 2012: {len(y_tr_no)} obs (removed {len(y_tr_with) - len(y_tr_no)} rows)")

meds_with = X_tr_with.median()
X_tr_with = X_tr_with.fillna(meds_with).fillna(0.0)
X_te_with = X_te.fillna(meds_with).fillna(0.0)

meds_no = X_tr_no.median()
X_tr_no = X_tr_no.fillna(meds_no).fillna(0.0)
X_te_no = X_te.fillna(meds_no).fillna(0.0)

# -------------------------------------------------------------------------
# 1. Train Models Under Both Conditions
# -------------------------------------------------------------------------
models = {
    "LightGBM": (
        lgb.LGBMRegressor(n_estimators=350, max_depth=7, num_leaves=45, learning_rate=0.04, subsample=0.85, colsample_bytree=0.75, random_state=42, verbose=-1),
        lgb.LGBMRegressor(n_estimators=350, max_depth=7, num_leaves=45, learning_rate=0.04, subsample=0.85, colsample_bytree=0.75, random_state=42, verbose=-1)
    ),
    "XGBoost": (
        xgb.XGBRegressor(n_estimators=350, max_depth=6, learning_rate=0.04, subsample=0.80, colsample_bytree=0.75, random_state=42, n_jobs=-1),
        xgb.XGBRegressor(n_estimators=350, max_depth=6, learning_rate=0.04, subsample=0.80, colsample_bytree=0.75, random_state=42, n_jobs=-1)
    ),
    "HistGB": (
        HistGradientBoostingRegressor(max_iter=350, max_depth=7, learning_rate=0.04, random_state=42),
        HistGradientBoostingRegressor(max_iter=350, max_depth=7, learning_rate=0.04, random_state=42)
    ),
    "CatBoost": (
        CatBoostRegressor(iterations=400, depth=6, learning_rate=0.04, random_seed=42, verbose=0),
        CatBoostRegressor(iterations=400, depth=6, learning_rate=0.04, random_seed=42, verbose=0)
    )
}

comparison_records = []
preds_dict = {"WITH_2012": {}, "WITHOUT_2012": {}}

for name, (m_with, m_no) in models.items():
    print(f"Training {name} with and without 2012...")
    m_with.fit(X_tr_with, y_tr_with)
    p_with = m_with.predict(X_te_with)
    preds_dict["WITH_2012"][name] = p_with
    r_with = calc_metrics(y_te, p_with)
    
    m_no.fit(X_tr_no, y_tr_no)
    p_no = m_no.predict(X_te_no)
    preds_dict["WITHOUT_2012"][name] = p_no
    r_no = calc_metrics(y_te, p_no)
    
    comparison_records.append({
        "Model": name,
        "With_2012_RMSE": r_with[0],
        "Without_2012_RMSE": r_no[0],
        "Delta_RMSE": round(r_no[0] - r_with[0], 4),
        "With_2012_MAE": r_with[1],
        "Without_2012_MAE": r_no[1],
        "Delta_MAE": round(r_no[1] - r_with[1], 4),
        "With_2012_R2": r_with[2],
        "Without_2012_R2": r_no[2],
        "Delta_R2": round(r_no[2] - r_with[2], 4),
        "With_2012_Bias": r_with[3],
        "Without_2012_Bias": r_no[3],
        "Delta_Bias": round(r_no[3] - r_with[3], 4)
    })

# Calibrated Quad-Tree Ensemble
p_ens_with = (preds_dict["WITH_2012"]["LightGBM"] * 0.35 +
              preds_dict["WITH_2012"]["CatBoost"] * 0.25 +
              preds_dict["WITH_2012"]["HistGB"] * 0.20 +
              preds_dict["WITH_2012"]["XGBoost"] * 0.20)

p_ens_no = (preds_dict["WITHOUT_2012"]["LightGBM"] * 0.35 +
            preds_dict["WITHOUT_2012"]["CatBoost"] * 0.25 +
            preds_dict["WITHOUT_2012"]["HistGB"] * 0.20 +
            preds_dict["WITHOUT_2012"]["XGBoost"] * 0.20)

r_ens_with = calc_metrics(y_te, p_ens_with)
r_ens_no = calc_metrics(y_te, p_ens_no)

comparison_records.append({
    "Model": "Calibrated Quad-Tree Ensemble",
    "With_2012_RMSE": r_ens_with[0],
    "Without_2012_RMSE": r_ens_no[0],
    "Delta_RMSE": round(r_ens_no[0] - r_ens_with[0], 4),
    "With_2012_MAE": r_ens_with[1],
    "Without_2012_MAE": r_ens_no[1],
    "Delta_MAE": round(r_ens_no[1] - r_ens_with[1], 4),
    "With_2012_R2": r_ens_with[2],
    "Without_2012_R2": r_ens_no[2],
    "Delta_R2": round(r_ens_no[2] - r_ens_with[2], 4),
    "With_2012_Bias": r_ens_with[3],
    "Without_2012_Bias": r_ens_no[3],
    "Delta_Bias": round(r_ens_no[3] - r_ens_with[3], 4)
})

df_comp = pd.DataFrame(comparison_records)
comp_path = os.path.join(OUTPUT_DIR, "MODEL_COMPARISON_WITH_VS_WITHOUT_2012.csv")
df_comp.to_csv(comp_path, index=False)
print(f"\nSaved {comp_path}:")
print(df_comp.to_string())

# -------------------------------------------------------------------------
# 2. Performance Breakdown by Year in Prospective Test (2023, 2024, 2025)
# -------------------------------------------------------------------------
df_eval = df.loc[test_mask, ['year', 'state', 'county_name', 'corn_yield_bu_acre']].copy()
df_eval['pred_with'] = p_ens_with
df_eval['pred_no'] = p_ens_no

annual_comp = []
for yr in [2023, 2024, 2025]:
    sub = df_eval[df_eval['year'] == yr]
    y_yr = sub['corn_yield_bu_acre'].values
    m_w = calc_metrics(y_yr, sub['pred_with'].values)
    m_n = calc_metrics(y_yr, sub['pred_no'].values)
    annual_comp.append({
        "Year": yr,
        "Agroclimatic_Condition": "Early season drought" if yr==2023 else ("Ideal bumper crop" if yr==2024 else "Near-trend normal"),
        "Counties": len(sub),
        "With_2012_RMSE": m_w[0],
        "Without_2012_RMSE": m_n[0],
        "Delta_RMSE": round(m_n[0] - m_w[0], 4),
        "With_2012_R2": m_w[2],
        "Without_2012_R2": m_n[2],
        "Delta_R2": round(m_n[2] - m_w[2], 4),
        "With_2012_Bias": m_w[3],
        "Without_2012_Bias": m_n[3]
    })

df_annual = pd.DataFrame(annual_comp)
annual_path = os.path.join(OUTPUT_DIR, "ANNUAL_TEST_PERFORMANCE_WITH_VS_WITHOUT_2012.csv")
df_annual.to_csv(annual_path, index=False)
print(f"\nSaved {annual_path}:")
print(df_annual.to_string())

# -------------------------------------------------------------------------
# 3. Performance on Extreme Drought / Low-Yield Strata
# -------------------------------------------------------------------------
# Stratify test set by actual yield to examine negative tail prediction
strata_records = []

# Subsets in 2023-2025
stress_filters = {
    "Low Yield Tail (< 140 bu/ac)": df_eval['corn_yield_bu_acre'] < 140,
    "Severe Stress Tail (< 120 bu/ac)": df_eval['corn_yield_bu_acre'] < 120,
    "Normal Yield (140 - 190 bu/ac)": (df_eval['corn_yield_bu_acre'] >= 140) & (df_eval['corn_yield_bu_acre'] < 190),
    "Bumper Yield (>= 190 bu/ac)": df_eval['corn_yield_bu_acre'] >= 190
}

for label, mask_s in stress_filters.items():
    sub = df_eval[mask_s]
    if len(sub) > 0:
        y_s = sub['corn_yield_bu_acre'].values
        m_w = calc_metrics(y_s, sub['pred_with'].values)
        m_n = calc_metrics(y_s, sub['pred_no'].values)
        strata_records.append({
            "Stratum": label,
            "Counties": len(sub),
            "Actual_Mean_Yield": round(float(np.mean(y_s)), 2),
            "With_2012_RMSE": m_w[0],
            "Without_2012_RMSE": m_n[0],
            "Delta_RMSE": round(m_n[0] - m_w[0], 4),
            "With_2012_Bias": m_w[3],
            "Without_2012_Bias": m_n[3],
            "Delta_Bias": round(m_n[3] - m_w[3], 4)
        })

df_strata = pd.DataFrame(strata_records)
strata_path = os.path.join(OUTPUT_DIR, "DROUGHT_STRATA_PERFORMANCE_WITH_VS_WITHOUT_2012.csv")
df_strata.to_csv(strata_path, index=False)
print(f"\nSaved {strata_path}:")
print(df_strata.to_string())

# -------------------------------------------------------------------------
# 4. Feature Lag Impact Analysis
# -------------------------------------------------------------------------
# Check how 2012 impacts 2013-2017 feature engineering
lag_audit = []
for yr in range(2013, 2018):
    sub = df[df['year'] == yr]
    lag_audit.append({
        "Year": yr,
        "Observations": len(sub),
        "Mean_Yield_Lag_1yr": round(float(sub['yield_lag_1yr'].mean()), 2) if 'yield_lag_1yr' in sub.columns else np.nan,
        "Mean_Yield_Lag_2yr": round(float(sub['yield_lag_2yr'].mean()), 2) if 'yield_lag_2yr' in sub.columns else np.nan,
        "Mean_Yield_Roll_5yr": round(float(sub['yield_roll_mean_5yr'].mean()), 2) if 'yield_roll_mean_5yr' in sub.columns else np.nan,
        "Includes_2012_In_Lag1": (yr == 2013),
        "Includes_2012_In_Lag2": (yr == 2014),
        "Includes_2012_In_Roll5": (2013 <= yr <= 2017)
    })

df_lag_audit = pd.DataFrame(lag_audit)
lag_path = os.path.join(OUTPUT_DIR, "LAG_IMPACT_OF_2012_ON_SUBSEQUENT_YEARS.csv")
df_lag_audit.to_csv(lag_path, index=False)
print(f"\nSaved {lag_path}:")
print(df_lag_audit.to_string())

print("\n--- 2012 Ablation Script Completed Successfully ---")

