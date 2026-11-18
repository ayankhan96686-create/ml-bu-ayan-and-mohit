"""
Phase 8: Advanced Modeling, Feature Ablation & Ensembling
Script: run_advanced_experiments.py
Description: Executes Task 5 (Feature Sets A-J Ablation), Task 6 (6 Productivity Trend Models),
             Task 7 (GAM, Mixed-Effects, Tree Benchmarks, Stacking Meta-Models),
             and generates the 5 certified experiment CSVs in 03_Advanced_Experiments/.
"""

import os
import time
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge, Lasso, LinearRegression, TheilSenRegressor
from sklearn.preprocessing import StandardScaler, SplineTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

PROJECT_ROOT = r"E:\ml do not open\Maize_Yield_Research_2002_2026"
DATA_PATH = os.path.join(PROJECT_ROOT, "13_Publication_Enhancement", "02_Feature_Research", "Enhanced_Corn_Belt_Historical_Features.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "13_Publication_Enhancement", "03_Advanced_Experiments")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading enhanced dataset...")
df = pd.read_csv(DATA_PATH)
print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns.")

# Define train and test masks
train_mask = df['year'] <= 2022
test_mask = df['year'] >= 2023

y_train = df.loc[train_mask, 'corn_yield_bu_acre'].values
y_test = df.loc[test_mask, 'corn_yield_bu_acre'].values

def evaluate_preds(y_true, y_pred, name="Model"):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    bias = np.mean(y_pred - y_true)
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 10.0))) * 100.0
    return {
        "Model": name,
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "R2": round(r2, 4),
        "Bias": round(bias, 4),
        "MAPE_pct": round(mape, 2)
    }

# ==============================================================================
# TASK 5: FEATURE FAMILY ABLATION (Sets A through J)
# ==============================================================================
print("\n" + "="*80)
print("EXECUTING TASK 5: FEATURE FAMILY ABLATION (Sets A through J)")
print("="*80)

lag_cols = ['yield_lag_1yr', 'yield_lag_2yr', 'yield_roll_mean_5yr']

soil_cols = [c for c in df.columns if c.startswith('soil_') and not c.endswith('_index') and not c.endswith('_contrast')]

raw_climate_cols = [
    'noaa_heat_stress_max_f', 'noaa_prcp_annual_in', 'noaa_prcp_aug_in', 'noaa_prcp_jul_in',
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

# Build Feature Sets A through J
feature_sets = {
    "Set A: Historical Yield Memory Only": lag_cols,
    "Set B: Soil Only": soil_cols,
    "Set C: Climate Only": raw_climate_cols,
    "Set D: Soil + Climate": soil_cols + raw_climate_cols,
    "Set E: Baseline (Memory + Soil + Climate)": lag_cols + soil_cols + raw_climate_cols,
    "Set F: Memory + Advanced Climate": lag_cols + raw_climate_cols + adv_climate_cols,
    "Set G: Memory + Irrigation + Climate": lag_cols + raw_climate_cols + irrigation_cols,
    "Set H: Memory + Soil + Climate + Irrigation": lag_cols + soil_cols + raw_climate_cols + irrigation_cols,
    "Set I: Memory + Remote Sensing Proxy": lag_cols + raw_climate_cols + adv_climate_cols + ['silking_thermal_interaction'],
    "Set J: Full Integrated Architecture": lag_cols + soil_cols + raw_climate_cols + adv_climate_cols + irrigation_cols + phenology_cols + adv_soil_cols
}

ablation_results = []

for set_name, cols in feature_sets.items():
    print(f"Training XGBoost on {set_name} ({len(cols)} features)...")
    X_tr = df.loc[train_mask, cols].copy()
    X_te = df.loc[test_mask, cols].copy()
    
    # Impute missing with median from training fold
    meds = X_tr.median()
    X_tr = X_tr.fillna(meds).fillna(0.0)
    X_te = X_te.fillna(meds).fillna(0.0)
    
    # Standard XGBoost spec matching Phase 5/6 certified configuration
    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_tr, y_train)
    preds = model.predict(X_te)
    
    res = evaluate_preds(y_test, preds, set_name)
    res["Num_Features"] = len(cols)
    ablation_results.append(res)
    print(f"  -> RMSE: {res['RMSE']} | MAE: {res['MAE']} | R2: {res['R2']}")

df_ablation = pd.DataFrame(ablation_results)
ablation_path = os.path.join(OUTPUT_DIR, "ADVANCED_FEATURE_ABLATION.csv")
df_ablation.to_csv(ablation_path, index=False)
print(f"Saved ADVANCED_FEATURE_ABLATION.csv.")

# ==============================================================================
# TASK 6: PRODUCTIVITY TREND MODELING
# ==============================================================================
print("\n" + "="*80)
print("EXECUTING TASK 6: 6 PRODUCTIVITY TREND ARCHITECTURES")
print("="*80)

# Extract county historical time series
counties = df['county_fips'].unique()
df_tr = df[train_mask]
df_te = df[test_mask]

# Pre-allocate predictions
preds_ols = []
preds_theilsen = []
preds_rollmean = df_te['yield_roll_mean_5yr'].fillna(df_tr['corn_yield_bu_acre'].mean()).values
preds_exp_smooth = []
preds_hierarchical = []

# Fit trends per county strictly on training data
state_slopes = {}
# State-level OLS slope
for st in df_tr['state'].unique():
    st_df = df_tr[df_tr['state'] == st]
    if len(st_df) > 5:
        p = np.polyfit(st_df['year'] - 2002, st_df['corn_yield_bu_acre'], 1)
        state_slopes[st] = p[0]
    else:
        state_slopes[st] = 1.9

county_models = {}
for c in counties:
    c_tr = df_tr[df_tr['county_fips'] == c].sort_values('year')
    if len(c_tr) >= 3:
        x_yrs = (c_tr['year'] - 2002).values
        y_c = c_tr['corn_yield_bu_acre'].values
        # 1. OLS
        p_ols = np.polyfit(x_yrs, y_c, 1)
        # 2. Theil-Sen
        ts = TheilSenRegressor(random_state=42)
        ts.fit(x_yrs.reshape(-1, 1), y_c)
        # 4. Holt exponential smoothing proxy (last level + slope)
        alpha = 0.3
        s = y_c[0]
        b = p_ols[0]
        for val in y_c[1:]:
            s_prev = s
            s = alpha * val + (1 - alpha) * (s + b)
            b = 0.1 * (s - s_prev) + 0.9 * b
        # 5. Hierarchical slope (shrink county slope towards state slope)
        st = c_tr['state'].iloc[0]
        st_slope = state_slopes.get(st, 1.9)
        w = len(c_tr) / (len(c_tr) + 10.0) # empirical Bayes shrinkage
        hier_slope = w * p_ols[0] + (1 - w) * st_slope
        hier_intercept = np.mean(y_c) - hier_slope * np.mean(x_yrs)
        
        county_models[c] = {
            'ols': p_ols,
            'theilsen': (ts.coef_[0], ts.intercept_),
            'exp_smooth': (s, b, x_yrs[-1]),
            'hierarchical': (hier_slope, hier_intercept)
        }
    else:
        # Fallback to state average
        st = c_tr['state'].iloc[0] if len(c_tr) > 0 else 'IA'
        county_models[c] = {
            'ols': (state_slopes.get(st, 1.9), 150.0),
            'theilsen': (state_slopes.get(st, 1.9), 150.0),
            'exp_smooth': (150.0, state_slopes.get(st, 1.9), 20),
            'hierarchical': (state_slopes.get(st, 1.9), 150.0)
        }

for idx, row in df_te.iterrows():
    c = row['county_fips']
    yr = row['year'] - 2002
    m = county_models[c]
    # OLS
    preds_ols.append(m['ols'][0] * yr + m['ols'][1])
    # Theil-Sen
    preds_theilsen.append(m['theilsen'][0] * yr + m['theilsen'][1])
    # Exponential Smoothing
    s_last, b_last, yr_last = m['exp_smooth']
    preds_exp_smooth.append(s_last + b_last * (yr - yr_last))
    # Hierarchical
    preds_hierarchical.append(m['hierarchical'][0] * yr + m['hierarchical'][1])

preds_ols = np.array(preds_ols)
preds_theilsen = np.array(preds_theilsen)
preds_exp_smooth = np.array(preds_exp_smooth)
preds_hierarchical = np.array(preds_hierarchical)

# 6. Two-Stage Productivity Trend + Environmental Anomaly Model
# Compute county historical expected trend for all training rows
trend_tr = []
for idx, row in df_tr.iterrows():
    c = row['county_fips']
    yr = row['year'] - 2002
    m = county_models[c]
    trend_tr.append(m['theilsen'][0] * yr + m['theilsen'][1])
trend_tr = np.array(trend_tr)

# Anomaly target
y_anomaly_tr = y_train - trend_tr

# Train environmental anomaly model (using full environmental features)
env_cols = soil_cols + raw_climate_cols + adv_climate_cols + irrigation_cols
X_env_tr = df_tr[env_cols].fillna(df_tr[env_cols].median()).fillna(0.0)
X_env_te = df_te[env_cols].fillna(df_tr[env_cols].median()).fillna(0.0)

anomaly_model = xgb.XGBRegressor(
    n_estimators=250,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
anomaly_model.fit(X_env_tr, y_anomaly_tr)
pred_anomalies_te = anomaly_model.predict(X_env_te)

preds_twostage = preds_theilsen + pred_anomalies_te

trend_results = [
    evaluate_preds(y_test, preds_ols, "1. County-Specific Linear Trend (OLS)"),
    evaluate_preds(y_test, preds_theilsen, "2. County Robust Theil-Sen Trend"),
    evaluate_preds(y_test, preds_rollmean, "3. Historical Rolling Mean (5-Year)"),
    evaluate_preds(y_test, preds_exp_smooth, "4. Historical Exponential Smoothing (Holt)"),
    evaluate_preds(y_test, preds_hierarchical, "5. Hierarchical County/State Trend"),
    evaluate_preds(y_test, preds_twostage, "6. Two-Stage Trend + Environmental Anomaly (Proposed)")
]

df_trend = pd.DataFrame(trend_results)
trend_path = os.path.join(OUTPUT_DIR, "TREND_MODEL_COMPARISON.csv")
df_trend.to_csv(trend_path, index=False)
print(f"Saved TREND_MODEL_COMPARISON.csv.")
for r in trend_results:
    print(f"  {r['Model']}: RMSE={r['RMSE']}, R2={r['R2']}")

# ==============================================================================
# TASK 7: GAM, MIXED-EFFECTS & ADVANCED ENSEMBLE BENCHMARKING
# ==============================================================================
print("\n" + "="*80)
print("EXECUTING TASK 7: ADVANCED MODELING & ENSEMBLES")
print("="*80)

# Prepare common feature set for benchmarking (Set J: Full Integrated)
all_cols = feature_sets["Set J: Full Integrated Architecture"]
X_all_tr = df_tr[all_cols].copy()
X_all_te = df_te[all_cols].copy()
meds_all = X_all_tr.median()
X_all_tr = X_all_tr.fillna(meds_all).fillna(0.0)
X_all_te = X_all_te.fillna(meds_all).fillna(0.0)

# Scaled features for linear/spline/MLP models
scaler = StandardScaler()
X_all_tr_scaled = scaler.fit_transform(X_all_tr)
X_all_te_scaled = scaler.transform(X_all_te)

# 1. Generalized Additive Model (Spline Non-Linear Regression)
print("Fitting GAM (B-Splines on Climate + Regularized Ridge)...")
# Select 6 key climate variables for natural cubic splines
spline_cols = ['noaa_prcp_jul_in', 'noaa_tmax_jul_f', 'vpd_july_kpa', 'edd_july_f', 'gdd_may_aug_c', 'prism_ppt_mm']
spline_idx = [all_cols.index(c) for c in spline_cols]
non_spline_idx = [i for i in range(len(all_cols)) if i not in spline_idx]

spline_trans = SplineTransformer(n_knots=5, degree=3)
X_spline_tr = spline_trans.fit_transform(X_all_tr[spline_cols])
X_spline_te = spline_trans.transform(X_all_te[spline_cols])

X_gam_tr = np.hstack([X_all_tr_scaled[:, non_spline_idx], X_spline_tr])
X_gam_te = np.hstack([X_all_te_scaled[:, non_spline_idx], X_spline_te])

gam_model = Ridge(alpha=10.0, random_state=42)
gam_model.fit(X_gam_tr, y_train)
preds_gam = gam_model.predict(X_gam_te)

# 2. Hierarchical Mixed-Effects Regression Proxy
print("Fitting Hierarchical Mixed-Effects Model (County Random Intercepts + Ridge)...")
# Standard Ridge with County and State fixed/random effect proxies
df_tr_me = df_tr.copy()
df_te_me = df_te.copy()
# Encode county yield historical deviation
c_mean_yield = df_tr.groupby('county_fips')['corn_yield_bu_acre'].mean()
df_tr_me['county_effect'] = df_tr['county_fips'].map(c_mean_yield)
df_te_me['county_effect'] = df_te['county_fips'].map(c_mean_yield).fillna(df_tr['corn_yield_bu_acre'].mean())

X_me_tr = np.column_stack([X_all_tr_scaled, df_tr_me['county_effect']])
X_me_te = np.column_stack([X_all_te_scaled, df_te_me['county_effect']])

me_model = Ridge(alpha=20.0, random_state=42)
me_model.fit(X_me_tr, y_train)
preds_mixed = me_model.predict(X_me_te)

df_gam_mixed = pd.DataFrame([
    evaluate_preds(y_test, preds_gam, "Generalized Additive Model (GAM / B-Splines)"),
    evaluate_preds(y_test, preds_mixed, "Hierarchical Mixed-Effects Regression (Ridge)")
])
gam_mixed_path = os.path.join(OUTPUT_DIR, "GAM_MIXED_EFFECTS_RESULTS.csv")
df_gam_mixed.to_csv(gam_mixed_path, index=False)
print(f"Saved GAM_MIXED_EFFECTS_RESULTS.csv.")

# 3. Benchmark Base Tree Models & MLP
print("Training Candidate Base Learners for Stacking...")

# A. HistGradientBoosting
hgb = HistGradientBoostingRegressor(max_iter=300, max_depth=7, learning_rate=0.05, random_state=42)
hgb.fit(X_all_tr, y_train)
preds_hgb = hgb.predict(X_all_te)

# B. LightGBM
lgb_model = lgb.LGBMRegressor(n_estimators=300, max_depth=7, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)
lgb_model.fit(X_all_tr, y_train)
preds_lgb = lgb_model.predict(X_all_te)

# C. XGBoost
xgb_model = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
xgb_model.fit(X_all_tr, y_train)
preds_xgb = xgb_model.predict(X_all_te)

# D. CatBoost
cat_model = CatBoostRegressor(iterations=350, depth=6, learning_rate=0.05, random_seed=42, verbose=0)
cat_model.fit(X_all_tr, y_train)
preds_cat = cat_model.predict(X_all_te)

# E. Multi-Layer Perceptron (MLP)
mlp_model = MLPRegressor(hidden_layer_sizes=(128, 64), activation='relu', max_iter=200, random_state=42, early_stopping=True)
mlp_model.fit(X_all_tr_scaled, y_train)
preds_mlp = mlp_model.predict(X_all_te_scaled)

# 4. Out-of-Fold (OOF) Expanding-Window Predictions for Stacking
print("Computing Leakage-Free Expanding-Window OOF Predictions for Stacking...")
# Use 4 expanding folds on training data (2018, 2019, 2020, 2021-2022)
oof_preds = {
    'hgb': np.zeros(len(df_tr)),
    'lgb': np.zeros(len(df_tr)),
    'xgb': np.zeros(len(df_tr)),
    'cat': np.zeros(len(df_tr)),
    'mlp': np.zeros(len(df_tr))
}

# Evaluate OOF on validation years (2019-2022)
val_years = [2019, 2020, 2021, 2022]
val_indices = []

for val_yr in val_years:
    f_tr_mask = (df_tr['year'] < val_yr).values
    f_val_mask = (df_tr['year'] == val_yr).values
    val_idx_loc = np.where(f_val_mask)[0]
    val_indices.extend(val_idx_loc)
    
    f_X_tr = X_all_tr[f_tr_mask]
    f_y_tr = y_train[f_tr_mask]
    f_X_val = X_all_tr[f_val_mask]
    
    f_X_tr_sc = X_all_tr_scaled[f_tr_mask]
    f_X_val_sc = X_all_tr_scaled[f_val_mask]
    
    # Train fold models
    m_hgb = HistGradientBoostingRegressor(max_iter=150, max_depth=6, learning_rate=0.05, random_state=42).fit(f_X_tr, f_y_tr)
    m_lgb = lgb.LGBMRegressor(n_estimators=150, max_depth=6, learning_rate=0.05, random_state=42, verbose=-1).fit(f_X_tr, f_y_tr)
    m_xgb = xgb.XGBRegressor(n_estimators=150, max_depth=5, learning_rate=0.05, random_state=42, n_jobs=-1).fit(f_X_tr, f_y_tr)
    m_cat = CatBoostRegressor(iterations=150, depth=5, learning_rate=0.05, random_seed=42, verbose=0).fit(f_X_tr, f_y_tr)
    m_mlp = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=150, random_state=42).fit(f_X_tr_sc, f_y_tr)
    
    oof_preds['hgb'][val_idx_loc] = m_hgb.predict(f_X_val)
    oof_preds['lgb'][val_idx_loc] = m_lgb.predict(f_X_val)
    oof_preds['xgb'][val_idx_loc] = m_xgb.predict(f_X_val)
    oof_preds['cat'][val_idx_loc] = m_cat.predict(f_X_val)
    oof_preds['mlp'][val_idx_loc] = m_mlp.predict(f_X_val_sc)

val_indices = np.array(val_indices)
OOF_df = pd.DataFrame({
    'hgb': oof_preds['hgb'][val_indices],
    'lgb': oof_preds['lgb'][val_indices],
    'xgb': oof_preds['xgb'][val_indices],
    'cat': oof_preds['cat'][val_indices],
    'mlp': oof_preds['mlp'][val_indices]
})
y_val_oof = y_train[val_indices]

# Residual correlation matrix
corr_matrix = OOF_df.corr()
print("OOF Prediction Correlation Matrix:")
print(corr_matrix.round(3))

# Train Stacking Meta-Learner (Non-Negative Least Squares & Ridge)
test_base_preds = pd.DataFrame({
    'hgb': preds_hgb,
    'lgb': preds_lgb,
    'xgb': preds_xgb,
    'cat': preds_cat,
    'mlp': preds_mlp
})

# A. Equal Weighted Average
preds_equal_blend = test_base_preds.mean(axis=1).values

# B. Non-Negative Least Squares / Constrained Stacking
meta_ridge = Ridge(alpha=5.0, positive=True)
meta_ridge.fit(OOF_df, y_val_oof)
preds_stacking_ridge = meta_ridge.predict(test_base_preds)
stacking_weights = dict(zip(OOF_df.columns, meta_ridge.coef_.round(3)))
print(f"Stacking Meta-Learner Weights: {stacking_weights}")

# C. Top-3 Tree Ensemble (HistGB + LightGBM + XGBoost)
preds_top3_blend = (preds_hgb * 0.35 + preds_lgb * 0.35 + preds_xgb * 0.30)

ensemble_results = [
    evaluate_preds(y_test, preds_equal_blend, "Equal-Weighted 5-Model Ensemble (HistGB+LGB+XGB+Cat+MLP)"),
    evaluate_preds(y_test, preds_top3_blend, "Optimized Tree Triad Ensemble (HistGB 35% + LightGBM 35% + XGBoost 30%)"),
    evaluate_preds(y_test, preds_stacking_ridge, "Stacking Meta-Learner (Ridge on Expanding OOF Predictions)")
]

df_ensemble = pd.DataFrame(ensemble_results)
ensemble_path = os.path.join(OUTPUT_DIR, "ENSEMBLE_MODEL_RESULTS.csv")
df_ensemble.to_csv(ensemble_path, index=False)
print(f"Saved ENSEMBLE_MODEL_RESULTS.csv.")

# 5. Build FINAL_MODEL_COMPARISON.csv
all_benchmark_models = [
    evaluate_preds(y_test, preds_ols, "County Linear Trend (OLS)"),
    evaluate_preds(y_test, preds_theilsen, "County Theil-Sen Robust Trend"),
    evaluate_preds(y_test, preds_rollmean, "Historical 5-Year Rolling Mean"),
    evaluate_preds(y_test, preds_twostage, "Two-Stage Trend + Environmental Anomaly"),
    evaluate_preds(y_test, preds_gam, "Generalized Additive Model (GAM / B-Splines)"),
    evaluate_preds(y_test, preds_mixed, "Hierarchical Mixed-Effects Regression"),
    evaluate_preds(y_test, preds_mlp, "Multi-Layer Perceptron (MLP)"),
    evaluate_preds(y_test, preds_cat, "CatBoost Regressor"),
    evaluate_preds(y_test, preds_xgb, "XGBoost Regressor (Set J Integrated)"),
    evaluate_preds(y_test, preds_hgb, "HistGradientBoosting Regressor"),
    evaluate_preds(y_test, preds_lgb, "LightGBM Regressor (Set J Integrated)"),
    evaluate_preds(y_test, preds_top3_blend, "Optimized Tree Triad Ensemble (HistGB+LGB+XGB)"),
    evaluate_preds(y_test, preds_stacking_ridge, "Stacking Meta-Learner (Ridge OOF Blender)")
]

df_final_comparison = pd.DataFrame(all_benchmark_models).sort_values("RMSE")
final_path = os.path.join(OUTPUT_DIR, "FINAL_MODEL_COMPARISON.csv")
df_final_comparison.to_csv(final_path, index=False)
print(f"Saved FINAL_MODEL_COMPARISON.csv.")
print("\nFinal Model Comparison Leaderboard:")
print(df_final_comparison[['Model', 'RMSE', 'MAE', 'R2', 'Bias']].to_string(index=False))

