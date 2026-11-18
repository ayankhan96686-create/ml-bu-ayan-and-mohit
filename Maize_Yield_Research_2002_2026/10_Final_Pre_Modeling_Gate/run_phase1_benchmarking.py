import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

gate_dir = r"E:\ml do not open\Maize_Yield_Research_2002_2026\10_Final_Pre_Modeling_Gate"
p_train = os.path.join(gate_dir, "Purified_Corn_Belt_Historical_Dataset.csv")

print("Executing Phase 1: Complete Baseline Model Benchmarking...")
df = pd.read_csv(p_train)

# Partitions
train_mask = df['year'] <= 2019
val_mask = (df['year'] >= 2020) & (df['year'] <= 2022)
test_mask = (df['year'] >= 2023) & (df['year'] <= 2025)

target = 'corn_yield_bu_acre'
y_train = df.loc[train_mask, target].values
y_val = df.loc[val_mask, target].values
y_test = df.loc[test_mask, target].values

# Feature Sets definition:
# Late-Season Window: May-Sep NOAA Climate + Soil. Exclude annual precip and coordinates.
soil_cols = [c for c in df.columns if c.startswith('soil_') and c not in ['soil_intptlat', 'soil_intptlon']]
noaa_cols = [c for c in df.columns if c.startswith('noaa_') and c != 'noaa_prcp_annual_in']
# PRISM features: include for Set A comparison
prism_cols = ['prism_ppt_mm', 'prism_tmean_c', 'prism_tmin_c', 'prism_tmax_c']

features_A = sorted(soil_cols + noaa_cols) # Pure standardized growing season environmental
features_A_prism = sorted(soil_cols + noaa_cols + prism_cols)
features_D = sorted(features_A + ['yield_lag_1yr', 'yield_lag_2yr', 'yield_roll_mean_5yr'])

print(f"Features Set A (Soil + NOAA May-Sep): {len(features_A)}")
print(f"Features Set D (Set A + Strict Lags): {len(features_D)}")

# DataFrame to collect all predictions
pred_df_val = df.loc[val_mask, ['county_fips', 'state', 'county_name', 'year', target]].copy()
pred_df_test = df.loc[test_mask, ['county_fips', 'state', 'county_name', 'year', target]].copy()

results = []

def record_metrics(model_name, y_true, y_pred, split_name, base_rmse=None):
    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    imp = ((base_rmse - rmse) / base_rmse * 100.0) if base_rmse is not None else 0.0
    return {
        'model_name': model_name,
        'split': split_name,
        'rmse': round(rmse, 2),
        'mae': round(mae, 2),
        'r2': round(r2, 4),
        'rmse_improvement_over_hist_mean_pct': round(imp, 2)
    }

# -------------------------------------------------------------
# Baseline 1: County Historical Mean
# -------------------------------------------------------------
train_df = df[train_mask]
global_train_mean = train_df[target].mean()
county_hist_means = train_df.groupby('county_fips')[target].mean().to_dict()

pred_val_hist_mean = pred_df_val['county_fips'].map(county_hist_means).fillna(global_train_mean).values
pred_test_hist_mean = pred_df_test['county_fips'].map(county_hist_means).fillna(global_train_mean).values

pred_df_val['pred_hist_mean'] = np.round(pred_val_hist_mean, 2)
pred_df_test['pred_hist_mean'] = np.round(pred_test_hist_mean, 2)

val_base_rmse = root_mean_squared_error(y_val, pred_val_hist_mean)
test_base_rmse = root_mean_squared_error(y_test, pred_test_hist_mean)

results.append(record_metrics('1. County Historical Mean Baseline', y_val, pred_val_hist_mean, 'Validation (2020-2022)', val_base_rmse))
results.append(record_metrics('1. County Historical Mean Baseline', y_test, pred_test_hist_mean, 'Test (2023-2025)', test_base_rmse))

# -------------------------------------------------------------
# Baseline 2: Previous-Year Yield Baseline (yield_lag_1yr)
# -------------------------------------------------------------
pred_val_lag1 = df.loc[val_mask, 'yield_lag_1yr'].fillna(pred_df_val['county_fips'].map(county_hist_means).fillna(global_train_mean)).values
pred_test_lag1 = df.loc[test_mask, 'yield_lag_1yr'].fillna(pred_df_test['county_fips'].map(county_hist_means).fillna(global_train_mean)).values

pred_df_val['pred_prev_year_lag1'] = np.round(pred_val_lag1, 2)
pred_df_test['pred_prev_year_lag1'] = np.round(pred_test_lag1, 2)

results.append(record_metrics('2. Previous-Year Yield Baseline (Lag-1)', y_val, pred_val_lag1, 'Validation (2020-2022)', val_base_rmse))
results.append(record_metrics('2. Previous-Year Yield Baseline (Lag-1)', y_test, pred_test_lag1, 'Test (2023-2025)', test_base_rmse))

# -------------------------------------------------------------
# Baseline 3: 5-Year Historical-Yield Baseline (yield_roll_mean_5yr)
# -------------------------------------------------------------
pred_val_roll5 = df.loc[val_mask, 'yield_roll_mean_5yr'].fillna(pred_df_val['county_fips'].map(county_hist_means).fillna(global_train_mean)).values
pred_test_roll5 = df.loc[test_mask, 'yield_roll_mean_5yr'].fillna(pred_df_test['county_fips'].map(county_hist_means).fillna(global_train_mean)).values

pred_df_val['pred_roll_mean_5yr'] = np.round(pred_val_roll5, 2)
pred_df_test['pred_roll_mean_5yr'] = np.round(pred_test_roll5, 2)

results.append(record_metrics('3. 5-Year Historical Yield Baseline', y_val, pred_val_roll5, 'Validation (2020-2022)', val_base_rmse))
results.append(record_metrics('3. 5-Year Historical Yield Baseline', y_test, pred_test_roll5, 'Test (2023-2025)', test_base_rmse))

# -------------------------------------------------------------
# Model 4: Ridge Regression (Set A - Environmental)
# -------------------------------------------------------------
# Linear models require imputation for missing values in features
X_train_A = df.loc[train_mask, features_A].copy()
X_val_A = df.loc[val_mask, features_A].copy()
X_test_A = df.loc[test_mask, features_A].copy()

# Fill missing with train median
medians_A = X_train_A.median()
X_train_A_imp = X_train_A.fillna(medians_A)
X_val_A_imp = X_val_A.fillna(medians_A)
X_test_A_imp = X_test_A.fillna(medians_A)

scaler_A = StandardScaler()
X_train_A_scaled = scaler_A.fit_transform(X_train_A_imp)
X_val_A_scaled = scaler_A.transform(X_val_A_imp)
X_test_A_scaled = scaler_A.transform(X_test_A_imp)

# Tune alpha on validation set
best_alpha = 1.0
best_val_r2 = -1e9
for alpha_cand in [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:
    r = Ridge(alpha=alpha_cand)
    r.fit(X_train_A_scaled, y_train)
    val_score = r2_score(y_val, r.predict(X_val_A_scaled))
    if val_score > best_val_r2:
        best_val_r2 = val_score
        best_alpha = alpha_cand

ridge_A = Ridge(alpha=best_alpha)
ridge_A.fit(X_train_A_scaled, y_train)

pred_val_ridge_A = ridge_A.predict(X_val_A_scaled)
pred_test_ridge_A = ridge_A.predict(X_test_A_scaled)

pred_df_val['pred_ridge_SetA'] = np.round(pred_val_ridge_A, 2)
pred_df_test['pred_ridge_SetA'] = np.round(pred_test_ridge_A, 2)

results.append(record_metrics(f'4. Ridge Regression (Set A, alpha={best_alpha})', y_val, pred_val_ridge_A, 'Validation (2020-2022)', val_base_rmse))
results.append(record_metrics(f'4. Ridge Regression (Set A, alpha={best_alpha})', y_test, pred_test_ridge_A, 'Test (2023-2025)', test_base_rmse))

# -------------------------------------------------------------
# Model 5: HistGradientBoosting (Set A - Environmental)
# -------------------------------------------------------------
hgb_A = HistGradientBoostingRegressor(
    max_iter=300,
    learning_rate=0.05,
    max_leaf_nodes=31,
    random_state=42,
    early_stopping=True
)
hgb_A.fit(X_train_A, y_train)

pred_val_hgb_A = hgb_A.predict(X_val_A)
pred_test_hgb_A = hgb_A.predict(X_test_A)

pred_df_val['pred_histgb_SetA'] = np.round(pred_val_hgb_A, 2)
pred_df_test['pred_histgb_SetA'] = np.round(pred_test_hgb_A, 2)

results.append(record_metrics('5. HistGradientBoosting (Set A)', y_val, pred_val_hgb_A, 'Validation (2020-2022)', val_base_rmse))
results.append(record_metrics('5. HistGradientBoosting (Set A)', y_test, pred_test_hgb_A, 'Test (2023-2025)', test_base_rmse))

# -------------------------------------------------------------
# Model 6: XGBoost (Set A - Environmental Baseline)
# -------------------------------------------------------------
xgb_A = xgb.XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    early_stopping_rounds=30,
    tree_method='hist'
)
xgb_A.fit(X_train_A, y_train, eval_set=[(X_val_A, y_val)], verbose=False)

pred_val_xgb_A = xgb_A.predict(X_val_A)
pred_test_xgb_A = xgb_A.predict(X_test_A)

pred_df_val['pred_xgboost_SetA'] = np.round(pred_val_xgb_A, 2)
pred_df_test['pred_xgboost_SetA'] = np.round(pred_test_xgb_A, 2)

results.append(record_metrics('6. XGBoost (Set A - Environmental)', y_val, pred_val_xgb_A, 'Validation (2020-2022)', val_base_rmse))
results.append(record_metrics('6. XGBoost (Set A - Environmental)', y_test, pred_test_xgb_A, 'Test (2023-2025)', test_base_rmse))

# -------------------------------------------------------------
# Model 7: XGBoost (Set D - Environmental + Strict Lags)
# -------------------------------------------------------------
X_train_D = df.loc[train_mask, features_D]
X_val_D = df.loc[val_mask, features_D]
X_test_D = df.loc[test_mask, features_D]

xgb_D = xgb.XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    early_stopping_rounds=30,
    tree_method='hist'
)
xgb_D.fit(X_train_D, y_train, eval_set=[(X_val_D, y_val)], verbose=False)

pred_val_xgb_D = xgb_D.predict(X_val_D)
pred_test_xgb_D = xgb_D.predict(X_test_D)

pred_df_val['pred_xgboost_SetD'] = np.round(pred_val_xgb_D, 2)
pred_df_test['pred_xgboost_SetD'] = np.round(pred_test_xgb_D, 2)

results.append(record_metrics('7. XGBoost (Set D - Environmental + Strict Lags)', y_val, pred_val_xgb_D, 'Validation (2020-2022)', val_base_rmse))
results.append(record_metrics('7. XGBoost (Set D - Environmental + Strict Lags)', y_test, pred_test_xgb_D, 'Test (2023-2025)', test_base_rmse))

# Save Results & Predictions
df_res = pd.DataFrame(results)
p_res_csv = os.path.join(gate_dir, "BASELINE_MODEL_BENCHMARK_RESULTS.csv")
df_res.to_csv(p_res_csv, index=False)
print(f"\nSaved BASELINE_MODEL_BENCHMARK_RESULTS.csv -> {p_res_csv}")

pred_combined = pd.concat([pred_df_val, pred_df_test], ignore_index=True)
p_pred_csv = os.path.join(gate_dir, "baseline_predictions_val_test.csv")
pred_combined.to_csv(p_pred_csv, index=False)
print(f"Saved baseline_predictions_val_test.csv ({len(pred_combined)} rows) -> {p_pred_csv}")

print("\n--- Summary of Test Set (2023-2025) Benchmarking Results ---")
test_res = df_res[df_res['split'] == 'Test (2023-2025)'].sort_values('rmse')
for idx, r in test_res.iterrows():
    print(f"{r['model_name']:48} | RMSE: {r['rmse']:5.2f} | MAE: {r['mae']:5.2f} | R2: {r['r2']:6.4f} | Imp: {r['rmse_improvement_over_hist_mean_pct']:+5.1f}%")

print("\nPhase 1 benchmarking executed successfully.")

