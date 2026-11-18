import os
import time
import json
import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error, max_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    GradientBoostingRegressor
)
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

project_root = r"E:\ml do not open\Maize_Yield_Research_2002_2026"
gate_dir = os.path.join(project_root, "10_Final_Pre_Modeling_Gate")
out_dir = os.path.join(project_root, "11_Independent_Model_Validation", "04_Advanced_ML_Benchmarks")
os.makedirs(out_dir, exist_ok=True)

data_path = os.path.join(gate_dir, "Purified_Corn_Belt_Historical_Dataset.csv")
df = pd.read_csv(data_path)

train_mask = df['year'] <= 2019
val_mask = (df['year'] >= 2020) & (df['year'] <= 2022)
test_mask = (df['year'] >= 2023) & (df['year'] <= 2025)

target = 'corn_yield_bu_acre'
y_train = df.loc[train_mask, target].values
y_val = df.loc[val_mask, target].values
y_test = df.loc[test_mask, target].values

# Integrated Feature Set D (40 features)
soil_cols = sorted([c for c in df.columns if c.startswith('soil_') and c not in ['soil_intptlat', 'soil_intptlon']])
noaa_cols = sorted([c for c in df.columns if c.startswith('noaa_') and c != 'noaa_prcp_annual_in'])
features_D = sorted(soil_cols + noaa_cols + ['yield_lag_1yr', 'yield_lag_2yr', 'yield_roll_mean_5yr'])

print(f"Total features in Set D: {len(features_D)}")

X_train_raw = df.loc[train_mask, features_D].copy()
X_val_raw = df.loc[val_mask, features_D].copy()
X_test_raw = df.loc[test_mask, features_D].copy()

# Fit median and scaler strictly on train
train_medians = X_train_raw.median()
X_train_imp = X_train_raw.fillna(train_medians)
X_val_imp = X_val_raw.fillna(train_medians)
X_test_imp = X_test_raw.fillna(train_medians)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imp)
X_val_scaled = scaler.transform(X_val_imp)
X_test_scaled = scaler.transform(X_test_imp)

# Save median & scaler parameters for reproducibility
prep_meta = {
    'features': features_D,
    'medians': train_medians.to_dict(),
    'means': dict(zip(features_D, scaler.mean_)),
    'stds': dict(zip(features_D, scaler.scale_))
}
with open(os.path.join(out_dir, "PREPROCESSING_PARAMS_SET_D.json"), "w") as f:
    json.dump(prep_meta, f, indent=2)

def compute_metrics(y_true, y_pred, split_name, model_name, elapsed_sec=0.0):
    rmse = float(root_mean_squared_error(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    mape = float(mean_absolute_percentage_error(y_true, y_pred) * 100.0)
    mx_err = float(max_error(y_true, y_pred))
    return {
        'model_name': model_name,
        'split': split_name,
        'rmse': round(rmse, 2),
        'mae': round(mae, 2),
        'r2': round(r2, 4),
        'mape_pct': round(mape, 2),
        'max_error': round(mx_err, 2),
        'training_time_sec': round(elapsed_sec, 2)
    }

# Track all models, search results, and predictions
benchmark_results = []
hparam_search_records = []
best_hparams = {}

test_pred_df = df.loc[test_mask, ['county_fips', 'state', 'county_name', 'year', target]].copy().reset_index(drop=True)

# -------------------------------------------------------------
# 1. Ridge Regression (tuning alpha)
# -------------------------------------------------------------
print("\n[1/11] Tuning Ridge Regression...")
best_ridge_alpha = 1.0
best_ridge_val_rmse = 1e9
for a in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:
    r = Ridge(alpha=a, random_state=42)
    r.fit(X_train_scaled, y_train)
    v_rmse = root_mean_squared_error(y_val, r.predict(X_val_scaled))
    hparam_search_records.append({'model': 'Ridge', 'param': f'alpha={a}', 'val_rmse': round(v_rmse, 3)})
    if v_rmse < best_ridge_val_rmse:
        best_ridge_val_rmse = v_rmse
        best_ridge_alpha = a

t0 = time.perf_counter()
model_ridge = Ridge(alpha=best_ridge_alpha, random_state=42)
model_ridge.fit(X_train_scaled, y_train)
t_ridge = time.perf_counter() - t0
best_hparams['Ridge'] = {'alpha': best_ridge_alpha}

p_tr = model_ridge.predict(X_train_scaled)
p_va = model_ridge.predict(X_val_scaled)
p_te = model_ridge.predict(X_test_scaled)
test_pred_df['pred_Ridge'] = np.round(p_te, 2)
benchmark_results.append(compute_metrics(y_train, p_tr, 'Train (<=2019)', 'Ridge', t_ridge))
benchmark_results.append(compute_metrics(y_val, p_va, 'Validation (2020-2022)', 'Ridge', t_ridge))
benchmark_results.append(compute_metrics(y_test, p_te, 'Test (2023-2025)', 'Ridge', t_ridge))

# -------------------------------------------------------------
# 2. ElasticNet (tuning alpha and l1_ratio)
# -------------------------------------------------------------
print("[2/11] Tuning ElasticNet...")
best_en_params = (1.0, 0.5)
best_en_val_rmse = 1e9
for a in [0.01, 0.1, 1.0, 10.0]:
    for l1 in [0.1, 0.5, 0.9]:
        en = ElasticNet(alpha=a, l1_ratio=l1, random_state=42, max_iter=2000)
        en.fit(X_train_scaled, y_train)
        v_rmse = root_mean_squared_error(y_val, en.predict(X_val_scaled))
        hparam_search_records.append({'model': 'ElasticNet', 'param': f'alpha={a}, l1_ratio={l1}', 'val_rmse': round(v_rmse, 3)})
        if v_rmse < best_en_val_rmse:
            best_en_val_rmse = v_rmse
            best_en_params = (a, l1)

t0 = time.perf_counter()
model_en = ElasticNet(alpha=best_en_params[0], l1_ratio=best_en_params[1], random_state=42, max_iter=2000)
model_en.fit(X_train_scaled, y_train)
t_en = time.perf_counter() - t0
best_hparams['ElasticNet'] = {'alpha': best_en_params[0], 'l1_ratio': best_en_params[1]}

p_tr = model_en.predict(X_train_scaled)
p_va = model_en.predict(X_val_scaled)
p_te = model_en.predict(X_test_scaled)
test_pred_df['pred_ElasticNet'] = np.round(p_te, 2)
benchmark_results.append(compute_metrics(y_train, p_tr, 'Train (<=2019)', 'ElasticNet', t_en))
benchmark_results.append(compute_metrics(y_val, p_va, 'Validation (2020-2022)', 'ElasticNet', t_en))
benchmark_results.append(compute_metrics(y_test, p_te, 'Test (2023-2025)', 'ElasticNet', t_en))

# -------------------------------------------------------------
# 3. Support Vector Regression (SVR)
# -------------------------------------------------------------
print("[3/11] Tuning SVR...")
best_svr_C = 50.0
best_svr_val_rmse = 1e9
for c_cand in [10.0, 50.0, 100.0]:
    svr = SVR(C=c_cand, epsilon=0.5, kernel='rbf')
    svr.fit(X_train_scaled, y_train)
    v_rmse = root_mean_squared_error(y_val, svr.predict(X_val_scaled))
    hparam_search_records.append({'model': 'SVR', 'param': f'C={c_cand}, eps=0.5', 'val_rmse': round(v_rmse, 3)})
    if v_rmse < best_svr_val_rmse:
        best_svr_val_rmse = v_rmse
        best_svr_C = c_cand

t0 = time.perf_counter()
model_svr = SVR(C=best_svr_C, epsilon=0.5, kernel='rbf')
model_svr.fit(X_train_scaled, y_train)
t_svr = time.perf_counter() - t0
best_hparams['SVR'] = {'kernel': 'rbf', 'C': best_svr_C, 'epsilon': 0.5}

p_tr = model_svr.predict(X_train_scaled)
p_va = model_svr.predict(X_val_scaled)
p_te = model_svr.predict(X_test_scaled)
test_pred_df['pred_SVR'] = np.round(p_te, 2)
benchmark_results.append(compute_metrics(y_train, p_tr, 'Train (<=2019)', 'SVR', t_svr))
benchmark_results.append(compute_metrics(y_val, p_va, 'Validation (2020-2022)', 'SVR', t_svr))
benchmark_results.append(compute_metrics(y_test, p_te, 'Test (2023-2025)', 'SVR', t_svr))

# -------------------------------------------------------------
# 4. Multi-Layer Perceptron (MLPRegressor)
# -------------------------------------------------------------
print("[4/11] Tuning MLPRegressor...")
t0 = time.perf_counter()
model_mlp = MLPRegressor(
    hidden_layer_sizes=(128, 64),
    activation='relu',
    solver='adam',
    alpha=0.01,
    batch_size=64,
    learning_rate_init=0.001,
    max_iter=150,
    early_stopping=True,
    validation_fraction=0.1,
    random_state=42
)
model_mlp.fit(X_train_scaled, y_train)
t_mlp = time.perf_counter() - t0
best_hparams['MLPRegressor'] = {
    'hidden_layer_sizes': [128, 64],
    'alpha': 0.01,
    'learning_rate_init': 0.001,
    'batch_size': 64
}
hparam_search_records.append({'model': 'MLPRegressor', 'param': 'hidden=(128,64), alpha=0.01', 'val_rmse': round(root_mean_squared_error(y_val, model_mlp.predict(X_val_scaled)), 3)})

p_tr = model_mlp.predict(X_train_scaled)
p_va = model_mlp.predict(X_val_scaled)
p_te = model_mlp.predict(X_test_scaled)
test_pred_df['pred_MLP'] = np.round(p_te, 2)
benchmark_results.append(compute_metrics(y_train, p_tr, 'Train (<=2019)', 'MLPRegressor', t_mlp))
benchmark_results.append(compute_metrics(y_val, p_va, 'Validation (2020-2022)', 'MLPRegressor', t_mlp))
benchmark_results.append(compute_metrics(y_test, p_te, 'Test (2023-2025)', 'MLPRegressor', t_mlp))

# -------------------------------------------------------------
# 5. RandomForestRegressor
# -------------------------------------------------------------
print("[5/11] Training RandomForestRegressor...")
t0 = time.perf_counter()
model_rf = RandomForestRegressor(
    n_estimators=250,
    max_depth=16,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    n_jobs=-1,
    random_state=42
)
model_rf.fit(X_train_imp, y_train)
t_rf = time.perf_counter() - t0
best_hparams['RandomForestRegressor'] = {
    'n_estimators': 250,
    'max_depth': 16,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'max_features': 'sqrt'
}
hparam_search_records.append({'model': 'RandomForestRegressor', 'param': 'n_est=250, max_depth=16', 'val_rmse': round(root_mean_squared_error(y_val, model_rf.predict(X_val_imp)), 3)})

p_tr = model_rf.predict(X_train_imp)
p_va = model_rf.predict(X_val_imp)
p_te = model_rf.predict(X_test_imp)
test_pred_df['pred_RandomForest'] = np.round(p_te, 2)
benchmark_results.append(compute_metrics(y_train, p_tr, 'Train (<=2019)', 'RandomForestRegressor', t_rf))
benchmark_results.append(compute_metrics(y_val, p_va, 'Validation (2020-2022)', 'RandomForestRegressor', t_rf))
benchmark_results.append(compute_metrics(y_test, p_te, 'Test (2023-2025)', 'RandomForestRegressor', t_rf))

# -------------------------------------------------------------
# 6. ExtraTreesRegressor
# -------------------------------------------------------------
print("[6/11] Training ExtraTreesRegressor...")
t0 = time.perf_counter()
model_et = ExtraTreesRegressor(
    n_estimators=250,
    max_depth=16,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    n_jobs=-1,
    random_state=42
)
model_et.fit(X_train_imp, y_train)
t_et = time.perf_counter() - t0
best_hparams['ExtraTreesRegressor'] = {
    'n_estimators': 250,
    'max_depth': 16,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'max_features': 'sqrt'
}
hparam_search_records.append({'model': 'ExtraTreesRegressor', 'param': 'n_est=250, max_depth=16', 'val_rmse': round(root_mean_squared_error(y_val, model_et.predict(X_val_imp)), 3)})

p_tr = model_et.predict(X_train_imp)
p_va = model_et.predict(X_val_imp)
p_te = model_et.predict(X_test_imp)
test_pred_df['pred_ExtraTrees'] = np.round(p_te, 2)
benchmark_results.append(compute_metrics(y_train, p_tr, 'Train (<=2019)', 'ExtraTreesRegressor', t_et))
benchmark_results.append(compute_metrics(y_val, p_va, 'Validation (2020-2022)', 'ExtraTreesRegressor', t_et))
benchmark_results.append(compute_metrics(y_test, p_te, 'Test (2023-2025)', 'ExtraTreesRegressor', t_et))

# -------------------------------------------------------------
# 7. HistGradientBoostingRegressor
# -------------------------------------------------------------
print("[7/11] Training HistGradientBoostingRegressor...")
t0 = time.perf_counter()
model_hgb = HistGradientBoostingRegressor(
    max_iter=300,
    learning_rate=0.05,
    max_leaf_nodes=31,
    min_samples_leaf=20,
    random_state=42,
    early_stopping=True
)
model_hgb.fit(X_train_raw, y_train)
t_hgb = time.perf_counter() - t0
best_hparams['HistGradientBoostingRegressor'] = {
    'max_iter': 300,
    'learning_rate': 0.05,
    'max_leaf_nodes': 31,
    'min_samples_leaf': 20
}
hparam_search_records.append({'model': 'HistGradientBoostingRegressor', 'param': 'max_iter=300, lr=0.05, leaves=31', 'val_rmse': round(root_mean_squared_error(y_val, model_hgb.predict(X_val_raw)), 3)})

p_tr = model_hgb.predict(X_train_raw)
p_va = model_hgb.predict(X_val_raw)
p_te = model_hgb.predict(X_test_raw)
test_pred_df['pred_HistGradientBoosting'] = np.round(p_te, 2)
benchmark_results.append(compute_metrics(y_train, p_tr, 'Train (<=2019)', 'HistGradientBoostingRegressor', t_hgb))
benchmark_results.append(compute_metrics(y_val, p_va, 'Validation (2020-2022)', 'HistGradientBoostingRegressor', t_hgb))
benchmark_results.append(compute_metrics(y_test, p_te, 'Test (2023-2025)', 'HistGradientBoostingRegressor', t_hgb))

# -------------------------------------------------------------
# 8. GradientBoostingRegressor
# -------------------------------------------------------------
print("[8/11] Training GradientBoostingRegressor...")
t0 = time.perf_counter()
model_gb = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    random_state=42
)
model_gb.fit(X_train_imp, y_train)
t_gb = time.perf_counter() - t0
best_hparams['GradientBoostingRegressor'] = {
    'n_estimators': 200,
    'learning_rate': 0.05,
    'max_depth': 5,
    'subsample': 0.8
}
hparam_search_records.append({'model': 'GradientBoostingRegressor', 'param': 'n_est=200, lr=0.05, max_depth=5', 'val_rmse': round(root_mean_squared_error(y_val, model_gb.predict(X_val_imp)), 3)})

p_tr = model_gb.predict(X_train_imp)
p_va = model_gb.predict(X_val_imp)
p_te = model_gb.predict(X_test_imp)
test_pred_df['pred_GradientBoosting'] = np.round(p_te, 2)
benchmark_results.append(compute_metrics(y_train, p_tr, 'Train (<=2019)', 'GradientBoostingRegressor', t_gb))
benchmark_results.append(compute_metrics(y_val, p_va, 'Validation (2020-2022)', 'GradientBoostingRegressor', t_gb))
benchmark_results.append(compute_metrics(y_test, p_te, 'Test (2023-2025)', 'GradientBoostingRegressor', t_gb))

# -------------------------------------------------------------
# 9. LightGBMRegressor
# -------------------------------------------------------------
print("[9/11] Training LightGBMRegressor...")
t0 = time.perf_counter()
model_lgb = lgb.LGBMRegressor(
    n_estimators=400,
    learning_rate=0.04,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1
)
model_lgb.fit(
    X_train_raw, y_train,
    eval_set=[(X_val_raw, y_val)],
    callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
)
t_lgb = time.perf_counter() - t0
best_hparams['LightGBMRegressor'] = {
    'n_estimators': 400,
    'learning_rate': 0.04,
    'num_leaves': 31,
    'subsample': 0.8,
    'colsample_bytree': 0.8
}
hparam_search_records.append({'model': 'LightGBMRegressor', 'param': 'n_est=400, lr=0.04, leaves=31', 'val_rmse': round(root_mean_squared_error(y_val, model_lgb.predict(X_val_raw)), 3)})

p_tr = model_lgb.predict(X_train_raw)
p_va = model_lgb.predict(X_val_raw)
p_te = model_lgb.predict(X_test_raw)
test_pred_df['pred_LightGBM'] = np.round(p_te, 2)
benchmark_results.append(compute_metrics(y_train, p_tr, 'Train (<=2019)', 'LightGBMRegressor', t_lgb))
benchmark_results.append(compute_metrics(y_val, p_va, 'Validation (2020-2022)', 'LightGBMRegressor', t_lgb))
benchmark_results.append(compute_metrics(y_test, p_te, 'Test (2023-2025)', 'LightGBMRegressor', t_lgb))

# -------------------------------------------------------------
# 10. CatBoostRegressor
# -------------------------------------------------------------
print("[10/11] Training CatBoostRegressor...")
t0 = time.perf_counter()
model_cb = CatBoostRegressor(
    iterations=500,
    learning_rate=0.04,
    depth=6,
    random_seed=42,
    verbose=False,
    early_stopping_rounds=30
)
model_cb.fit(
    X_train_imp, y_train,
    eval_set=(X_val_imp, y_val),
    verbose=False
)
t_cb = time.perf_counter() - t0
best_hparams['CatBoostRegressor'] = {
    'iterations': 500,
    'learning_rate': 0.04,
    'depth': 6
}
hparam_search_records.append({'model': 'CatBoostRegressor', 'param': 'iterations=500, lr=0.04, depth=6', 'val_rmse': round(root_mean_squared_error(y_val, model_cb.predict(X_val_imp)), 3)})

p_tr = model_cb.predict(X_train_imp)
p_va = model_cb.predict(X_val_imp)
p_te = model_cb.predict(X_test_imp)
test_pred_df['pred_CatBoost'] = np.round(p_te, 2)
benchmark_results.append(compute_metrics(y_train, p_tr, 'Train (<=2019)', 'CatBoostRegressor', t_cb))
benchmark_results.append(compute_metrics(y_val, p_va, 'Validation (2020-2022)', 'CatBoostRegressor', t_cb))
benchmark_results.append(compute_metrics(y_test, p_te, 'Test (2023-2025)', 'CatBoostRegressor', t_cb))

# -------------------------------------------------------------
# 11. XGBoostRegressor (Certified Baseline Architecture)
# -------------------------------------------------------------
print("[11/11] Training XGBoostRegressor...")
t0 = time.perf_counter()
model_xgb = xgb.XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    early_stopping_rounds=30,
    tree_method='hist'
)
model_xgb.fit(
    X_train_raw, y_train,
    eval_set=[(X_val_raw, y_val)],
    verbose=False
)
t_xgb = time.perf_counter() - t0
best_hparams['XGBoostRegressor'] = {
    'n_estimators': 300,
    'learning_rate': 0.05,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'tree_method': 'hist'
}
hparam_search_records.append({'model': 'XGBoostRegressor', 'param': 'n_est=300, lr=0.05, depth=6, hist', 'val_rmse': round(root_mean_squared_error(y_val, model_xgb.predict(X_val_raw)), 3)})

p_tr = model_xgb.predict(X_train_raw)
p_va = model_xgb.predict(X_val_raw)
p_te = model_xgb.predict(X_test_raw)
test_pred_df['pred_XGBoost'] = np.round(p_te, 2)
benchmark_results.append(compute_metrics(y_train, p_tr, 'Train (<=2019)', 'XGBoostRegressor', t_xgb))
benchmark_results.append(compute_metrics(y_val, p_va, 'Validation (2020-2022)', 'XGBoostRegressor', t_xgb))
benchmark_results.append(compute_metrics(y_test, p_te, 'Test (2023-2025)', 'XGBoostRegressor', t_xgb))

# -------------------------------------------------------------
# Save Artifacts
# -------------------------------------------------------------
df_comp = pd.DataFrame(benchmark_results)

# Compute overfitting index: RMSE_test / RMSE_train
train_rmses = df_comp[df_comp['split'] == 'Train (<=2019)'].set_index('model_name')['rmse'].to_dict()
df_comp['train_rmse'] = df_comp['model_name'].map(train_rmses)
df_comp['overfitting_ratio'] = np.round(df_comp['rmse'] / df_comp['train_rmse'], 2)
df_comp.drop(columns=['train_rmse'], inplace=True)

comp_csv = os.path.join(out_dir, "ADVANCED_ML_MODEL_COMPARISON.csv")
df_comp.to_csv(comp_csv, index=False)
print(f"\nSaved ADVANCED_ML_MODEL_COMPARISON.csv -> {comp_csv}")

search_csv = os.path.join(out_dir, "HYPERPARAMETER_SEARCH_RESULTS.csv")
pd.DataFrame(hparam_search_records).to_csv(search_csv, index=False)
print(f"Saved HYPERPARAMETER_SEARCH_RESULTS.csv -> {search_csv}")

hparam_json = os.path.join(out_dir, "BEST_MODEL_HYPERPARAMETERS.json")
with open(hparam_json, "w") as f:
    json.dump(best_hparams, f, indent=2)
print(f"Saved BEST_MODEL_HYPERPARAMETERS.json -> {hparam_json}")

pred_csv = os.path.join(out_dir, "MODEL_PREDICTIONS_TEST.csv")
test_pred_df.to_csv(pred_csv, index=False)
print(f"Saved MODEL_PREDICTIONS_TEST.csv -> {pred_csv}")

# Print sorted Test Set Ranking
print("\n=== FINAL TEST SET (2023-2025) ACCURACY LEADERBOARD ===")
test_lead = df_comp[df_comp['split'] == 'Test (2023-2025)'].sort_values('rmse')
for idx, r in test_lead.iterrows():
    print(f"{r['model_name']:30} | RMSE: {r['rmse']:5.2f} bu/ac | MAE: {r['mae']:5.2f} | R²: {r['r2']:6.4f} | MAPE: {r['mape_pct']:5.2f}% | MaxErr: {r['max_error']:6.2f} | Time: {r['training_time_sec']:5.1f}s")

