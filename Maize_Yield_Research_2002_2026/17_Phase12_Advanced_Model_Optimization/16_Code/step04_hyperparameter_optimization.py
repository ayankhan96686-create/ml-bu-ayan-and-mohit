"""
Phase 12 - Step 04: Advanced Hyperparameter Optimization
Script: 16_Code/step04_hyperparameter_optimization.py
Generates:
  08_Hyperparameter_Optimization/OPTUNA_SEARCH_RESULTS.csv
  08_Hyperparameter_Optimization/BEST_MODEL_PARAMETERS.json
  08_Hyperparameter_Optimization/MODEL_TUNING_REPORT.md
  08_Hyperparameter_Optimization/TEMPORAL_TUNING_CONFIGURATION.json
"""

import os
import json
import itertools
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

PROJECT_ROOT = r"E:\ml do not open\Maize_Yield_Research_2002_2026"
PHASE12_ROOT = os.path.join(PROJECT_ROOT, "17_Phase12_Advanced_Model_Optimization")
OUT_DIR = os.path.join(PHASE12_ROOT, "08_Hyperparameter_Optimization")
os.makedirs(OUT_DIR, exist_ok=True)

print("--- Starting Phase 12 Step 04: Advanced Hyperparameter Optimization ---")

# Load Phase 12 Historical Feature Dataset
hist_path = os.path.join(PHASE12_ROOT, "13_Final_Model", "Historical_Features_Phase12.csv")
df_hist = pd.read_csv(hist_path)

# Comprehensive Feature Set (quarantining leakage)
exclude_cols = [
    "county_fips", "state_fips", "state", "state_name", "county_name",
    "year", "corn_yield_bu_acre", "eligible_for_2026_forecast", "county_fips_5digit",
    "noaa_prcp_annual_in", "cv_percent"
]
feature_cols = [c for c in df_hist.columns if c not in exclude_cols]

print(f"Total candidate predictors for hyperparameter tuning: {len(feature_cols)}")

# Define Expanding-Window Temporal Folds (Strictly within 2002-2022)
folds = [
    {"fold": 1, "train_years": list(range(2002, 2017)), "val_years": [2017, 2018]},
    {"fold": 2, "train_years": list(range(2002, 2019)), "val_years": [2019, 2020]},
    {"fold": 3, "train_years": list(range(2002, 2021)), "val_years": [2021, 2022]}
]

with open(os.path.join(OUT_DIR, "TEMPORAL_TUNING_CONFIGURATION.json"), "w") as f:
    json.dump({"description": "Expanding-Window Temporal Validation Folds (2002-2022)", "folds": folds}, f, indent=2)

def evaluate_cv(model_fn):
    scores = []
    for f in folds:
        tr_mask = df_hist["year"].isin(f["train_years"])
        val_mask = df_hist["year"].isin(f["val_years"])
        
        X_tr = df_hist.loc[tr_mask, feature_cols].fillna(df_hist.loc[tr_mask, feature_cols].median()).fillna(0.0)
        y_tr = df_hist.loc[tr_mask, "corn_yield_bu_acre"].values
        
        X_val = df_hist.loc[val_mask, feature_cols].fillna(df_hist.loc[tr_mask, feature_cols].median()).fillna(0.0)
        y_val = df_hist.loc[val_mask, "corn_yield_bu_acre"].values
        
        model = model_fn()
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        scores.append(rmse)
    return float(np.mean(scores))

search_records = []
best_params = {}

# 1. TUNE LIGHTGBM
print("\nTuning LightGBM Regressor...")
lgb_grid = [
    {"n_estimators": 250, "learning_rate": 0.03, "num_leaves": 31, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 1.0},
    {"n_estimators": 350, "learning_rate": 0.02, "num_leaves": 45, "subsample": 0.8, "colsample_bytree": 0.7, "reg_alpha": 0.5, "reg_lambda": 2.0},
    {"n_estimators": 400, "learning_rate": 0.03, "num_leaves": 63, "subsample": 0.85, "colsample_bytree": 0.8, "reg_alpha": 1.0, "reg_lambda": 5.0},
    {"n_estimators": 300, "learning_rate": 0.04, "num_leaves": 31, "subsample": 0.75, "colsample_bytree": 0.75, "reg_alpha": 0.1, "reg_lambda": 0.5}
]
best_lgb_score = 999.0
best_lgb_p = None

for i, p in enumerate(lgb_grid):
    score = evaluate_cv(lambda: lgb.LGBMRegressor(**p, random_state=42, verbose=-1))
    search_records.append({"Algorithm": "LightGBM", "Trial": i+1, "Parameters": json.dumps(p), "Mean_CV_RMSE_bu_ac": round(score, 4)})
    print(f"  Trial {i+1}: CV RMSE = {score:.4f} bu/ac")
    if score < best_lgb_score:
        best_lgb_score = score
        best_lgb_p = p

best_params["LightGBM"] = {"params": best_lgb_p, "cv_rmse": round(best_lgb_score, 4)}

# 2. TUNE XGBOOST
print("\nTuning XGBoost Regressor...")
xgb_grid = [
    {"n_estimators": 250, "learning_rate": 0.03, "max_depth": 6, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 1.0},
    {"n_estimators": 350, "learning_rate": 0.02, "max_depth": 7, "subsample": 0.85, "colsample_bytree": 0.75, "reg_alpha": 0.5, "reg_lambda": 2.0},
    {"n_estimators": 300, "learning_rate": 0.03, "max_depth": 5, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 1.0, "reg_lambda": 3.0}
]
best_xgb_score = 999.0
best_xgb_p = None

for i, p in enumerate(xgb_grid):
    score = evaluate_cv(lambda: xgb.XGBRegressor(**p, random_state=42, verbosity=0))
    search_records.append({"Algorithm": "XGBoost", "Trial": i+1, "Parameters": json.dumps(p), "Mean_CV_RMSE_bu_ac": round(score, 4)})
    print(f"  Trial {i+1}: CV RMSE = {score:.4f} bu/ac")
    if score < best_xgb_score:
        best_xgb_score = score
        best_xgb_p = p

best_params["XGBoost"] = {"params": best_xgb_p, "cv_rmse": round(best_xgb_score, 4)}

# 3. TUNE CATBOOST
print("\nTuning CatBoost Regressor...")
cat_grid = [
    {"iterations": 300, "learning_rate": 0.04, "depth": 6, "l2_leaf_reg": 3.0},
    {"iterations": 400, "learning_rate": 0.03, "depth": 7, "l2_leaf_reg": 5.0},
    {"iterations": 350, "learning_rate": 0.03, "depth": 6, "l2_leaf_reg": 4.0}
]
best_cat_score = 999.0
best_cat_p = None

for i, p in enumerate(cat_grid):
    score = evaluate_cv(lambda: CatBoostRegressor(**p, random_seed=42, verbose=0))
    search_records.append({"Algorithm": "CatBoost", "Trial": i+1, "Parameters": json.dumps(p), "Mean_CV_RMSE_bu_ac": round(score, 4)})
    print(f"  Trial {i+1}: CV RMSE = {score:.4f} bu/ac")
    if score < best_cat_score:
        best_cat_score = score
        best_cat_p = p

best_params["CatBoost"] = {"params": best_cat_p, "cv_rmse": round(best_cat_score, 4)}

# 4. TUNE HISTGRADIENTBOOSTING
print("\nTuning HistGradientBoosting Regressor...")
hgb_grid = [
    {"max_iter": 250, "learning_rate": 0.03, "max_leaf_nodes": 31, "l2_regularization": 1.0},
    {"max_iter": 350, "learning_rate": 0.02, "max_leaf_nodes": 45, "l2_regularization": 2.0}
]
best_hgb_score = 999.0
best_hgb_p = None

for i, p in enumerate(hgb_grid):
    score = evaluate_cv(lambda: HistGradientBoostingRegressor(**p, random_state=42))
    search_records.append({"Algorithm": "HistGradientBoosting", "Trial": i+1, "Parameters": json.dumps(p), "Mean_CV_RMSE_bu_ac": round(score, 4)})
    print(f"  Trial {i+1}: CV RMSE = {score:.4f} bu/ac")
    if score < best_hgb_score:
        best_hgb_score = score
        best_hgb_p = p

best_params["HistGradientBoosting"] = {"params": best_hgb_p, "cv_rmse": round(best_hgb_score, 4)}

# 5. TUNE MLP
print("\nTuning Multilayer Perceptron (MLP)...")
mlp_grid = [
    {"hidden_layer_sizes": (128, 64), "alpha": 0.01, "learning_rate_init": 0.001, "max_iter": 200},
    {"hidden_layer_sizes": (64, 32), "alpha": 0.1, "learning_rate_init": 0.002, "max_iter": 200}
]
best_mlp_score = float("inf")
best_mlp_p = mlp_grid[0]

for i, p in enumerate(mlp_grid):
    score = evaluate_cv(lambda: make_pipeline(StandardScaler(), MLPRegressor(**p, random_state=42, early_stopping=True)))
    search_records.append({"Algorithm": "MLP", "Trial": i+1, "Parameters": json.dumps(p), "Mean_CV_RMSE_bu_ac": round(score, 4)})
    print(f"  Trial {i+1}: CV RMSE = {score:.4f} bu/ac")
    if score < best_mlp_score:
        best_mlp_score = score
        best_mlp_p = p

best_params["MLP"] = {"params": best_mlp_p, "cv_rmse": round(best_mlp_score, 4)}

# Save outputs
pd.DataFrame(search_records).to_csv(os.path.join(OUT_DIR, "OPTUNA_SEARCH_RESULTS.csv"), index=False)
with open(os.path.join(OUT_DIR, "BEST_MODEL_PARAMETERS.json"), "w") as f:
    json.dump(best_params, f, indent=2)

tuning_md = f"""# ADVANCED HYPERPARAMETER OPTIMIZATION REPORT

## Methodology & Validation Guardrails
To prevent data leakage into the prospective evaluation benchmarks:
1. All hyperparameter tuning was conducted strictly within the **2002–2022 historical training period**.
2. An **expanding-window temporal cross-validation protocol** (3 multi-year folds: 2017–2018, 2019–2020, 2021–2022) was utilized.
3. The 2023–2025 final benchmark was strictly withheld during hyperparameter selection.

## Best Hyperparameter Configurations & Validation Scores

| Algorithm | Best Configuration | Expanding-Window CV RMSE (bu/ac) |
| :--- | :--- | :---: |
| **LightGBM** | `learning_rate`: {best_lgb_p['learning_rate']}, `num_leaves`: {best_lgb_p['num_leaves']}, `n_estimators`: {best_lgb_p['n_estimators']}, `reg_alpha`: {best_lgb_p['reg_alpha']} | **{best_lgb_score:.4f}** |
| **XGBoost** | `learning_rate`: {best_xgb_p['learning_rate']}, `max_depth`: {best_xgb_p['max_depth']}, `n_estimators`: {best_xgb_p['n_estimators']}, `reg_lambda`: {best_xgb_p['reg_lambda']} | **{best_xgb_score:.4f}** |
| **CatBoost** | `learning_rate`: {best_cat_p['learning_rate']}, `depth`: {best_cat_p['depth']}, `iterations`: {best_cat_p['iterations']}, `l2_leaf_reg`: {best_cat_p['l2_leaf_reg']} | **{best_cat_score:.4f}** |
| **HistGradientBoosting** | `learning_rate`: {best_hgb_p['learning_rate']}, `max_leaf_nodes`: {best_hgb_p['max_leaf_nodes']}, `max_iter`: {best_hgb_p['max_iter']} | **{best_hgb_score:.4f}** |
| **MLP** | `hidden_layer_sizes`: {best_mlp_p['hidden_layer_sizes']}, `alpha`: {best_mlp_p['alpha']}, `learning_rate_init`: {best_mlp_p['learning_rate_init']} | **{best_mlp_score:.4f}** |
"""
with open(os.path.join(OUT_DIR, "MODEL_TUNING_REPORT.md"), "w", encoding="utf-8") as f:
    f.write(tuning_md)

print("\nSaved all Hyperparameter Optimization deliverables in 08_Hyperparameter_Optimization/")
print("--- Phase 12 Step 04 Completed Successfully ---")
