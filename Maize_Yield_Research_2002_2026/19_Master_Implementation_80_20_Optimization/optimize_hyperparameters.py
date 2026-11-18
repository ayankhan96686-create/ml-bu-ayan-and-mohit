"""
19_Master_Implementation_80_20_Optimization/optimize_hyperparameters.py
========================================================================
STEP 5: Hyperparameter Optimization Using Optuna Across 4 Chronological Folds
Tuning performed EXCLUSIVELY on Primary 80% Training Set (N = 13,112).
Objective: Minimize Mean Chronological Validation RMSE across Folds 1-4.
Generates:
  - 19_Master_Implementation_80_20_Optimization/OPTUNA_TRIAL_HISTORY.csv
  - 19_Master_Implementation_80_20_Optimization/BEST_HYPERPARAMETERS.json
  - 19_Master_Implementation_80_20_Optimization/HYPERPARAMETER_OPTIMIZATION_REPORT.md
"""

import os
import sys
import time
import json
import optuna
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
)
from sklearn.svm import LinearSVR
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

optuna.logging.set_verbosity(optuna.logging.WARNING)

PROJECT_ROOT = r"E:\ml do not open\Maize_Yield_Research_2002_2026"
OUT_DIR = os.path.join(PROJECT_ROOT, "19_Master_Implementation_80_20_Optimization")
os.makedirs(OUT_DIR, exist_ok=True)

DATA_PATH = os.path.join(PROJECT_ROOT, "17_Phase12_Advanced_Model_Optimization", "13_Final_Model", "Historical_Features_Phase12.csv")
MANIFEST_PATH = os.path.join(PROJECT_ROOT, "17_Phase12_Advanced_Model_Optimization", "13_Final_Model", "FINAL_FEATURE_MANIFEST.csv")
OBS_PATH = os.path.join(OUT_DIR, "TRAIN_TEST_OBSERVATION_IDS_POOLED.csv")
FOLDS_CONFIG_PATH = os.path.join(OUT_DIR, "VALIDATION_FOLDS_CONFIG.json")


def run_tuning():
    print("=" * 80, flush=True)
    print("--- [STEP 5] RUNNING HYPERPARAMETER OPTIMIZATION (OPTUNA) ---", flush=True)
    print("=" * 80, flush=True)
    
    df = pd.read_csv(DATA_PATH)
    obs = pd.read_csv(OBS_PATH)
    mf = pd.read_csv(MANIFEST_PATH)
    feature_cols = mf['Feature_Name'].tolist()
    
    with open(FOLDS_CONFIG_PATH, 'r') as f:
        folds_info = json.load(f)['folds']
        
    df = df.merge(obs[['county_fips', 'year', 'split_assignment']], on=['county_fips', 'year'])
    tr_df = df[df['split_assignment'] == 'TRAIN_80'].copy().reset_index(drop=True)
    
    # Pre-impute and scale per training set
    tr_meds = tr_df[feature_cols].median()
    X_tr = tr_df[feature_cols].fillna(tr_meds).fillna(0.0)
    y_tr = tr_df['corn_yield_bu_acre'].values
    
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    
    all_trials = []
    best_params = {}
    
    # Pre-extract fold indices to maximize execution speed
    cached_folds = []
    for f in folds_info:
        tr_b = tr_df['year'].isin(f['train_years']).values
        va_b = tr_df['year'].isin(f['val_years']).values
        cached_folds.append((tr_b, va_b))
        
    def evaluate_cv(model, is_scaled=False):
        X_mat = X_tr_s if is_scaled else X_tr
        rmses, maes, r2s, biases = [], [], [], []
        for tr_b, va_b in cached_folds:
            X_f_tr = X_mat[tr_b] if isinstance(X_mat, np.ndarray) else X_mat.iloc[tr_b]
            y_f_tr = y_tr[tr_b]
            X_f_va = X_mat[va_b] if isinstance(X_mat, np.ndarray) else X_mat.iloc[va_b]
            y_f_va = y_tr[va_b]
            
            from sklearn.base import clone
            m_c = clone(model)
            m_c.fit(X_f_tr, y_f_tr)
            p_va = m_c.predict(X_f_va)
            
            rmses.append(np.sqrt(mean_squared_error(y_f_va, p_va)))
            maes.append(mean_absolute_error(y_f_va, p_va))
            r2s.append(r2_score(y_f_va, p_va))
            biases.append(np.mean(p_va - y_f_va))
            
        return float(np.mean(rmses)), float(np.mean(maes)), float(np.mean(r2s)), float(np.mean(biases))

    # --- 1. LightGBM Optimization ---
    print("\n[1/6] Optimizing LightGBM with Optuna (30 trials)...", flush=True)
    def obj_lgb(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 600, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.08, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 63),
            "subsample": trial.suggest_float("subsample", 0.75, 0.95),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.70, 0.90),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 10.0, log=True),
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1
        }
        m = lgb.LGBMRegressor(**params)
        rmse, mae, r2, bias = evaluate_cv(m, is_scaled=False)
        all_trials.append({
            "Trial_ID": len(all_trials) + 1, "Model": "LightGBM", "Params": json.dumps(params),
            "Val_RMSE": round(rmse, 2), "Val_MAE": round(mae, 2), "Val_R2": round(r2, 4), "Val_Bias": round(bias, 2)
        })
        return rmse

    study_lgb = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study_lgb.optimize(obj_lgb, n_trials=30)
    best_lgb_p = study_lgb.best_params
    best_lgb_p.update({"random_state": 42, "n_jobs": -1, "verbose": -1})
    best_params["LightGBM"] = {"params": best_lgb_p, "best_val_rmse": round(study_lgb.best_value, 2)}
    print(f"  Best LightGBM Val RMSE: {study_lgb.best_value:.2f}", flush=True)

    # --- 2. XGBoost Optimization (with tree_method='hist') ---
    print("\n[2/6] Optimizing XGBoost with Optuna (25 trials)...", flush=True)
    def obj_xgb(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 250, 500, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.08, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 7),
            "subsample": trial.suggest_float("subsample", 0.75, 0.95),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.70, 0.90),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 10.0, log=True),
            "tree_method": "hist",
            "random_state": 42,
            "n_jobs": -1
        }
        m = xgb.XGBRegressor(**params)
        rmse, mae, r2, bias = evaluate_cv(m, is_scaled=False)
        all_trials.append({
            "Trial_ID": len(all_trials) + 1, "Model": "XGBoost", "Params": json.dumps(params),
            "Val_RMSE": round(rmse, 2), "Val_MAE": round(mae, 2), "Val_R2": round(r2, 4), "Val_Bias": round(bias, 2)
        })
        return rmse

    study_xgb = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study_xgb.optimize(obj_xgb, n_trials=25)
    best_xgb_p = study_xgb.best_params
    best_xgb_p.update({"tree_method": "hist", "random_state": 42, "n_jobs": -1})
    best_params["XGBoost"] = {"params": best_xgb_p, "best_val_rmse": round(study_xgb.best_value, 2)}
    print(f"  Best XGBoost Val RMSE: {study_xgb.best_value:.2f}", flush=True)

    # --- 3. CatBoost Optimization ---
    print("\n[3/6] Optimizing CatBoost with Optuna (15 trials)...", flush=True)
    def obj_cat(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 250, 450, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.08, log=True),
            "depth": trial.suggest_int("depth", 5, 7),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 2.0, 8.0),
            "random_seed": 42,
            "verbose": 0,
            "thread_count": -1
        }
        m = CatBoostRegressor(**params)
        rmse, mae, r2, bias = evaluate_cv(m, is_scaled=False)
        all_trials.append({
            "Trial_ID": len(all_trials) + 1, "Model": "CatBoost", "Params": json.dumps(params),
            "Val_RMSE": round(rmse, 2), "Val_MAE": round(mae, 2), "Val_R2": round(r2, 4), "Val_Bias": round(bias, 2)
        })
        return rmse

    study_cat = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study_cat.optimize(obj_cat, n_trials=15)
    best_cat_p = study_cat.best_params
    best_cat_p.update({"random_seed": 42, "verbose": 0, "thread_count": -1})
    best_params["CatBoost"] = {"params": best_cat_p, "best_val_rmse": round(study_cat.best_value, 2)}
    print(f"  Best CatBoost Val RMSE: {study_cat.best_value:.2f}", flush=True)

    # --- 4. HistGradientBoosting Optimization ---
    print("\n[4/6] Optimizing HistGradientBoosting with Optuna (15 trials)...", flush=True)
    def obj_hgb(trial):
        params = {
            "max_iter": trial.suggest_int("max_iter", 250, 450, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.06, log=True),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 31, 55),
            "l2_regularization": trial.suggest_float("l2_regularization", 0.5, 4.0),
            "random_state": 42
        }
        m = HistGradientBoostingRegressor(**params)
        rmse, mae, r2, bias = evaluate_cv(m, is_scaled=False)
        all_trials.append({
            "Trial_ID": len(all_trials) + 1, "Model": "HistGB", "Params": json.dumps(params),
            "Val_RMSE": round(rmse, 2), "Val_MAE": round(mae, 2), "Val_R2": round(r2, 4), "Val_Bias": round(bias, 2)
        })
        return rmse

    study_hgb = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study_hgb.optimize(obj_hgb, n_trials=15)
    best_hgb_p = study_hgb.best_params
    best_hgb_p.update({"random_state": 42})
    best_params["HistGradientBoosting"] = {"params": best_hgb_p, "best_val_rmse": round(study_hgb.best_value, 2)}
    print(f"  Best HistGradientBoosting Val RMSE: {study_hgb.best_value:.2f}", flush=True)

    # --- 5. ElasticNet Optimization ---
    print("\n[5/6] Optimizing ElasticNet with Optuna (20 trials)...", flush=True)
    def obj_enet(trial):
        alpha = trial.suggest_float("alpha", 1e-4, 5.0, log=True)
        l1_ratio = trial.suggest_float("l1_ratio", 0.1, 0.9)
        params = {"alpha": alpha, "l1_ratio": l1_ratio, "max_iter": 2000, "tol": 1e-3, "random_state": 42}
        m = ElasticNet(**params)
        rmse, mae, r2, bias = evaluate_cv(m, is_scaled=True)
        all_trials.append({
            "Trial_ID": len(all_trials) + 1, "Model": "ElasticNet", "Params": json.dumps(params),
            "Val_RMSE": round(rmse, 2), "Val_MAE": round(mae, 2), "Val_R2": round(r2, 4), "Val_Bias": round(bias, 2)
        })
        return rmse

    study_enet = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study_enet.optimize(obj_enet, n_trials=20)
    best_enet_p = study_enet.best_params
    best_enet_p.update({"max_iter": 2000, "tol": 1e-3, "random_state": 42})
    best_params["ElasticNet"] = {"params": best_enet_p, "best_val_rmse": round(study_enet.best_value, 2)}
    print(f"  Best ElasticNet Val RMSE: {study_enet.best_value:.2f}", flush=True)

    # --- 6. Ridge, Lasso, SVR Grid Checks ---
    print("\n[6/6] Fine-Tuning Ridge, Lasso, and SVR...", flush=True)
    for a in [5.0, 20.0, 50.0, 100.0, 200.0]:
        m = Ridge(alpha=a, random_state=42)
        rmse, mae, r2, bias = evaluate_cv(m, is_scaled=True)
        all_trials.append({
            "Trial_ID": len(all_trials) + 1, "Model": "Ridge", "Params": json.dumps({"alpha": a}),
            "Val_RMSE": round(rmse, 2), "Val_MAE": round(mae, 2), "Val_R2": round(r2, 4), "Val_Bias": round(bias, 2)
        })
    best_ridge_row = [t for t in all_trials if t["Model"] == "Ridge"]
    best_ridge = min(best_ridge_row, key=lambda x: x["Val_RMSE"])
    best_params["Ridge"] = {"params": json.loads(best_ridge["Params"]), "best_val_rmse": best_ridge["Val_RMSE"]}
    print(f"  Best Ridge Val RMSE: {best_ridge['Val_RMSE']:.2f}", flush=True)

    for a in [0.01, 0.05, 0.1, 0.2, 0.5]:
        m = Lasso(alpha=a, max_iter=2000, tol=1e-3, random_state=42)
        rmse, mae, r2, bias = evaluate_cv(m, is_scaled=True)
        all_trials.append({
            "Trial_ID": len(all_trials) + 1, "Model": "Lasso", "Params": json.dumps({"alpha": a}),
            "Val_RMSE": round(rmse, 2), "Val_MAE": round(mae, 2), "Val_R2": round(r2, 4), "Val_Bias": round(bias, 2)
        })
    best_lasso_row = [t for t in all_trials if t["Model"] == "Lasso"]
    best_lasso = min(best_lasso_row, key=lambda x: x["Val_RMSE"])
    best_params["Lasso"] = {"params": json.loads(best_lasso["Params"]), "best_val_rmse": best_lasso["Val_RMSE"]}
    print(f"  Best Lasso Val RMSE: {best_lasso['Val_RMSE']:.2f}", flush=True)

    for c in [0.2, 0.5, 1.0, 2.0, 5.0]:
        m = LinearSVR(C=c, max_iter=5000, random_state=42)
        rmse, mae, r2, bias = evaluate_cv(m, is_scaled=True)
        all_trials.append({
            "Trial_ID": len(all_trials) + 1, "Model": "SVR", "Params": json.dumps({"C": c}),
            "Val_RMSE": round(rmse, 2), "Val_MAE": round(mae, 2), "Val_R2": round(r2, 4), "Val_Bias": round(bias, 2)
        })
    best_svr_row = [t for t in all_trials if t["Model"] == "SVR"]
    best_svr = min(best_svr_row, key=lambda x: x["Val_RMSE"])
    best_params["SVR"] = {"params": json.loads(best_svr["Params"]), "best_val_rmse": best_svr["Val_RMSE"]}
    print(f"  Best SVR Val RMSE: {best_svr['Val_RMSE']:.2f}", flush=True)

    # Export Trials CSV
    trials_df = pd.DataFrame(all_trials)
    trials_csv = os.path.join(OUT_DIR, "OPTUNA_TRIAL_HISTORY.csv")
    trials_df.to_csv(trials_csv, index=False)
    print(f"\nSaved {len(trials_df)} optimization trials to {trials_csv}", flush=True)

    # Export Best Hyperparameters JSON
    best_json = os.path.join(OUT_DIR, "BEST_HYPERPARAMETERS.json")
    with open(best_json, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"Saved best model parameters to {best_json}", flush=True)

    # Export Markdown Report
    report_md = os.path.join(OUT_DIR, "HYPERPARAMETER_OPTIMIZATION_REPORT.md")
    with open(report_md, "w", encoding="utf-8") as f:
        f.write("# HYPERPARAMETER OPTIMIZATION REPORT (OPTUNA & CHRONOLOGICAL FOLDS)\n\n")
        f.write("## 1. Optimization Protocol & Objective Function\n\n")
        f.write("All hyperparameter searches were conducted strictly inside the **Primary 80% Training Partition** ($N = 13,112$) across the 4 expanding-window chronological validation folds. ")
        f.write("The primary loss objective was **Mean Chronological Validation RMSE** across Folds 1–4. The 20% testing holdout remained completely quarantined.\n\n")
        f.write("## 2. Summary of Selected Hyperparameters\n\n")
        f.write("| Model Architecture | Best Validation RMSE (bu/ac) | Key Tuned Hyperparameters |\n")
        f.write("| :--- | :---: | :--- |\n")
        for m_name, info in best_params.items():
            f.write(f"| **{m_name}** | {info['best_val_rmse']:.2f} | `{json.dumps(info['params'])}` |\n")
            
        f.write("\n\n## 3. Top Trial History by Architecture\n\n")
        top_trials = trials_df.sort_values(by="Val_RMSE", ascending=True).groupby("Model").head(3).reset_index(drop=True)
        f.write(top_trials[['Model', 'Val_RMSE', 'Val_MAE', 'Val_R2', 'Val_Bias', 'Params']].to_markdown(index=False))
        f.write("\n")
    print(f"Saved hyperparameter optimization report to {report_md}", flush=True)


if __name__ == "__main__":
    run_tuning()
