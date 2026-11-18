import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

gate_dir = r"E:\ml do not open\Maize_Yield_Research_2002_2026\10_Final_Pre_Modeling_Gate"
p_train = os.path.join(gate_dir, "Purified_Corn_Belt_Historical_Dataset.csv")
p_preds = os.path.join(gate_dir, "baseline_predictions_val_test.csv")

print("Executing Phase 2: Model Reliability, Geographic Generalization, and Error Analysis...")

df = pd.read_csv(p_train)
preds_df = pd.read_csv(p_preds)

# 1. Error Analysis by Year, State, and History Length
# Residuals for best model: pred_xgboost_SetD
preds_df['residual'] = preds_df['corn_yield_bu_acre'] - preds_df['pred_xgboost_SetD']
preds_df['abs_error'] = preds_df['residual'].abs()
preds_df['sq_error'] = preds_df['residual'] ** 2

# Yearly Error Breakdown
yearly_stats = []
for yr, grp in preds_df.groupby('year'):
    rmse = np.sqrt(grp['sq_error'].mean())
    mae = grp['abs_error'].mean()
    r2 = r2_score(grp['corn_yield_bu_acre'], grp['pred_xgboost_SetD'])
    mean_act = grp['corn_yield_bu_acre'].mean()
    mean_pred = grp['pred_xgboost_SetD'].mean()
    yearly_stats.append({
        'group_type': 'Year',
        'group_name': str(yr),
        'n_counties': len(grp),
        'actual_mean_yield': round(mean_act, 2),
        'predicted_mean_yield': round(mean_pred, 2),
        'bias': round(mean_pred - mean_act, 2),
        'rmse': round(rmse, 2),
        'mae': round(mae, 2),
        'r2': round(r2, 4)
    })

# State Error Breakdown (on Test set 2023-2025)
test_preds = preds_df[preds_df['year'] >= 2023]
for st, grp in test_preds.groupby('state'):
    rmse = np.sqrt(grp['sq_error'].mean())
    mae = grp['abs_error'].mean()
    r2 = r2_score(grp['corn_yield_bu_acre'], grp['pred_xgboost_SetD'])
    mean_act = grp['corn_yield_bu_acre'].mean()
    mean_pred = grp['pred_xgboost_SetD'].mean()
    yearly_stats.append({
        'group_type': 'State (2023-2025 Test)',
        'group_name': st,
        'n_counties': len(grp),
        'actual_mean_yield': round(mean_act, 2),
        'predicted_mean_yield': round(mean_pred, 2),
        'bias': round(mean_pred - mean_act, 2),
        'rmse': round(rmse, 2),
        'mae': round(mae, 2),
        'r2': round(r2, 4)
    })

df_error_analysis = pd.DataFrame(yearly_stats)
p_error_csv = os.path.join(gate_dir, "STATE_YEAR_ERROR_ANALYSIS.csv")
df_error_analysis.to_csv(p_error_csv, index=False)
print(f"Saved STATE_YEAR_ERROR_ANALYSIS.csv -> {p_error_csv}")

# 2. Geographic Generalization Experiment (Spatial Cross-Validation)
# Evaluate on unseen geographical zones (Leave-One-State-Out CV on 2002-2022 data)
soil_cols = [c for c in df.columns if c.startswith('soil_') and c not in ['soil_intptlat', 'soil_intptlon']]
noaa_cols = [c for c in df.columns if c.startswith('noaa_') and c != 'noaa_prcp_annual_in']
features_D = sorted(soil_cols + noaa_cols + ['yield_lag_1yr', 'yield_lag_2yr', 'yield_roll_mean_5yr'])

train_val_df = df[df['year'] <= 2022].copy()
states = sorted(train_val_df['state'].unique())

loso_results = []
print("\nRunning Leave-One-State-Out Geographic Generalization...")
for test_state in states:
    loso_train = train_val_df[train_val_df['state'] != test_state]
    loso_test = train_val_df[train_val_df['state'] == test_state]
    
    X_tr = loso_train[features_D]
    y_tr = loso_train['corn_yield_bu_acre']
    X_te = loso_test[features_D]
    y_te = loso_test['corn_yield_bu_acre']
    
    model = xgb.XGBRegressor(
        n_estimators=150,
        learning_rate=0.08,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        tree_method='hist'
    )
    model.fit(X_tr, y_tr, verbose=False)
    preds = model.predict(X_te)
    
    rmse = root_mean_squared_error(y_te, preds)
    mae = mean_absolute_error(y_te, preds)
    r2 = r2_score(y_te, preds)
    
    loso_results.append({
        'held_out_state': test_state,
        'test_counties_count': loso_test['county_fips'].nunique(),
        'test_observations_count': len(loso_test),
        'state_mean_yield': round(y_te.mean(), 2),
        'predicted_mean_yield': round(preds.mean(), 2),
        'bias': round(preds.mean() - y_te.mean(), 2),
        'spatial_holdout_rmse': round(rmse, 2),
        'spatial_holdout_mae': round(mae, 2),
        'spatial_holdout_r2': round(r2, 4)
    })
    print(f"Held-out State: {test_state:12} | RMSE: {rmse:5.2f} | MAE: {mae:5.2f} | R2: {r2:6.4f}")

df_loso = pd.DataFrame(loso_results)
p_loso_csv = os.path.join(gate_dir, "GEOGRAPHIC_GENERALIZATION_RESULTS.csv")
df_loso.to_csv(p_loso_csv, index=False)
print(f"Saved GEOGRAPHIC_GENERALIZATION_RESULTS.csv -> {p_loso_csv}")

# 3. Feature Importance Analysis
train_mask = df['year'] <= 2019
val_mask = (df['year'] >= 2020) & (df['year'] <= 2022)

X_tr = df.loc[train_mask, features_D]
y_tr = df.loc[train_mask, 'corn_yield_bu_acre']
X_v = df.loc[val_mask, features_D]
y_v = df.loc[val_mask, 'corn_yield_bu_acre']

final_xgb_D = xgb.XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    early_stopping_rounds=30,
    tree_method='hist'
)
final_xgb_D.fit(X_tr, y_tr, eval_set=[(X_v, y_v)], verbose=False)

# Extract feature importances
importance_types = ['gain', 'weight', 'cover']
imp_dict = {}
for it in importance_types:
    imp_dict[it] = final_xgb_D.get_booster().get_score(importance_type=it)

imp_rows = []
for f in features_D:
    gain_val = imp_dict['gain'].get(f, 0.0)
    weight_val = imp_dict['weight'].get(f, 0)
    cover_val = imp_dict['cover'].get(f, 0.0)
    
    if f.startswith('yield_'):
        cat = 'Historical Yield Lag'
    elif f.startswith('soil_'):
        cat = 'Soil Property'
    elif f.startswith('noaa_prcp'):
        cat = 'Precipitation'
    elif f.startswith('noaa_t') or f.startswith('noaa_heat'):
        cat = 'Temperature / Heat'
    else:
        cat = 'Other'
        
    imp_rows.append({
        'feature_name': f,
        'feature_category': cat,
        'importance_gain': round(gain_val, 2),
        'importance_weight': weight_val,
        'importance_cover': round(cover_val, 2)
    })

df_imp = pd.DataFrame(imp_rows).sort_values('importance_gain', ascending=False).reset_index(drop=True)
p_imp_csv = os.path.join(gate_dir, "FEATURE_IMPORTANCE_ANALYSIS.csv")
df_imp.to_csv(p_imp_csv, index=False)
print(f"\nSaved FEATURE_IMPORTANCE_ANALYSIS.csv -> {p_imp_csv}")

print("\nTop 10 Most Important Features by Gain:")
for idx, r in df_imp.head(10).iterrows():
    print(f"{idx+1:2d}. {r['feature_name']:28} ({r['feature_category']:20}) | Gain: {r['importance_gain']:8.2f}")

print("\nPhase 2 reliability analysis complete.")

