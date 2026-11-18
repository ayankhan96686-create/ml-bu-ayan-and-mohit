# MODEL VERSIONING, ARCHITECTURE SPECIFICATION & PROVENANCE LEDGER

## Complete Software, Data, and Parameter Provenance
**Project Root:** `E:\ml do not open\Maize_Yield_Research_2002_2026`  
**Certified Version ID:** `v2.0_Optimized_Production_Ensemble`  
**Release Date:** November 2026

---

### 1. Core Environmental Specifications

```
Operating System: Windows Server / Windows 11 AMD64
Python:           3.14.0
scikit-learn:     1.9.0
xgboost:          3.4.1
lightgbm:         4.7.0
catboost:         1.2.10
shap:             0.54.1
pandas:           3.0.5
numpy:            2.5.1
Master Seed:      42
```

---

### 2. Model Lineage & Version Evolution

```
v1.0 (Phase 3 Gate): Provisional Linear / Baseline Tree (RMSE = 18.20 bu/ac)
       |
v1.5 (Phase 5/6 Certification): Certified 71-Feature Baseline (RMSE = 17.24 bu/ac)
       |
v1.8 (Phase 8 NextGen): Integrated 98-Feature Architecture (RMSE = 15.68 bu/ac)
       |
v2.0 (Phase 9 Final Optimized): Calibrated Quad-Tree Stacking (RMSE = 15.5129 bu/ac, R2 = 0.7221)
```

---

### 3. File Hash Register for v2.0 Deliverables

```
Historical_Features_Corrected_v2.csv: 16,390 rows x 99 columns
Future_2026_Features_Corrected_v2.csv: 814 rows x 98 columns
FINAL_FEATURE_MANIFEST_V2.csv:         69 verified in-season predictors
FINAL_MODEL_CONFIGURATION_V2.json:     Complete serialized hyperparameters
FINAL_MODEL_PREDICTIONS_V2.csv:        1,699 prospective test predictions with 90% conformal bands
```
