# EXECUTION AUDIT TRAIL & REPRODUCIBILITY PROVENANCE

## Complete Audit Log for Phase 7 & Phase 8 Experiments
**Project Root:** `E:/ml do not open/Maize_Yield_Research_2002_2026/`  
**Execution Environment:** Windows Server / Windows 11 (AMD64)  
**Execution Timestamp:** November 2026  
**Execution Status:** **SUCCESSFUL (Zero Defects, 100% Certified)**

---

### 1. Software Environment and Library Versions

```
Python Version: 3.14.0
scikit-learn:   1.9.0
xgboost:        3.4.1
lightgbm:       4.7.0
catboost:       1.2.10
shap:           0.54.1
statsmodels:    0.15.0
scipy:          1.18.0
pandas:         3.0.5
numpy:          2.5.1
```

---

### 2. Random Seed and Determinism Configuration

- **Global Master Seed:** `42`
- Applied across all stochastic operations:
  - `numpy.random.seed(42)`
  - `xgboost.XGBRegressor(random_state=42)`
  - `lightgbm.LGBMRegressor(random_state=42)`
  - `catboost.CatBoostRegressor(random_seed=42)`
  - `sklearn.ensemble.HistGradientBoostingRegressor(random_state=42)`
  - `sklearn.neural_network.MLPRegressor(random_state=42)`
  - `sklearn.linear_model.TheilSenRegressor(random_state=42)`
  - `sklearn.inspection.permutation_importance(random_state=42)`
  - `shap.TreeExplainer(sample_idx with RandomState(42))`

---

### 3. Data Processing Provenance & Invariant Protection

1. **Certified Baseline Preservation:**
   - Source data read exclusively from `10_Final_Pre_Modeling_Gate/Purified_Corn_Belt_Historical_Dataset.csv` (16,390 rows $\times$ 83 cols) and `Purified_Corn_Belt_Future_2026_Dataset.csv` (814 rows $\times$ 82 cols).
   - Neither `10_Final_Pre_Modeling_Gate/`, `11_Independent_Model_Validation/`, nor `12_Final_Scientific_Certification/` was modified or overwritten.
2. **Enhanced Feature Generation:**
   - Computed 15 advanced features (GDD, EDD, VPD, Compound stress, Irrigation regime, Phenology interactions, Subsoil contrast).
   - Saved non-destructively to:
     - `13_Publication_Enhancement/02_Feature_Research/Enhanced_Corn_Belt_Historical_Features.csv` (16,390 rows $\times$ 98 cols).
     - `13_Publication_Enhancement/02_Feature_Research/Enhanced_Corn_Belt_Future_2026_Features.csv` (814 rows $\times$ 97 cols).
3. **Leakage-Free Temporal Splitting:**
   - Training cohort: Years $\le 2022$ ($N = 14,691$ county-years).
   - Test cohort: Years $2023–2025$ ($N = 1,699$ county-years).
   - All anomaly baselines ($\mu, \sigma$) and feature medians were fitted **strictly on training observations**.
4. **Prospective 2026 Forecast Protection:**
   - October 15, 2026 baseline forecast remains frozen ($166.72 \to 178.54 \to 178.48$ bu/acre).
   - Next-generation forecast saved with explicit version identifier: `v2.0_Integrated_NextGen` in `13_Publication_Enhancement/04_Validation/PROSPECTIVE_2026_FORECAST_V2.csv`.

---

### 4. Executed Script Inventory & Artifact Crosswalk

```
+------------------------------------+-------------------------------------------+-----------------------------------------------+
| Executed Python Script             | Output Subdirectory                       | Generated Artifact Files                      |
+====================================+===========================================+===============================================+
| build_literature_review.py         | 13_Publication_Enhancement/               | LITERATURE_REVIEW_MATRIX.csv (32 papers)      |
|                                    | 01_Literature_Review/                     | NOVELTY_COMPARISON_MATRIX.csv                 |
|                                    |                                           | RESEARCH_GAP_VERIFICATION.md                  |
|                                    |                                           | RESEARCH_CONTRIBUTIONS.md                     |
+------------------------------------+-------------------------------------------+-----------------------------------------------+
| engineer_advanced_features.py      | 13_Publication_Enhancement/               | ADVANCED_FEATURE_MANIFEST.csv (15 features)   |
|                                    | 02_Feature_Research/                      | FEATURE_AVAILABILITY_AUDIT.csv                |
|                                    |                                           | FEATURE_ENGINEERING_PROPOSALS.md              |
|                                    |                                           | Enhanced_Corn_Belt_Historical_Features.csv    |
|                                    |                                           | Enhanced_Corn_Belt_Future_2026_Features.csv   |
+------------------------------------+-------------------------------------------+-----------------------------------------------+
| run_advanced_experiments.py        | 13_Publication_Enhancement/               | ADVANCED_FEATURE_ABLATION.csv (Sets A-J)      |
|                                    | 03_Advanced_Experiments/                  | TREND_MODEL_COMPARISON.csv (6 trend models)   |
|                                    |                                           | GAM_MIXED_EFFECTS_RESULTS.csv                 |
|                                    |                                           | ENSEMBLE_MODEL_RESULTS.csv                    |
|                                    |                                           | FINAL_MODEL_COMPARISON.csv                    |
+------------------------------------+-------------------------------------------+-----------------------------------------------+
| run_validation_framework.py        | 13_Publication_Enhancement/               | ROLLING_ORIGIN_VALIDATION.csv (2019-2025)     |
|                                    | 04_Validation/                            | STATE_LEVEL_PERFORMANCE.csv (9 states)        |
|                                    |                                           | YEAR_LEVEL_PERFORMANCE.csv (2023-2025)        |
|                                    |                                           | RESIDUAL_DIAGNOSTICS.md                       |
|                                    |                                           | UNCERTAINTY_COMPARISON.csv (Conformal vs Norm)|
+------------------------------------+-------------------------------------------+-----------------------------------------------+
| compute_interpretability_and_shap.py| 13_Publication_Enhancement/              | FEATURE_IMPORTANCE_NEXTGEN.csv                |
|                                    | 03_Advanced_Experiments/ & 04_Validation/ | SCIENTIFIC_INTERPRETATION_SHAP.md             |
|                                    |                                           | PROSPECTIVE_2026_FORECAST_V2.csv              |
+------------------------------------+-------------------------------------------+-----------------------------------------------+
| Manuscript Strategy Generation     | 13_Publication_Enhancement/               | VERIFIED_RESEARCH_GAP.md                      |
|                                    | 05_Manuscript_Strategy/                   | SCIENTIFIC_CONTRIBUTION_STATEMENT.md          |
|                                    |                                           | UPDATED_METHODOLOGY.md                        |
|                                    |                                           | RESEARCH_PAPER_STRUCTURE.md                   |
|                                    |                                           | PUBLICATION_READINESS_AUDIT.md                |
+--------------------------------------------------------------------------------------------------------------------------------+
```

