# EXECUTION AUDIT TRAIL AND REPRODUCIBILITY LOG

**Project:** County-Level Maize Grain Yield Prediction Research Project (2002–2026)  
**Audit Phase:** Phase 5 — Independent Model Validation, Advanced ML Benchmarking & Forecast Certification  
**Host Platform:** Windows 11 (AMD64)  
**Execution Timestamp:** November 11, 2026  

---

## 1. Software Environment and Dependency Inventory

| Package | Installed Version | Verification Status | Role in Pipeline |
| :--- | :--- | :---: | :--- |
| **Python** | 3.14.0 | Certified | Core execution runtime |
| **xgboost** | 3.4.1 | Certified | Certified baseline and controlled ablation |
| **lightgbm** | 4.7.0 | Certified | High-speed gradient boosting benchmarking |
| **catboost** | 1.2.10 | Certified | Gradient boosted decision trees on categorical/numerical features |
| **scikit-learn**| 1.9.0 | Certified | Ridge, ElasticNet, HistGB, RF, ExtraTrees, SVR, MLP, metrics |
| **shap** | 0.54.1 | Certified | Exact TreeSHAP explanations and feature attribution |
| **statsmodels** | 0.15.0 | Certified | LOWESS non-linear curve fitting for biophysical dependence |
| **numpy** | 2.5.1 | Certified | Array manipulations and numerical operations |
| **pandas** | 3.0.5 | Certified | Tabular data manipulation and aggregation |
| **scipy** | 1.18.0 | Certified | Statistical tests (paired t-test, Wilcoxon signed-rank) |
| **matplotlib** | 3.11.1 | Certified | Figure generation and plotting |
| **seaborn** | 0.13.2 | Certified | Statistical visualization and dependence plots |

---

## 2. Global Invariants and Determinism Controls

- **Master Random Seed:** `42` applied across all pseudo-random generators (`random_state=42`, `np.random.seed(42)`, `random_seed=42`).
- **Data Protection:** Source directory `10_Final_Pre_Modeling_Gate/` was accessed strictly in read-only mode. All Phase 5 scripts wrote exclusively into `11_Independent_Model_Validation/`.
- **Temporal Split Boundaries:**
  - **Training Set:** $\mathcal{T}_{\text{train}} = \{t \mid t \le 2019\}$ ($12,660$ observations, $784$ unique counties).
  - **Validation Set (Calibration):** $\mathcal{T}_{\text{val}} = \{t \mid 2020 \le t \le 2022\}$ ($2,031$ observations, $763$ unique counties).
  - **Test Set (Holdout):** $\mathcal{T}_{\text{test}} = \{t \mid 2023 \le t \le 2025\}$ ($1,699$ observations, $707$ unique counties).
  - **Prospective Forecast:** $\mathcal{T}_{\text{fut}} = \{t \mid t = 2026\}$ ($814$ counties: $788$ certified eligible, $26$ quarantined non-agricultural).

---

## 3. Script Execution Provenance

| Execution Step | Script Path | Output Artifacts Generated | Status |
| :--- | :--- | :--- | :---: |
| **Phases A, B, C** | `09_Code/execute_phase_a_b_c.py` | `01_Project_Audit/PHASE5_PROJECT_INVENTORY.csv`<br>`01_Project_Audit/DATASET_INTEGRITY_AUDIT.csv`<br>`01_Project_Audit/COMPLETE_FEATURE_ENGINEERING_MANIFEST.csv`<br>`01_Project_Audit/PHASE5_EXISTING_PIPELINE_AUDIT.md` | PASS |
| **Phase C (Doc)** | (Direct Document) | `01_Project_Audit/PREPROCESSING_AND_ENCODING_AUDIT.md` | PASS |
| **Phase D** | `09_Code/run_phase_d_reproducibility.py` | `02_Reproducibility/REPRODUCED_BASELINE_RESULTS.csv`<br>`02_Reproducibility/EXPERIMENT_CONFIGURATION.json`<br>`02_Reproducibility/REPRODUCTION_DISCREPANCIES.md` | PASS |
| **Phase E** | `09_Code/run_phase_e_ablation.py` | `03_Ablation_Experiments/CONTROLLED_ABLATION_BENCHMARK.csv`<br>`03_Ablation_Experiments/LAG_ONLY_XGBOOST_RESULTS.csv`<br>`03_Ablation_Experiments/ENVIRONMENTAL_INCREMENTAL_VALUE_REPORT.md` | PASS |
| **Phases F, G** | `09_Code/run_phase_f_g_benchmarks.py` | `04_Advanced_ML_Benchmarks/ADVANCED_ML_MODEL_COMPARISON.csv`<br>`04_Advanced_ML_Benchmarks/BEST_MODEL_HYPERPARAMETERS.json`<br>`04_Advanced_ML_Benchmarks/HYPERPARAMETER_SEARCH_RESULTS.csv`<br>`04_Advanced_ML_Benchmarks/MODEL_PREDICTIONS_TEST.csv`<br>`04_Advanced_ML_Benchmarks/PREPROCESSING_PARAMS_SET_D.json` | PASS |
| **Phase H** | `09_Code/run_phase_h_geographic_validation.py` | `05_Geographic_Validation/LOSO_RECONCILIATION.csv`<br>`05_Geographic_Validation/CORRECTED_LOSO_RESULTS.csv`<br>`05_Geographic_Validation/GEOGRAPHIC_GENERALIZATION_REPORT.md` | PASS |
| **Phase I** | `09_Code/run_phase_i_interpretability.py` | `06_Interpretability/FEATURE_FAMILY_ABLATION.csv`<br>`06_Interpretability/FEATURE_IMPORTANCE_RANKINGS.csv`<br>`06_Interpretability/SHAP_GLOBAL_IMPORTANCE.png`<br>`06_Interpretability/SHAP_DEPENDENCE_ANALYSIS.png`<br>`06_Interpretability/MODEL_INTERPRETABILITY_REPORT.md` | PASS |
| **Phases J, K** | `09_Code/run_phase_j_k_forecast_audit.py` | `07_Forecast_Audit/COUNTY_PREDICTION_VALIDATION.csv`<br>`07_Forecast_Audit/PREDICTION_INTERVAL_VALIDATION.csv`<br>`07_Forecast_Audit/STATE_AGGREGATION_COMPARISON.csv`<br>`07_Forecast_Audit/2026_FORECAST_AUDIT.md` | PASS |
| **Phase L** | (Direct Documents) | `08_Final_Research_Report/PHASE5_INDEPENDENT_AUDIT_REPORT.md`<br>`08_Final_Research_Report/FINAL_MODEL_SELECTION_REPORT.md`<br>`08_Final_Research_Report/UPDATED_RESEARCH_PAPER_RECOMMENDATIONS.md` | PASS |

