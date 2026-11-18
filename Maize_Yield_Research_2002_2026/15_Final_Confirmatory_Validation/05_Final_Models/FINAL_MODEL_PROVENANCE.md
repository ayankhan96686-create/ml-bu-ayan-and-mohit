# FINAL MODEL PROVENANCE AND RESEARCH INTEGRITY RECORD

## Complete Audit Trail of Data, Preprocessing, Hyperparameters, and Execution Artifacts
**Project:** County-Level Maize/Corn Grain Yield Prediction (2002–2026)  
**Output Directory:** `15_Final_Confirmatory_Validation/05_Final_Models/`  
**Master Random Seed:** 42  
**Date:** November 2026

---

### 1. Data Source and Lineage

- **Authoritative Dataset:** `14_Final_Data_Model_Optimization/08_Final_Model/Historical_Features_Corrected_v2.csv`
- **Total Historical Panel:** 16,390 county-year records across 788 unique eligible agricultural counties (2002–2025).
- **Temporal Partitions:**
  - Training Cohort: 2002–2022 ($N = 14,691$ records). Includes all 711 verified observations from the 2012 severe drought shock.
  - Prospective Out-of-Time Test Cohort: 2023–2025 ($N = 1,699$ records).
- **Prospective Horizon:** 2026 ($N = 814$ prospective county forecasts).

---

### 2. Feature Space Integrity

- **Predictor Count:** 69 strict in-season predictors.
- **Data Leakage Proof:**
  - Zero post-August or end-of-season covariates used for yield prediction.
  - Historical yield lags (`yield_lag_1yr`, `yield_lag_2yr`, `yield_roll_mean_5yr`) strictly computed on the complete historical calendar spine prior to any modeling.
  - Preprocessing imputations (medians) strictly computed on training data ($2002–2022$) and applied forward to test and prospective datasets without reverse leakage.

---

### 3. Model Reproducibility Standard

All model fitting procedures utilize deterministic random seeds (`random_state=42`, `random_seed=42`). Model evaluation metrics are computed using exact algebraic definitions ($R^2 = 1 - \text{SSE}/\text{SST}$) over identical sample subsets.
