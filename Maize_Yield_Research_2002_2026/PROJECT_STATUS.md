# PROJECT STATUS REPORT: MAIZE YIELD PREDICTION FRAMEWORK (2002–2026)

**Project Identity:** County-Level Maize (Zea mays L.) Grain Yield Prediction Using Multi-Modal Environmental Stress, Satellite Remote Sensing, Soil Hydrology, and Machine Learning  
**Authors:** Ayan Khan & Mohit  
**Repository:** [https://github.com/ayankhan96686-create/ml-bu-ayan-and-mohit](https://github.com/ayankhan96686-create/ml-bu-ayan-and-mohit)  
**Active Branch:** `main`  
**Git HEAD Commit:** `da5061898f13287c2d6cc19a5757a74e92d4239c`  
**Last Updated:** 2026-11-22 02:00:14 UTC  

---

## 1. Executive Summary & Research State

This project has successfully completed **Phase 19: Master Implementation and Optimization**. All datasets, historical features (2002–2025), prospective scenarios (2026), validation frameworks, baseline suites, Optuna Bayesian hyperparameter optimizations, feature ablations, stacking meta-learners, data leakage audits, and dual-evaluation robustness benchmarks are finalized, certified with zero defects, and pushed to GitHub.

A publication-ready scientific PDF report has been compiled and is maintained in:
`19_Master_Implementation_80_20_Optimization/MASTER_MAIZE_YIELD_OPTIMIZATION_REPORT.pdf`

---

## 2. Study Area & Dataset Population

- **Geographic Domain:** 9 Primary US Corn Belt States (`IA`, `IL`, `IN`, `MN`, `MO`, `NE`, `OH`, `SD`, `WI`).
- **Spatial Coverage:** 814 counties (788 primary agricultural counties, 26 quarantined non-agricultural/urban fringe counties).
- **Historical Ground Truth:** 16,390 certified county-year records (2002–2025) with USDA NASS observed grain yield.
- **Target Variable:** `corn_yield_bu_acre` (Mean: $156.44\text{ bu/acre}$, Std: $36.72\text{ bu/acre}$, Range: $0.0\text{ to }248.6\text{ bu/acre}$).
- **Prospective 2026 Status:** 814 county records with valid environmental predictors present. Actual yield is 100% unobserved (harvest has not occurred); strictly quarantined from loss functions and test evaluations.
- **Predictor Dimensions:** 117 multi-modal features encompassing surface weather (NOAA/PRISM), soil properties (SoilGrids), satellite vegetation (MODIS NDVI/EVI), management (RMA), and historical yield memory lags.

---

## 3. Core Validation & Experimental Framework

### A. Primary Pooled Stratified 80:20 Holdout ($N = 16,390$)
- **Training Set (80.0%):** 13,112 observations (stratified by calendar year, seed = 42).
- **Testing Holdout (20.0%):** 3,278 observations (stratified by calendar year, seed = 42).
- **Partition Integrity:** Exactly zero row overlap between training and testing sets.

### B. In-Sample Chronological Validation Folds ($N = 13,112$)
Situated strictly inside the 80% training set across 4 expanding temporal windows:
- **Fold 1:** 2002–2013 Train ($N=6,561$) ──► 2014–2016 Val ($N=1,617$)
- **Fold 2:** 2002–2016 Train ($N=8,178$) ──► 2017–2019 Val ($N=1,471$)
- **Fold 3:** 2002–2019 Train ($N=9,649$) ──► 2020–2022 Val ($N=1,626$)
- **Fold 4:** 2002–2022 Train ($N=11,275$) ──► 2023–2025 Val ($N=1,359$)

---

## 4. Benchmark Performance & Leaderboard

### A. Independent Final Test Holdout Results ($N = 3,278$)

| Rank | Model Architecture | Test $R^2$ | RMSE (bu/ac) | MAE (bu/ac) | MAPE (%) | Mean Bias | Max Error | Drought Tail RMSE ($<120$ bu) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **Optimized XGBoost** | **0.8986** | **11.67** | **8.74** | **6.71%** | **+0.11** | 65.88 | **17.15** |
| **2** | **Optimized LightGBM** | 0.8944 | 11.91 | 8.90 | 6.85% | +0.13 | 77.57 | 17.58 |
| **3** | **Ridge Stacking Meta-Learner** | 0.8931 | 11.98 | 8.97 | 6.92% | +0.15 | 73.22 | 17.70 |
| **4** | **Calibrated Quad-Blend** | 0.8825 | 12.56 | 9.44 | 7.39% | +0.16 | 83.01 | 19.01 |
| **5** | **Optimized CatBoost** | 0.8700 | 13.22 | 9.96 | 7.73% | +0.20 | 74.77 | 19.58 |
| **6** | **Optimized Ridge** | 0.7998 | 16.40 | 12.36 | 9.99% | +0.18 | 113.80 | 26.43 |
| **7** | **Optimized ElasticNet** | 0.7968 | 16.52 | 12.42 | 10.08% | +0.19 | 110.53 | 26.91 |

### B. Dual Robustness: The Methodological Optimism Gap
- **Pooled Random Split:** XGBoost achieves $R^2 = 0.8986$, $\text{RMSE} = 11.67\text{ bu/ac}$.
- **Chronological Out-of-Time Split (2002–20 Train vs 2021–25 Test):** Regularized Linear & Stacking models achieve $R^2 = 0.7276\text{--}0.7610$, $\text{RMSE} = 15.37\text{--}16.41\text{ bu/ac}$.
- **Finding:** Contemporaneous regional weather allows spatial interpolation in random splits. True out-of-time forecasting requires handling unobserved annual weather regimes.

---

## 5. Scientific Integrity & Leakage Verification

All 8 automated checks in `DATA_LEAKAGE_AUDIT.csv` passed:
1. Train/Test ID separation: 0 overlap.
2. Validation fold isolation: 0 overlap.
3. Imputation containment: Train medians only.
4. Feature scaling: Fitted strictly on training subsets.
5. Lag directionality: Strict $t - k \ge 1$ enforced; zero contemporaneous target data in features.
6. Seasonal boundary: September 30 cut-off; zero post-harvest meteorological data.
7. Target integrity: No target feature derived from own-row target.
8. 2026 quarantine: Excluded from loss functions.

---

## 6. GitHub Synchronization & Deliverable Integrity

- **Repository:** Synchronized at commit `da5061898f13287c2d6cc19a5757a74e92d4239c`.
- **Git LFS:** 140 objects (578 MB) uploaded (`*.tif`, `*.shp`).
- **Files Tracked:** 1,607 files. Working tree clean.

---

## 7. Next Research Priorities

1. **In-Season 2026 Scenario Projections:** Simulate prospective 2026 county yields under ENSO neutral, El Niño, and La Niña climate trajectories.
2. **Sub-County Intra-Field Resolution:** Explore Sentinel-1 SAR and Sentinel-2 optical fusion to model intra-county soil moisture variations.
3. **Journal Submission:** Prepare final manuscript package for *Computers and Electronics in Agriculture* or *Environmental Research Letters*.
