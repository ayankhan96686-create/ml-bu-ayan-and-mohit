# CHANGELOG: MAIZE YIELD PREDICTION RESEARCH FRAMEWORK (2002–2026)

All notable changes, milestone audits, model versions, and architectural decisions for this project are documented in this file.
The project adheres to scientific reproducibility standards and Semantic Versioning.

---

## [v1.0.0] - Phase 19: Master Implementation, Optimization & Certification (2026-11-15)

### Added
- **Single Primary Pooled 80:20 Split:** Implemented centralized stratified splitting across all 24 historical years ($N=16,390$; Train: 13,112 rows, Test: 3,278 rows, seed=42) in `create_pooled_80_20_split.py`.
- **Chronological Validation Framework:** Engineered 4 expanding temporal folds inside the 80% training set in `create_chronological_validation_folds.py`.
- **Optuna Bayesian Optimization (120 Trials):** Fine-tuned LightGBM, XGBoost, CatBoost, HistGB, ElasticNet, Ridge, and LinearSVR in `optimize_hyperparameters.py`.
- **Multi-Domain Feature Ablation:** Formulated Feature Configurations A (Climate & Soil), B (+Lags), C (+Agro-Stress), D (Top 45 Selected), and E (Master 117 Feats) in `feature_engineering_and_selection.py`.
- **Automated 8-Point Data Leakage Audit:** Programmatic verification of partition independence, imputation, scaling, lag directionality, and 2026 quarantine in `audit_data_leakage.py`.
- **Ensemble Optimization:** Built and evaluated Calibrated Quad-Tree Blend and Ridge Stacking Meta-Learner ($lpha=5.0$) on out-of-fold predictions.
- **Independent 20% Holdout Evaluation:** Refit champion models on full 80% train set and evaluated once on 20% test set, including state-wise, year-wise, and $<120$ bu/acre drought tail stress audits in `evaluate_independent_final_testing.py`.
- **Dual Robustness Evaluation:** Addressed the methodological "Optimism Gap" comparing pooled random 80:20 ($R^2 = 0.8986$) against pure chronological out-of-time holdouts ($R^2 = 0.7610$) in `chronological_robustness_experiment.py`.
- **ReportLab Publication PDF Engine:** Created `generate_master_publication_pdf.py` compiling publication-grade deliverable `MASTER_MAIZE_YIELD_OPTIMIZATION_REPORT.pdf` (17.6 KB).
- **Master Continuity Documentation:** Created `PROJECT_STATUS.md`, `CHANGELOG.md`, `EXPERIMENT_TRACKER.csv`, `RUN_MANIFEST.json`, and `RESEARCH_DECISIONS.md`.

### Results Certified
- **Champion Architecture:** Optimized XGBoost ($R^2 = 0.8986$, $\text{RMSE} = 11.67\text{ bu/ac}$, $\text{MAE} = 8.74\text{ bu/ac}$, $\text{MAPE} = 6.71\%$, Severe Drought Tail RMSE: $17.15\text{ bu/ac}$).
- **Stacking Meta-Learner:** $R^2 = 0.8931$, $\text{RMSE} = 11.98\text{ bu/ac}$.
- **All 8 Leakage Checks Passed.**

---

## [v0.9.0] - Phase 18: Consistent Chronological 80:20 Out-of-Time Split (2026-11-15)

### Added
- Formulated strict chronological block split: 2002–2020 Training (19 years, 79.17%) vs. 2021–2025 Testing (5 years, 20.83%).
- Generated `CONSOLIDATED_MODEL_EVALUATION_80_20.csv` benchmarking all 17 models on pure temporal extrapolation.
- Proved regularized linear models (ElasticNet $R^2 = 0.7610$) retain highest stability when predicting unobserved future years.

---

## [v0.8.0] - Phase 17: Phase 12 Advanced ML & Spatial Modeling (2026-11-13)

### Added
- Hierarchical spatial regression and state-level random intercept models.
- Integrated multi-seed benchmark suites and 117-feature final manifest `FINAL_FEATURE_MANIFEST.csv`.

---

## [v0.7.0] - Phase 16: Final Publication Package & Manuscript Formulation (2026-11-12)

### Added
- Drafted academic research manuscript sections: Abstract, Introduction, Materials & Methods, Results, Discussion.
- Formulated LaTeX high-resolution figures for spatial error distributions and feature importance plots.

---

## [v0.6.0] - Phase 15: Final Confirmatory Validation (2026-11-12)

### Added
- Automated sanity checks across prediction ranges, bounding boxes, and residual normality.

---

## [v0.5.0] - Phase 14: Data Model Optimization & 2012 Drought Ablation (2026-11-12)

### Changed
- Executed 2012 severe drought exclusion experiment.
- **Critical Finding:** Verified that deleting 2012 artificially inflates metrics but destroys model capacity to generalize to future climate shocks. Mandated permanent retention of 2012.

---

## [v0.4.0] - Phase 13: Novelty Enhancement & Advanced Features (2026-11-11)

### Added
- Engineered non-linear agro-stress features: Extreme Degree Days (`edd_july_aug`), Vapor Pressure Deficit (`vpd_aug_mean`), and rolling multi-year trend deviations.

---

## [v0.3.0] - Phase 12: Independent Red-Team Statistical Audit (2026-11-11)

### Verified
- Controlled ablation statistical audit of Phase 11 artifacts. Confirmed zero synthetic data fabrication.

---

## [v0.2.0] - Phases 08 to 11: Validation Gates & Early Benchmarking (2026-11-10)

### Added
- Pre-modeling quality gates and initial gradient boosting vs linear regression benchmarks.

---

## [v0.1.0] - Phases 00 to 07: Ingestion, Harmonization & Baseline Pipeline (2026-11-09)

### Added
- Downloaded and processed USDA NASS historical corn grain yield (2002–2025).
- Ingested PRISM 4km gridded weather and NOAA ClimDiv county series.
- Ingested SoilGrids physical and chemical hydrological soil properties.
- Processed MODIS Terra 16-day NDVI/EVI satellite vegetation dynamics.
- Established FIPS county geographical spine across the 9 primary Corn Belt states.
