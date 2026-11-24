# MASTER PROJECT CONTEXT: MAIZE YIELD PREDICTION FRAMEWORK (2002–2026)

**Project Title:** Advanced County-Level Maize (*Zea mays* L.) Grain Yield Prediction Using Multi-Modal Environmental Stress, Satellite Remote Sensing, Soil Hydrology, and Machine Learning  
**Authors:** Ayan Khan and Mohit  
**GitHub Repository:** [https://github.com/ayankhan96686-create/ml-bu-ayan-and-mohit](https://github.com/ayankhan96686-create/ml-bu-ayan-and-mohit)  
**Primary Branch:** `main` | **Certified Commit SHA:** `b00a8367c2173be6974004e99d2a3f43ec833d15`  
**Study Horizon:** 2002–2026 inclusive (Historical Ground Truth: 2002–2025; Prospective Quarantine: 2026)  
**Geographic Coverage:** 9 US Corn Belt States (814 Counties: 788 Agricultural, 26 Non-Agricultural Quarantined)  

---

## 1. Research Objective & Problem Statement

### 1.1 The Biophysical and Agricultural Challenge
Maize (*Zea mays* L.) is the cornerstone of global agricultural commodities, animal feed, and biofuel production, with the United States producing over one-third of the global supply. Over 85% of US maize production is concentrated within the Midwestern "Corn Belt." Accurate, pre-harvest, county-level yield forecasting is essential for grain elevator logistics, national food security planning, supply-chain risk mitigation, and commodity market pricing.

However, operational yield prediction faces three profound scientific obstacles:
1. **The Extreme Climate Shock Dilemma:** Climate change has increased the frequency of acute compound heat and drought extremes (such as the devastating 2012 drought). While technological advances (hybrid genetics, variable-rate seeding, precision fertilization) create a steady upward secular trend (~1.8 to 2.2 bu/acre/year), extreme climate shocks cause non-linear physiological yield collapses (>30–50% drops) that linear models fail to capture.
2. **The Autoregressive Memory Trap:** In normal years, historical yield lags (e.g., 5-year rolling averages) serve as the most predictive features because they encapsulate localized soil natural capital and farmer management baselines. However, during acute shock years, heavy reliance on autoregressive memory acts as an upward statistical anchor, causing models to severely under-predict yield collapses (+22 to +23 bu/acre positive bias in severe drought).
3. **The Literature "Optimism Gap" (Evaluation Dichotomy):** Existing peer-reviewed literature is split into two conflicting camps:
   - Studies utilizing random k-fold or pooled cross-validation report optimistic coefficients of determination ($R^2 \approx 0.82\text{--}0.92$) by allowing contemporaneous regional weather from test years into the training split.
   - Studies utilizing strict out-of-time chronological forecasting report much lower performance ($R^2 \approx 0.70\text{--}0.76$) because models encounter unobserved macro-climate regimes.

### 1.2 Core Research Objectives
- **Objective 1:** Assemble an exhaustive, 24-year multimodal dataset (16,390 certified county-year observations across 2002–2025) integrating surface weather (PRISM/NOAA), soil pedotransfer profiles (SoilGrids), satellite vegetation dynamics (MODIS), crop progress/condition (USDA NASS), crop insurance risk ratings (USDA RMA), and groundwater irrigation infrastructure.
- **Objective 2:** Implement an unambiguous, reproducible **80:20 Train-Test Split** architecture and establish in-sample expanding chronological validation folds to eliminate temporal and spatial data leakage.
- **Objective 3:** Systematically benchmark 17 machine learning architectures—ranging from regularized linear models (Ridge, Lasso, ElasticNet) to advanced tree ensembles (LightGBM, XGBoost, CatBoost) and stacking meta-learners.
- **Objective 4:** Provide absolute scientific transparency by executing a dual-evaluation paradigm: proving why pooled random splits yield $R^2 = 0.8986$ while strict out-of-time temporal forecasting yields $R^2 = 0.7610$.
- **Objective 5:** Establish a quarantined, leak-free prospective framework for prospective 2026 county-level forecasting.

---

## 2. Research Scope & Geographical Coverage

### 2.1 The 9 Primary Corn Belt States
The study area encompasses the nine states forming the core of the US Corn Belt:

| State Code | State Name | Total Counties | Agricultural Eligible | Quarantined Non-Ag | Historical Observations (2002–2025) | Primary Agricultural Characteristics |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **IA** | Iowa | 99 | 99 | 0 | 2,360 | Rainfed, prime Mollisol soils, highest state production |
| **IL** | Illinois | 102 | 102 | 0 | 2,028 | Deep loess soils, high organic matter, intensive rainfed maize |
| **IN** | Indiana | 92 | 92 | 0 | 1,852 | Mixed glacial till soils, humid continental climate |
| **MN** | Minnesota | 87 | 81 | 6 | 1,719 | Northern Corn Belt; thermal accumulation (GDD) constrained |
| **MO** | Missouri | 115 | 114 | 1 | 1,739 | Southern fringe; claypan subsoils, high summer heat stress |
| **NE** | Nebraska | 93 | 93 | 0 | 2,023 | Western Corn Belt; ~55% irrigated via Ogallala Aquifer |
| **OH** | Ohio | 88 | 88 | 0 | 2,020 | Eastern humid boundary; heavier clay soils, low irrigation |
| **SD** | South Dakota | 66 | 47 | 19 | 1,250 | Northwestern semi-arid transition; extreme drought vulnerability |
| **WI** | Wisconsin | 72 | 72 | 0 | 1,419 | Northern dairy/grain rotation; shorter growing season |
| **TOTAL** | **9 States** | **814** | **788** | **26** | **16,390** | **Comprehensive US Corn Belt Core Population** |

### 2.2 County Eligibility & Quarantining
- **788 Agricultural Counties:** Possess continuous or near-continuous USDA NASS commercial corn grain yield reporting, active Cropland Data Layer (CDL) corn cultivation pixels, and crop insurance participation.
- **26 Quarantined Counties:** Located in non-agricultural urban centers (e.g., Hennepin/Ramsey in MN, St. Louis City in MO) or forested/arid non-corn counties in western South Dakota. Quarantined to prevent noise in agricultural loss functions.

---

## 3. Full Project History (Phase 00 to Phase 19)

The project represents a systematic, multi-stage engineering and scientific progression:

```
[Phase 00-07] Data Ingestion & Harmonization (USDA, NOAA, PRISM, Soils, MODIS)
      │
[Phase 08-11] Validation Gates & Baseline Benchmarking (Random Forests, Gradient Boosting)
      │
[Phase 12]    Independent Red-Team Statistical Audit (Verification of Ground Truth)
      │
[Phase 13]    Novelty & Feature Engineering (EDD, VPD, Non-Linear Stress, SHAP)
      │
[Phase 14]    Data Model Optimization & 2012 Severe Drought Ablation Experiment
      │
[Phase 15]    Final Confirmatory Validation (Sanity & Bounding Audits)
      │
[Phase 16]    Final Publication Package (Manuscript, Figures, Tables, Author Statements)
      │
[Phase 17]    Advanced Machine Learning Optimization (117 Features, Spatial Models)
      │
[Phase 18]    Consistent Chronological 80:20 Out-of-Time Split (2002-20 vs 2021-25)
      │
[Phase 19]    Master Implementation & Optimization (Pooled 80:20, Optuna, Stacking, PDF)
```

### Detailed Phase Narrative:
- **Phases 00–07 (Data Engineering & Spatial Ingestion):**
  - Downloaded raw USDA NASS QuickStats corn yield statistics (2002–2025).
  - Ingested PRISM 4km daily gridded weather grids and NOAA ClimDiv county series.
  - Processed SoilGrids 250m multi-depth physical and chemical attributes.
  - Extracted USDA RMA crop insurance experience files (`sobcov` 2002–2026).
  - Built 5-digit zero-padded Federal Information Processing Standards (FIPS) county spine.
- **Phases 08–11 (Pre-Modeling Quality Gates & Initial Modeling):**
  - Implemented initial train-test splits and audited prediction bounds.
  - Ran first-generation Random Forest and Gradient Boosting regressors.
- **Phase 12 (Independent Red-Team Statistical Audit):**
  - Conducted adversarial audit of Phase 11 artifacts to verify zero synthetic data fabrication.
  - Certified 16,390 historical county-year ground truth yields against official USDA NASS databases.
- **Phase 13 (Novelty Enhancement & Advanced Biophysical Features):**
  - Engineered non-linear agro-stress metrics: Extreme Degree Days (EDD > 30 deg C), afternoon Vapor Pressure Deficit (VPD), and silking-temperature interactions.
  - Integrated TreeSHAP interpretability pipeline.
- **Phase 14 (Data Model Optimization & 2012 Drought Ablation):**
  - Tested hypothesis of excluding the 2012 severe drought year to improve model fit.
  - **Critical Finding:** Excluding 2012 artificially boosted training R2 but increased severe drought tail test error by >40%. Mandated permanent retention of 2012 in all training populations.
- **Phase 15 (Final Confirmatory Validation):**
  - Executed residual normality checks, prediction bounding box verifications (0 to 300 bu/ac), and spatial contiguity diagnostics.
- **Phase 16 (Final Publication Package Compilation):**
  - Authored full journal manuscript draft (`FINAL_RESEARCH_MANUSCRIPT.md`), high-resolution figures, publication tables, and journal submission cover letters.
- **Phase 17 (Phase 12 Advanced ML & Hierarchical Spatial Modeling):**
  - Engineered master 117-feature multimodal table (`Historical_Features_Phase12.csv`).
  - Tested spatial error autoregressive models and state-level random intercepts.
- **Phase 18 (Consistent Chronological 80:20 Out-of-Time Split):**
  - Implemented strict chronological block split: 2002–2020 Train (19 years, 12,976 obs) vs 2021–2025 Test (5 years, 3,414 obs).
  - Uncovered the resilience of regularized linear models (ElasticNet R2 = 0.7610, RMSE = 15.37 bu/ac) on pure temporal extrapolation.
- **Phase 19 (Master Implementation, Optimization & Certification):**
  - Formulated single authoritative Pooled Stratified 80:20 split (13,112 train, 3,278 test) across all 24 historical years.
  - Created 4-fold expanding chronological validation framework inside the 80% training set.
  - Executed 120 Optuna Bayesian hyperparameter optimization trials.
  - Conducted 5-configuration feature ablations (Configs A–E) proving Config D (Top 45) matches full 117-feature performance.
  - Refit champion models: **Optimized XGBoost** achieved **Test R2 = 0.8986, RMSE = 11.67 bu/ac, MAE = 8.74 bu/ac, MAPE = 6.71%**.
  - Built Ridge Stacking Meta-Learner (Test R2 = 0.8931, RMSE = 11.98 bu/ac).
  - Programmatically audited data leakage across 8 checks (100% passed).
  - Compiled publication PDF deliverable via ReportLab: `MASTER_MAIZE_YIELD_OPTIMIZATION_REPORT.pdf`.

---

## 4. Completed Work vs. Incomplete / In-Progress Work

### 4.1 Completed Work (Fully Certified & Verified)
- Comprehensive ingestion and spatial merging of 16,390 historical county-year records (2002–2025).
- Engineering, documentation, and validation of all 117 multimodal predictors.
- Single unified 80:20 pooled stratified split and 4 chronological validation folds.
- 120-trial Optuna Bayesian hyperparameter tuning.
- Benchmark evaluations across 17 distinct machine learning algorithms.
- Full independent holdout evaluation with severe drought tail stress analysis (<120 bu/ac).
- Dual robustness comparison quantifying the "Optimism Gap" between random and chronological splits.
- Automated 8-point data leakage audit with 100% pass rate.
- Generation of publication PDF report (`MASTER_MAIZE_YIELD_OPTIMIZATION_REPORT.pdf`).
- Synchronization of master research continuity deliverables to GitHub repository at commit `b00a8367c2173be6974004e99d2a3f43ec833d15`.

### 4.2 Incomplete / In-Progress Work (Scientific Boundaries)
- **Actual 2026 Harvest Verification:** The 814 county observations for year 2026 contain valid pre-season environmental predictors, but actual harvest yields are unobserved (harvest has not occurred). These records are quarantined until official USDA NASS release in late 2026.
- **Prospective Climate Scenario Trajectories:** Simulating prospective 2026 yields under distinct ENSO phases (El Niño, La Niña, ENSO Neutral).
- **Sub-County Intra-Field Modeling:** Ingesting Sentinel-1 SAR and Sentinel-2 optical 10m imagery to model sub-county soil moisture variability.
- **Journal Peer-Review Submission:** Formal submission to *Computers and Electronics in Agriculture* or *Environmental Research Letters*.

---

## 5. Current Project Status

The project is currently in **Production-Certified & Publication-Ready Status**:
- **Codebase:** Fully functional, documented, and tested under Python 3.14.0 on Windows Server / 11 AMD64.
- **Repository:** Clean working tree tracking `origin/main` at commit `b00a8367c2173be6974004e99d2a3f43ec833d15`.
- **Reproducibility:** Machine-verifiable via `RUN_MANIFEST.json` with SHA-256 data checksums and deterministic random seeds (seed=42).
