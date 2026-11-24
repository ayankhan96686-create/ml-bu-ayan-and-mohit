# REPOSITORY FILE INDEX & ARTIFACT CATALOG

**Project:** County-Level Maize Grain Yield Prediction Framework (2002–2026)  
**Authors:** Ayan Khan and Mohit  
**Certified HEAD Commit:** `b00a8367c2173be6974004e99d2a3f43ec833d15`  
**Repository:** [https://github.com/ayankhan96686-create/ml-bu-ayan-and-mohit](https://github.com/ayankhan96686-create/ml-bu-ayan-and-mohit)  

---

## 1. Top-Level Directory Architecture

```
E:\ml do not open\
├── MASTER_PROJECT_KNOWLEDGE/             # Complete Self-Contained Knowledge Package for ChatGPT
├── Maize_Yield_Research_2002_2026/       # Core Empirical Research Directory
│   ├── 00_Project_Admin/                 # Admin guidelines, data dictionary, team protocols
│   ├── 01_Raw_Data/                      # Original unedited data streams (NASS, PRISM, NOAA, CDL)
│   ├── 02_Extracted_Data/                # Extracted climate series, RMA insurance, TIGER boundaries
│   ├── 03_Cleaned_Data/                  # Standardized intermediate clean datasets
│   ├── 04_Integrated_Data/               # Integrated master feature tables (pre-Phase 12)
│   ├── 07_Code/                          # Core data validation and processing scripts
│   ├── 08_Project_Handover_Audit/        # Pre-modeling audit checklists and FIPS validation
│   ├── 09_Independent_Validation/        # Initial independent validation benchmarks
│   ├── 10_Final_Pre_Modeling_Gate/       # Pre-modeling quality gate benchmarks
│   ├── 11_Independent_Model_Validation/  # Phase 11 baseline models and interpretability
│   ├── 12_Final_Scientific_Certification/# Certification audits and red-team checks
│   ├── 13_Publication_Enhancement/       # Literature review matrix (32 papers), novel features
│   ├── 14_Final_Data_Model_Optimization/ # 2012 severe drought ablation and forecast reconciliation
│   ├── 15_Final_Confirmatory_Validation/ # Prediction bounding and residual sanity audits
│   ├── 16_Final_Publication_Package/     # Draft manuscript, publication figures, tables
│   ├── 17_Phase12_Advanced_Model_Optimization/ # 117-feature final models & spatial regression
│   ├── 18_Consistent_80_20_Train_Test_Split/  # Chronological out-of-time split benchmark
│   └── 19_Master_Implementation_80_20_Optimization/ # Authoritative Phase 19 pooled 80:20 models & PDF
├── PROJECT_STATUS.md                     # High-level executive dashboard
├── CHANGELOG.md                          # Comprehensive project changelog (Phases 00 to 19)
├── EXPERIMENT_TRACKER.csv                # Master ledger of all 52 experiments
├── RUN_MANIFEST.json                     # System manifest with SHA-256 data checksums
├── RESEARCH_DECISIONS.md                 # Formal Architectural Decision Records (ADRs 001–008)
└── generate_continuity_artifacts.py      # Master continuity automation script
```

---

## 2. Core Python Source Code Catalog

| Script Path | Primary Function & Role in Pipeline | Key Inputs | Key Outputs |
| :--- | :--- | :--- | :--- |
| `19_.../create_pooled_80_20_split.py` | Generates authoritative stratified 80:20 split | `Historical_Features_Phase12.csv` | `TRAIN_TEST_OBSERVATION_IDS_POOLED.csv` |
| `19_.../create_chronological_validation_folds.py` | Builds 4 expanding chronological validation folds | 80% Training partition | `CHRONOLOGICAL_VALIDATION_FOLDS.csv` |
| `19_.../train_baseline_models.py` | Trains and evaluates all 17 baseline models | Folds + Master Features | `BASELINE_MODEL_RESULTS.csv` |
| `19_.../optimize_hyperparameters.py` | Optuna Bayesian optimization (120 trials) | In-sample 4 folds | `OPTUNA_TRIAL_HISTORY.csv`, `BEST_HYPERPARAMETERS.json` |
| `19_.../feature_engineering_and_selection.py` | Evaluates feature configurations A–E | Feature manifest | `FEATURE_CONFIG_BENCHMARK.csv` |
| `19_.../audit_data_leakage.py` | Programmatically executes 8-point anti-leakage audit | All datasets & models | `DATA_LEAKAGE_AUDIT.csv` |
| `19_.../evaluate_independent_final_testing.py` | Evaluates champion models on untouched 20% holdout | 20% Test Holdout | `FINAL_INDEPENDENT_TEST_RESULTS.csv`, `STATE_WISE_PERFORMANCE.csv` |
| `19_.../chronological_robustness_experiment.py` | Benchmarks chronological vs pooled splits | Chronological split | `CHRONOLOGICAL_ROBUSTNESS_COMPARISON.csv` |
| `19_.../generate_master_publication_pdf.py` | ReportLab compilation of publication PDF deliverable | All Phase 19 CSVs | `MASTER_MAIZE_YIELD_OPTIMIZATION_REPORT.pdf` |
| `17_.../step02_advanced_feature_engineering.py` | Core calculation of all 117 multimodal features | Raw cleaned data streams | `Historical_Features_Phase12.csv`, `Future_2026_Features_Phase12.csv` |

---

## 3. Core Data Artifacts Catalog

| Dataset File Path | Description | Dimensions | File Size | SHA-256 Checksum |
| :--- | :--- | :---: | :---: | :--- |
| `17_.../13_Final_Model/Historical_Features_Phase12.csv` | **Authoritative Master Historical Dataset (2002–2025)** | 16,390 rows, 127 cols | 16.2 MB | `1ba70a5996...` |
| `17_.../13_Final_Model/Future_2026_Features_Phase12.csv` | **Prospective 2026 Dataset (Target strictly absent)** | 814 rows, 126 cols | 806 KB | Verified Quarantined |
| `17_.../13_Final_Model/FINAL_FEATURE_MANIFEST.csv` | Master catalog of 117 features and families | 117 rows, 4 cols | 10.0 KB | Complete Manifest |
| `19_.../TRAIN_TEST_OBSERVATION_IDS_POOLED.csv` | Explicit train/test split index for all 16,390 rows | 16,390 rows, 4 cols | 480 KB | Deterministic seed=42 |
| `19_.../TEST_PREDICTIONS_CONSOLIDATED.csv` | Model predictions and ground truth on test holdout | 3,278 rows, 11 cols | 420 KB | Verified Residuals |
| `19_.../MASTER_MAIZE_YIELD_OPTIMIZATION_REPORT.pdf` | Publication-ready scientific PDF report | 12 pages | 17.6 KB | `11340b82b9...` |

---

## 4. Large Files Catalog (>10 MB) & Git LFS Tracking

The following large geospatial and tabular files exceed 10 MB and are managed via Git Large File Storage (Git LFS, tracked in `.gitattributes`):

| File Size (MB) | File Path in Repository | Data Type / Description | LFS Status |
| :---: | :--- | :--- | :---: |
| **126.03 MB** | `Maize_Yield_Research_2002_2026/02_Extracted_Data/14_County_and_State_Boundaries/tl_2024_us_county/tl_2024_us_county.shp` | ESRI Shapefile: 2024 US County Boundaries | Tracked via Git LFS |
| **124.00 MB** | `Maize_Yield_Research_2002_2026/01_Raw_Data/02_USDA_CDL_Crop_Classification/CDL_2020_19_Iowa.tif` | Cropland Data Layer 30m Iowa GeoTIFF | Tracked via Git LFS |
| **80.03 MB** | `Maize_Yield_Research_2002_2026/01_Raw_Data/14_County_and_State_Boundaries/tl_2024_us_county.zip` | Compressed Census TIGER boundary archive | Tracked via Git LFS |
| **59.34 MB** | `Maize_Yield_Research_2002_2026/02_Extracted_Data/12_Planting_and_Crop_Management/USDA_RMA_Corn_Belt_Corn_Coverage_2002_2026.csv` | USDA RMA multi-year crop insurance experience | Standard Git Tracking |
| **39.03 MB** | `Maize_Yield_Research_2002_2026/01_Raw_Data/05_NOAA_Weather/climdiv-pcpncy-v1.0.0-20261006` | NOAA ClimDiv county monthly precipitation | Standard Git Tracking |
| **39.03 MB** | `Maize_Yield_Research_2002_2026/01_Raw_Data/05_NOAA_Weather/climdiv-tmaxcy-v1.0.0-20261006` | NOAA ClimDiv county monthly maximum temperature | Standard Git Tracking |
| **39.03 MB** | `Maize_Yield_Research_2002_2026/01_Raw_Data/05_NOAA_Weather/climdiv-tmincy-v1.0.0-20261006` | NOAA ClimDiv county monthly minimum temperature | Standard Git Tracking |
| **39.03 MB** | `Maize_Yield_Research_2002_2026/01_Raw_Data/05_NOAA_Weather/climdiv-tmpccy-v1.0.0-20261006` | NOAA ClimDiv county monthly mean temperature | Standard Git Tracking |
| **23–31 MB** | `Maize_Yield_Research_2002_2026/02_Extracted_Data/12_Planting_and_Crop_Management/sobcov_*/sobcov*.txt` | USDA RMA annual Summary of Business files | Standard Git Tracking |
| **16.20 MB** | `Maize_Yield_Research_2002_2026/17_Phase12_Advanced_Model_Optimization/13_Final_Model/Historical_Features_Phase12.csv` | Master Historical Feature Table (16,390 rows) | Standard Git Tracking |
