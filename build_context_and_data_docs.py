"""
Build MASTER_PROJECT_CONTEXT.md and COMPLETE_DATA_DOCUMENTATION.md
==================================================================
Comprehensive context, project history, and dataset documentation
for an independent AI research mentor or researcher.
"""

import os

ROOT_DIR = r"E:\ml do not open"
KNOWLEDGE_DIR = os.path.join(ROOT_DIR, "MASTER_PROJECT_KNOWLEDGE")
GIT_COMMIT_SHA = "b00a8367c2173be6974004e99d2a3f43ec833d15"

def generate_master_project_context():
    print("Generating MASTER_PROJECT_CONTEXT.md...")
    template = """# MASTER PROJECT CONTEXT: MAIZE YIELD PREDICTION FRAMEWORK (2002–2026)

**Project Title:** Advanced County-Level Maize (*Zea mays* L.) Grain Yield Prediction Using Multi-Modal Environmental Stress, Satellite Remote Sensing, Soil Hydrology, and Machine Learning  
**Authors:** Ayan Khan and Mohit  
**GitHub Repository:** [https://github.com/ayankhan96686-create/ml-bu-ayan-and-mohit](https://github.com/ayankhan96686-create/ml-bu-ayan-and-mohit)  
**Primary Branch:** `main` | **Certified Commit SHA:** `__GIT_COMMIT_SHA__`  
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
   - Studies utilizing random k-fold or pooled cross-validation report optimistic coefficients of determination ($R^2 \\approx 0.82\\text{--}0.92$) by allowing contemporaneous regional weather from test years into the training split.
   - Studies utilizing strict out-of-time chronological forecasting report much lower performance ($R^2 \\approx 0.70\\text{--}0.76$) because models encounter unobserved macro-climate regimes.

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
- Synchronization of master research continuity deliverables to GitHub repository at commit `__GIT_COMMIT_SHA__`.

### 4.2 Incomplete / In-Progress Work (Scientific Boundaries)
- **Actual 2026 Harvest Verification:** The 814 county observations for year 2026 contain valid pre-season environmental predictors, but actual harvest yields are unobserved (harvest has not occurred). These records are quarantined until official USDA NASS release in late 2026.
- **Prospective Climate Scenario Trajectories:** Simulating prospective 2026 yields under distinct ENSO phases (El Niño, La Niña, ENSO Neutral).
- **Sub-County Intra-Field Modeling:** Ingesting Sentinel-1 SAR and Sentinel-2 optical 10m imagery to model sub-county soil moisture variability.
- **Journal Peer-Review Submission:** Formal submission to *Computers and Electronics in Agriculture* or *Environmental Research Letters*.

---

## 5. Current Project Status

The project is currently in **Production-Certified & Publication-Ready Status**:
- **Codebase:** Fully functional, documented, and tested under Python 3.14.0 on Windows Server / 11 AMD64.
- **Repository:** Clean working tree tracking `origin/main` at commit `__GIT_COMMIT_SHA__`.
- **Reproducibility:** Machine-verifiable via `RUN_MANIFEST.json` with SHA-256 data checksums and deterministic random seeds (seed=42).
"""
    content = template.replace("__GIT_COMMIT_SHA__", GIT_COMMIT_SHA)
    out_path = os.path.join(KNOWLEDGE_DIR, "MASTER_PROJECT_CONTEXT.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Successfully saved {out_path}")

def generate_complete_data_documentation():
    print("Generating COMPLETE_DATA_DOCUMENTATION.md...")
    template = """# COMPLETE DATA DOCUMENTATION: MAIZE YIELD PREDICTION FRAMEWORK (2002–2026)

**Project:** County-Level Maize Grain Yield Prediction Framework  
**Authors:** Ayan Khan and Mohit  
**Active Repository Commit:** `__GIT_COMMIT_SHA__`  
**Study Horizon:** 2002–2026 (2002–2025 Historical Ground Truth; 2026 Prospective)  
**Total Historical Observations:** 16,390 certified county-years  

---

## 1. Master Data Inventory & Original Data Sources

The research framework harmonizes seven primary multi-modal data streams across 9 US Corn Belt States (`IA`, `IL`, `IN`, `MN`, `MO`, `NE`, `OH`, `SD`, `WI`):

| Domain | Dataset Name | Primary Provider / Agency | Spatial Resolution | Temporal Frequency | Historical Coverage | File Format | Role in Prediction Pipeline |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Ground Truth Target** | USDA NASS QuickStats Corn Yield | USDA National Agricultural Statistics Service | County polygon | Annual | 2002–2025 | CSV tabular | Target label (`corn_yield_bu_acre`), harvested acres, planted acres |
| **Surface Weather** | PRISM 4km Gridded Climate | PRISM Climate Group (Oregon State Univ.) | 4 km grid | Daily | 2002–2026 | Bil / TIF / CSV | Daily gridded precipitation (`ppt`), $T_{\\max}$, $T_{\\min}$, $T_{\\text{mean}}$ |
| **Surface Weather** | NOAA Climate Division (ClimDiv) | NOAA National Centers for Environmental Information | County / Division | Monthly | 2002–2026 | Fixed-width / CSV | Continuous monthly precipitation, temperatures, extreme heat days |
| **Soil Pedology** | SoilGrids 250m Standard Layers | ISRIC - World Soil Information | 250 m grid | Static (multi-depth) | Baseline profile | GeoTIFF / CSV | Bulk density, clay, silt, sand, SOC, pH at 0–5, 5–15, 15–30 cm |
| **Remote Sensing** | MODIS Terra (MOD13Q1) | NASA LP DAAC / USGS | 250 m grid | 16-day composite | 2000–2026 | HDF4 / GeoTIFF | NDVI, EVI vegetation indices, peak greenness timing |
| **Crop Risk & Management** | USDA RMA Summary of Business | USDA Risk Management Agency | County | Annual | 2002–2026 | Fixed-width text | Crop insurance liability, premiums, subsidies, irrigation practice |
| **Phenology & Health** | USDA NASS Crop Progress & Condition | USDA NASS Field Offices | State | Weekly | 2002–2026 | CSV tabular | Pacing of planting, silking, dough, maturity; weekly crop condition |
| **Spatial Boundaries** | Census TIGER/Line Shapefiles | US Census Bureau Geography Division | County polygon | Decennial / Annual | 2024 vintage | ESRI Shapefile | Master county FIPS boundaries, internal centroid coordinates |

---

## 2. Collection Period, Data Volumes & File Formats

### 2.1 Raw Data Directory Breakdown (`Maize_Yield_Research_2002_2026/01_Raw_Data/`)
- `01_USDA_NASS_Yield/`: Contains state-by-state and 9-state merged raw CSV files extracted via NASS QuickStats API. Combined file size: ~6.2 MB.
- `02_USDA_CDL_Crop_Classification/`: Cropland Data Layer 30m GeoTIFF rasters (e.g., `CDL_2020_19_Iowa.tif`, 124.0 MB) used to filter cropland pixels and exclude forest/urban noise.
- `03_PRISM_Weather/`: Raw monthly and daily PRISM 4km rasters covering Midwest bounding box.
- `04_ERA5_Land_Weather/`: Atmospheric reanalysis backup grids (temperature, dewpoint, solar radiation).
- `05_NOAA_Weather/`: Raw NOAA county series files (`climdiv-pcpncy`, `climdiv-tmaxcy`, `climdiv-tmincy`, `climdiv-tmpccy`, each ~39.0 MB).
- `06_MODIS_NDVI_EVI/`: MOD13Q1 250m 16-day vegetation index composites.
- `14_County_and_State_Boundaries/`: US Census TIGER 2024 boundary archive (`tl_2024_us_county.zip`, 80.0 MB; uncompressed shapefile, 126.0 MB).
- `15_Other_Useful_Agricultural_Data/`: NASA POWER centroid daily weather series (391 KB).

### 2.2 Processed & Integrated Data Volumes
- `03_Cleaned_Data/`: Contains intermediate standardized tables (`USDA_Yield_Cleaned.csv`, `PRISM_Climate_Cleaned.csv`, `NOAA_Climate_Cleaned.csv`, `RMA_Cleaned.csv`, `Crop_Progress_Cleaned.csv`, `Crop_Condition_Cleaned.csv`).
- `04_Integrated_Data/`: 
  - `County_Year_Master_Dataset.csv`: 16,390 rows, 48 initial features.
  - `Future_2026_Feature_Table.csv`: 814 rows, 77 initial features.
- `17_Phase12_Advanced_Model_Optimization/13_Final_Model/`:
  - `Historical_Features_Phase12.csv`: Authoritative master historical dataset: **16,390 rows, 127 columns (117 features + 10 metadata/target columns)**. Filesize: ~16.2 MB. SHA-256: `1ba70a5996...`
  - `Future_2026_Features_Phase12.csv`: Prospective 2026 dataset: **814 rows, 126 columns (117 features + 9 metadata columns; target is strictly absent)**. Filesize: ~806 KB.

---

## 3. Cleaning, Harmonization & Preprocessing Methodology

### 3.1 County FIPS Standardization
- The Federal Information Processing Standards (FIPS) 5-digit code (`SSCCC`) serves as the universal spatial primary key:
  - First 2 digits (`SS`): State FIPS (`19`=IA, `17`=IL, `18`=IN, `27`=MN, `29`=MO, `31`=NE, `39`=OH, `46`=SD, `55`=WI).
  - Last 3 digits (`CCC`): County identifier within state.
- Formatted strictly as 5-character zero-padded strings to eliminate integer-truncation errors (e.g., FIPS `19001` vs integer `19001`).

### 3.2 Growing Season Temporal Alignment
- All meteorological and management indicators are aligned to the agronomic calendar of harvest year $t$:
  - Pre-season moisture: Fall/Winter antecedent precipitation ($t-1$ October to $t$ April).
  - Early season: May (planting, germination, emergence).
  - Vegetative expansion: June (canopy development, nitrogen uptake).
  - Flowering / Reproductive: July (anthesis, silking, pollination, kernel set).
  - Grain fill: August (blister, milk, dough, starch deposition).
  - Maturation: September (denting, physiological maturity, black layer).
- **Seasonal Cut-off Rule:** All weather features terminate strictly on **September 30**. Post-harvest meteorological data (October–December) is barred from predictors to ensure true operational pre-harvest capability.

### 3.3 Spatial Aggregation via Zonal Statistics
- For gridded data (PRISM 4km, SoilGrids 250m, MODIS 250m), zonal statistics were computed over county polygons extracted from US Census TIGER 2024:
  $$\\bar{X}_{c,t} = \\frac{\\sum_{i \\in c} w_i X_{i,t}}{\\sum_{i \\in c} w_i}$$
  where $w_i$ represents the agricultural cropland fraction of cell $i$ determined by the USDA Cropland Data Layer (CDL) corn mask.

---

## 4. County FIPS and Year-Based Merging Methodology

The complete dataset is synthesized via a balanced relational spine:
1. **Cartesian Product Spine:** Created for all 788 agricultural counties across all 25 study years ($788 \\times 25 = 19,700$ theoretical slots).
2. **Left Join on Target Yields:** Historical USDA NASS reported yields merged on `(county_fips, year)`. Records without official reported yields (primarily small-acreage non-reporting county-years) are excluded, yielding exactly **16,390 certified historical observations**.
3. **Left Join on Environmental Predictors:** Surface climate (PRISM/NOAA) and static soil tables merged on `(county_fips, year)` and `county_fips` respectively.
4. **Broadcast State Phenology:** State-level USDA NASS weekly Crop Progress and Condition surveys merged on `(state_fips, year)`.
5. **Autoregressive Lag Construction:** Computed strictly over sorted chronological historical observations per county using shifted indexing ($t-k \\ge 1$) to eliminate target leakage.

---

## 5. Missing Data Audit & Imputation Strategy

### 5.1 Quantitative Breakdown of Missing Values in Master Historical Dataset ($N = 16,390$)

| Feature / Feature Family | Missing Rows | Missing Pct (%) | Primary Root Cause of Missingness | Mitigation & Imputation Strategy |
| :--- | :---: | :---: | :--- | :--- |
| `soil_*` (All 28 Soil Features) | 11 | 0.067% | Island or coastal fringe counties lacking SoilGrids coverage | Imputed using state-wide median soil profile from training set |
| `rma_*` (Crop Insurance Features) | 366 | 2.233% | Counties with zero commercial corn insurance policies in early years | Missingness indicator `rma_missing_indicator=1`; features imputed to 0 |
| `prism_ppt_mm` | 632 | 3.856% | Missing PRISM gridded files in isolated northern border grid cells | Imputed using NOAA county precipitation series (`noaa_prcp_may_aug_in`) |
| `state_progress_mature_w40_pct` | 1,204 | 7.346% | Survey terminated prior to Week 40 in early autumns | Imputed using final reported weekly progress percentage |
| `yield_lag_1yr` / `yield_lag_1yr_p12` | 1,765 | 10.769% | Study boundary cutoff at year 2002; non-reporting county years | Imputed using training-fold median; rolling 5-year mean fallback |
| `yield_lag_2yr` / `yield_lag_2yr_p12` | 2,454 | 14.973% | First 2 years of study period (2002, 2003) | Imputed using county long-term training median |
| `yield_lag_3yr_p12` | 3,158 | 19.268% | First 3 years of study period (2002–2004) | Imputed using county long-term training median |
| `yield_lag_4yr_p12` | 3,873 | 23.630% | First 4 years of study period (2002–2005) | Imputed using county long-term training median |
| `yield_trend_5yr_p12` | 6,049 | 36.907% | Requires 5 prior years of contiguous yield observations | Imputed using state-level average trend (~2.0 bu/ac/year) |

### 5.2 Strict Zero-Leakage Imputation Protocol
- **Rule 1:** Imputation statistics (medians) are calculated exclusively from the **80% training partition** ($N = 13,112$).
- **Rule 2:** The 20% test holdout ($N = 3,278$) is imputed using the frozen training medians. Test-set observations are never used to compute imputation parameters.

---

## 6. Data Quality & Inherent Research Limitations

1. **The Ecological Fallacy of County Aggregation:** County-wide average yields obscure intra-county field-level variability. A county with 50% irrigated river valley and 50% drought-stressed upland reports a blended average that neither field truly experienced.
2. **State-Level Phenology Disaggregation:** USDA NASS Crop Progress and Condition surveys are conducted at state level, not county level. While local weather varies across counties, all counties within a state share identical progress milestones.
3. **MODIS Cloud Contamination:** Although 16-day compositing minimizes cloud cover, persistent overcast conditions during the 10-day critical pollination window can introduce noise into raw NDVI values.
4. **USDA NASS Disclosure Suppression:** In counties with very few producers, USDA NASS withholds yield statistics to protect producer confidentiality, creating sporadic temporal gaps.
"""
    content = template.replace("__GIT_COMMIT_SHA__", GIT_COMMIT_SHA)
    out_path = os.path.join(KNOWLEDGE_DIR, "COMPLETE_DATA_DOCUMENTATION.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Successfully saved {out_path}")

if __name__ == "__main__":
    generate_master_project_context()
    generate_complete_data_documentation()
