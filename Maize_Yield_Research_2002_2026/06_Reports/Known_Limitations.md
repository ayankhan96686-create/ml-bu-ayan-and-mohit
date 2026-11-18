# Known Limitations & Methodological Constraints

**Project Title:** Maize (Corn) Grain Yield Prediction Using Machine Learning (2002–2025)  
**Study Region:** 9 US Corn Belt States (814 Counties)  
**Temporal Domain:** 2002–2026  
**Report Directory:** `E:/ml do not open/Maize_Yield_Research_2002_2026/06_Reports/`  

---

## 1. Overview of Known Limitations

Transparency is the foundation of reproducible science. This document explicitly outlines the known limitations, data gaps, spatial approximations, and operational constraints inherent in the current dataset and pipeline.

---

## 2. Specific Known Limitations

### 2.1 Unacquired Satellite Vegetation Indices (MODIS / Landsat / Sentinel)
- **Description:** Subfolders in `01_Raw_Data/` (`06_MODIS_NDVI_EVI`, `07_MODIS_LAI_FPAR`, `08_Landsat_Imagery`, `09_Sentinel2_Imagery`) contain documentation README files; no raw imagery or time series were acquired.
- **Impact:** The current feature set relies entirely on weather, soil, and management. It cannot directly measure actual crop canopy leaf area, biomass accumulation, or localized greenness anomalies.
- **Mitigation:** Query Google Earth Engine (GEE) to extract 16-day county MODIS MOD13A2 NDVI/EVI time series.

### 2.2 Incomplete USDA Cropland Data Layer (CDL)
- **Description:** The automated scraper acquired only 2020 Iowa CDL (124 MB); the remaining 8 states and 24 years are missing.
- **Impact:** Raster extractions (PRISM, SoilGrids) currently sample 3×3 pixel buffers around geometric county centroids rather than masking exclusively over pure corn-planted acreage pixels.
- **Mitigation:** In agricultural counties, geometric centroids accurately represent the dominant row-crop landscape, but in peri-urban or forested counties, centroid sampling introduces land-cover noise. Pure CDL masks should be acquired via GEE server-side reducers.

### 2.3 PRISM 2023 Precipitation Batch Download Gap
- **Description:** Year 2023 annual precipitation raster was inadvertently omitted in the initial PRISM download loop, creating 632 `NaN` values in `prism_ppt_mm`.
- **Impact:** Models relying specifically on `prism_ppt_mm` must impute or omit 2023.
- **Mitigation:** NOAA annual precipitation (`noaa_prcp_annual_in` and `noaa_prcp_may_aug_in`) has 100% complete coverage for 2023 and serves as a direct meteorological replacement. The missing PRISM file (`prism_ppt_us_25m_2023.tif`) should be downloaded from PRISM FTP.

### 2.4 State-Level Broadcast of Crop Progress & Condition
- **Description:** Weekly agricultural surveys are published by USDA NASS at state-level resolution, not county level.
- **Impact:** Assigns identical progress and condition values to all counties in a state, masking intra-state phenological variance (e.g. northern vs southern Illinois).
- **Mitigation:** Features provide macro-regional crop health signals. Models should treat them as regional covariates rather than micro-scale indicators.

### 2.5 Cook County, IL (`17031`) SoilGrids Shoreline Masking
- **Description:** Cook County's internal coordinate sits in downtown Chicago along Lake Michigan, falling into SoilGrids urban/water mask (`-32768`).
- **Impact:** 11 historical labeled observations for Cook County have `NaN` for soil features.
- **Mitigation:** Impute Cook County soil using adjacent collar agricultural counties (Will or Kane County), or exclude Cook County from row-crop training.

### 2.6 USDA NASS October 2013 Federal Government Shutdown Gap
- **Description:** Due to the October 1–16, 2013 federal government shutdown, USDA NASS operations were halted. Week 40 Crop Progress surveys were never conducted.
- **Impact:** `state_progress_mature_w40_pct` is `NaN` for all states in 2013 (683 rows).
- **Mitigation:** This is an authentic historical data gap, not an engineering bug. It should be handled via native tree-based missing value routing.

### 2.7 Temporal Autocorrelation & Data Leakage Prevention
- **Description:** Yield observations exhibit both secular technological growth (+1.95 bu/ac/yr) and spatial-temporal climate autocorrelation.
- **Impact:** Random cross-validation (K-Fold) severely inflates validation scores through data leakage.
- **Mitigation:** Researchers must enforce Time-Series Blocked Cross-Validation (Train: 2002–2019, Val: 2020–2022, Test: 2023–2025) and quarantine all post-harvest insurance outcome variables.

