# Technical Register of Known Errors, Bugs, and Data Warnings

**Project:** Maize/Corn Grain Yield Prediction Using Machine Learning (2002–2025)  
**Root Directory:** `E:/ml do not open/Maize_Yield_Research_2002_2026/`  
**Audit Date:** 2026-11-10  
**Auditor:** Senior Data Engineer & Technical Quality Assurance Lead  

---

## 1. Overview of the Technical Register

This register documents every technical error, syntax exception, schema misalignment, data truncation, and operational warning identified during the lifecycle of this project. Each entry details the root cause, immediate impact, resolution status, and ongoing implications for independent researchers.

---

## 2. Chronological Bug & Error Register

### Error #1: RMA 21-Column Header Misalignment (Schema Shift Bug)
- **Component:** `02_Extracted_Data/12_Planting_and_Crop_Management/USDA_RMA_Corn_Belt_Corn_Coverage_2002_2026.csv`
- **Discovery Date:** 2026-11-10
- **Severity:** **CRITICAL** (Corrupted raw extracted table)
- **Root Cause:** Raw USDA RMA SOBCOV files contain 28 pipe-delimited fields (`Year|State|Abbr|County|Name|...`). The initial extraction script applied a 21-column header list that omitted fields 15–17 (`Units_Earning_Prem`, `Units_Indemnified`, `Quantity_Type`). This shifted downstream numeric fields by 3 positions, causing `Total_Premium` to contain the text `'Acres'`.
- **Resolution:** Fully resolved in `07_Code/clean_agriculture.py`. The cleaning script re-parsed all 25 raw SOBCOV text files directly using the official 28-field layout, restoring authentic dollar values for liability, premium, subsidy, and indemnity payments.
- **Current Status:** **FIXED.** `03_Cleaned_Data/RMA_Cleaned.csv` is 100% verified.

---

### Error #2: PRISM 2023 Precipitation Batch Download Omission
- **Component:** `02_Extracted_Data/03_PRISM_Weather/ppt/`
- **Discovery Date:** 2026-11-10
- **Severity:** **HIGH** (Missing 632 values in feature table)
- **Root Cause:** In `00_Project_Admin/tools/download_prism_data.py`, the download loop for precipitation inadvertently omitted year 2023 or the FTP connection timed out without raising an unhandled exception.
- **Immediate Impact:** PRISM annual precipitation has 632 missing values in 2023 across all study counties.
- **Workaround:** NOAA annual precipitation (`noaa_prcp_annual_in` and `noaa_prcp_may_aug_in`) has complete 100% coverage for 2023 and serves as a direct meteorological substitute.
- **Recommended Remediation:** Download `prism_ppt_us_25m_2023.tif` from PRISM FTP and re-execute `07_Code/process_prism.py`.
- **Current Status:** **OPEN / DOCUMENTED DATA GAP.**

---

### Error #3: Cook County, IL (`17031`) SoilGrids Centroid Water Masking
- **Component:** `03_Cleaned_Data/SoilGrids_Cleaned.csv`
- **Discovery Date:** 2026-11-10
- **Severity:** **LOW** (Affects 1 urban county out of 814)
- **Root Cause:** The US Census internal coordinate (`INTPTLAT`, `INTPTLON`) for Cook County, IL sits in downtown Chicago adjacent to Lake Michigan. In the ISRIC SoilGrids 250m global rasters, this pixel is classified as urban impervious / open water, returning the NoData value (`-32768`).
- **Immediate Impact:** 11 historical labeled observations for Cook County have `NaN` for soil features.
- **Recommended Remediation:** Impute Cook County soil texture using adjacent rural collar counties (e.g. Will or Kane County, IL), or drop Cook County from agricultural ML training.
- **Current Status:** **OPEN / DOCUMENTED ANOMALY.**

---

### Error #4: USDA CDL Large-Raster Download Timeout
- **Component:** `01_Raw_Data/02_USDA_CDL_Crop_Classification/`
- **Discovery Date:** 2026-11-09
- **Severity:** **HIGH** (CDL missing for 8 of 9 states)
- **Root Cause:** `download_cdl_data.py` attempted to stream massive uncompressed state-level GeoTIFF rasters from CropScape. The connection terminated after downloading Iowa 2020 (124 MB), leaving all other states and years unacquired.
- **Immediate Impact:** CDL pure corn crop masks could not be generated for county climate/soil zonal extraction.
- **Recommended Remediation:** Do not download raw statewide GeoTIFFs. Use Google Earth Engine (GEE) to compute county-level corn acreage masks via server-side zonal reducers.
- **Current Status:** **OPEN / RESEARCH GAP.**

---

### Error #5: Satellite Remote Sensing Folders Left as Documentation Stubs
- **Component:** `01_Raw_Data/06_MODIS_NDVI_EVI/`, `07_MODIS_LAI_FPAR/`, `08_Landsat_Imagery/`, `09_Sentinel2_Imagery/`
- **Discovery Date:** 2026-11-09
- **Severity:** **CRITICAL RESEARCH GAP**
- **Root Cause:** NASA Earthdata and USGS EarthExplorer require OAuth authentication and specialized chunked downloading. The previous agent created informational README files (`README_MODIS_NDVI_EVI_Access.md`, etc.) rather than completing the raster acquisitions.
- **Immediate Impact:** Satellite vegetation canopy metrics (NDVI, EVI, LAI, FPAR) are absent from the dataset.
- **Recommended Remediation:** Acquire county-level 16-day MODIS MOD13A2 or MOD09A1 mean NDVI/EVI time series via GEE.
- **Current Status:** **OPEN / CRITICAL RESEARCH GAP.**

---

### Error #6: USDA NASS October 2013 Federal Government Shutdown Reporting Gap
- **Component:** `03_Cleaned_Data/Crop_Progress_Cleaned.csv`
- **Discovery Date:** 2026-11-10
- **Severity:** **LOW** (Authentic historical gap)
- **Root Cause:** During the US Federal Government shutdown of October 1–16, 2013, USDA NASS operations were suspended. Weekly Crop Progress surveys for Week 40 were never conducted or published.
- **Immediate Impact:** `state_progress_mature_w40_pct` is `NaN` for all states in 2013 (683 rows).
- **Current Status:** **AUTHENTIC HISTORICAL GAP (NOT AN AI ERROR).** Must be preserved as authentic missingness.

---

### Error #7: Seaborn v0.14 Deprecation Warnings
- **Component:** `07_Code/perform_eda.py`
- **Discovery Date:** 2026-11-10
- **Severity:** **TRIVIAL** (Console warning)
- **Root Cause:** Seaborn v0.14 deprecates passing `palette` without assigning `hue`.
- **Resolution:** Updated plotting calls in `perform_eda.py` to assign `hue='state'` or `hue='feature'` with `legend=False`.
- **Current Status:** **RESOLVED.**

