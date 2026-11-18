# Data Cleaning Technical Report

**Project Title:** Maize (Corn) Grain Yield Prediction Using Machine Learning (2002–2025)  
**Study Region:** 9 US Corn Belt States (814 Counties)  
**Temporal Domain:** 2002–2025 Labeled Yield Observations; 2026 Unlabeled Prospective Set  
**Report Directory:** `E:/ml do not open/Maize_Yield_Research_2002_2026/06_Reports/`  

---

## 1. Executive Summary

This report documents the end-to-end data cleaning and transformation procedures applied across eight domain datasets. All cleaning steps were executed programmatically via reproducible Python scripts in `07_Code/`.

### Core Cleaning Achievements:
1. **Target Variable Validation:** Extracted 16,390 genuine county grain yield observations from 16,956 raw NASS records. 566 residual balance records (`"OTHER COUNTIES"`) were pruned. Zero synthetic values were fabricated.
2. **Standardized Spatial Keys:** Constructed standardized 5-digit FIPS codes (`county_fips`) across all tabular datasets, resolving cross-agency identifier inconsistencies.
3. **NOAA Fixed-Width CIRS Decoding:** Decoded NCEI monthly precipitation, mean temperature, max temperature, and min temperature files across 814 counties for 25 years (20,350 county-years).
4. **PRISM Zonal Statistics:** Extracted 3×3 pixel neighborhood zonal means around county centroids for 119 GeoTIFF rasters (4km resolution) using PIL and NumPy without GIS dependency bottlenecks.
5. **SoilGrids Multi-Depth Integration:** Acquired 18 regional GeoTIFFs via ISRIC OGC WCS and extracted Clay, Sand, Silt, SOC, pH, and Bulk Density across 3 depth intervals (0–5cm, 5–15cm, 15–30cm), verifying that texture fractions sum to 100%.
6. **Agricultural Management & Insurance:** Reparsed 342,447 raw USDA RMA policy records using the official 28-field pipe-delimited specification, correcting an earlier 21-column schema misalignment. Quarantined post-harvest leakage variables.
7. **Phenology & Crop Condition:** Aggregated weekly state surveys into seasonal health indices (% Good+Excellent) and milestone calendar timing (week 50% planted, emerged, silking, mature).

---

## 2. Dataset Cleaning Specifications

### 2.1 USDA NASS Corn Grain Yield (`clean_yield.py`)
- **Raw File:** `01_Raw_Data/01_USDA_NASS_Yield/USDA_NASS_Corn_Grain_Yield_9States_2002_2026.csv`
- **Cleaned File:** `03_Cleaned_Data/USDA_Yield_Cleaned.csv` (16,390 rows, 6 cols).
- **Transformations:**
  - Filtered exclusively for `Data Item == 'CORN, GRAIN - YIELD, MEASURED IN BU / ACRE'`.
  - Excluded 566 `"OTHER COUNTIES"` state residual balance rows.
  - Formatted 5-digit `county_fips = f"{int(st):02d}{int(ct):03d}"`.
  - Coerced suppressed measurement codes (`(D)`, `(Z)`) to `NaN`. Never converted suppressed codes to zero.
  - Preserved USDA statistical Coefficient of Variation (`cv_percent`) where published.

### 2.2 NOAA CIRS Climate (`clean_climate.py`)
- **Raw Files:** `01_Raw_Data/05_NOAA_Weather/climdiv-*.txt` (4 files, 156 MB).
- **Cleaned File:** `03_Cleaned_Data/NOAA_Climate_Cleaned.csv` (20,350 rows, 18 cols).
- **Transformations:**
  - Decoded fixed-width bytes: State (0–2), County (2–5), Element (5–7), Year (7–11), Monthly values (12 $\times$ 7 bytes).
  - Crosswalked legacy CIRS state codes (`11`=IL, `12`=IN, `13`=IA, `21`=MN, `23`=MO, `25`=NE, `33`=OH, `39`=SD, `47`=WI) to FIPS.
  - Converted hundredths of inches to inches (precip / 100.0) and tenths of °F to °F (temp / 10.0).
  - Replaced sentinels (`-9.99`, `-99.90`) with `NaN`.
  - Derived growing season cumulative rainfall (`noaa_prcp_may_aug_in`) and maximum summer temperature (`noaa_heat_stress_max_f`).

### 2.3 PRISM Gridded Climate (`process_prism.py`)
- **Raw Files:** 119 GeoTIFF rasters in `02_Extracted_Data/03_PRISM_Weather/`.
- **Cleaned File:** `03_Cleaned_Data/PRISM_Climate_Cleaned.csv` (20,350 rows, 9 cols).
- **Transformations:**
  - Extracted county internal points (`INTPTLAT`, `INTPTLON`) from US Census boundaries.
  - Applied affine transform to map coordinates to pixel indices.
  - Computed 3×3 pixel window zonal averages, rejecting NoData (`-9999.0`).
  - Documented missing 2023 precipitation raster (`NaN` for 632 rows).

### 2.4 ISRIC SoilGrids 2.0 (`process_soil.py`)
- **Raw Files:** 18 regional GeoTIFFs in `01_Raw_Data/11_Soil_SoilGrids/regional_rasters/`.
- **Cleaned File:** `03_Cleaned_Data/SoilGrids_Cleaned.csv` (814 rows, 23 cols).
- **Transformations:**
  - Extracted 6 soil properties across 3 standard depth intervals (0–5cm, 5–15cm, 15–30cm).
  - Calculated weighted 0–30cm profile averages ($\text{weight}_{0-5}=5/30$, $\text{weight}_{5-15}=10/30$, $\text{weight}_{15-30}=15/30$).
  - Verified texture fractions sum to 100% (Mean: 27.1% clay, 23.0% sand, 49.9% silt).

### 2.5 Agricultural Management, Phenology & Insurance (`clean_agriculture.py`)
- **Cleaned Files:**
  - `Crop_Condition_Cleaned.csv` (225 rows, 13 cols): Seasonal % Good+Excellent, % Poor+Very Poor.
  - `Crop_Progress_Cleaned.csv` (225 rows, 14 cols): Milestone percentages and week 50% reached.
  - `RMA_Cleaned.csv` (18,817 rows, 15 cols): Reparsed 28-field pipe delimited files; partitioned post-harvest variables.
  - `Fertilizer_Cleaned.csv` (178 rows, 11 cols): State-year benchmark chemical rates.

---

## 3. Transformation Lineage Summary Table

| Cleaned Output File | Raw Input File | Rows Before | Rows After | Dropped Rows & Reason |
|---|---|---|---|---|
| `USDA_Yield_Cleaned.csv` | `USDA_NASS_Corn_Grain_Yield_9States_*.csv` | 16,956 | 16,390 | 566 dropped ("OTHER COUNTIES" state residuals) |
| `NOAA_Climate_Cleaned.csv` | `climdiv-*.txt` (4 files) | 3,100,000 | 20,350 | Filtered to 9 study states and 2002–2026 |
| `PRISM_Climate_Cleaned.csv` | 119 GeoTIFF rasters | 119 rasters | 20,350 | 0 dropped (2023 missing ppt stored as NaN) |
| `SoilGrids_Cleaned.csv` | 18 regional GeoTIFF rasters | 18 rasters | 814 | 0 dropped (Cook Co centroid masked by lake) |
| `Crop_Condition_Cleaned.csv` | `USDA_NASS_Corn_Crop_Condition_*.csv` | 37,289 | 225 | Filtered to 9 states, 2002–2026; weekly to annual |
| `Crop_Progress_Cleaned.csv` | `USDA_NASS_Corn_Crop_Progress_*.csv` | 23,486 | 225 | Filtered to 9 states, 2002–2026; weekly to annual |
| `RMA_Cleaned.csv` | `sobcov_*/sobcov*.txt` (25 years) | 342,447 | 18,817 | Filtered to corn (code 0041); aggregated to county-year |
| `Fertilizer_Cleaned.csv` | `USDA_NASS_Corn_Fertilizer_Chemical_*.csv` | 31,862 | 178 | Filtered to corn fertilizer survey years and states |

