# Data Cleaning & Transformation Technical Audit

**Project:** Maize/Corn Grain Yield Prediction Using Machine Learning (2002–2025)  
**Root Directory:** `E:/ml do not open/Maize_Yield_Research_2002_2026/`  
**Audit Date:** 2026-11-10  
**Auditor:** Senior Data Engineer & Agricultural Data Scientist  

---

## 1. Scope & Standards of the Cleaning Audit

This audit examines the exact procedures, Python code implementations, mathematical transformations, filtering operations, and unit conversions applied across all eight cleaned domain datasets located in `03_Cleaned_Data/`. 

### Core Scientific Auditing Rules Enforced:
1. **Preservation of Authentic Variance:** No interpolation, synthetic smoothing, or mean-imputation was performed on raw target observations or features.
2. **Standardized Spatial/Temporal Keys:** Every table is keyed on standardized 5-digit Federal Information Processing Standards (FIPS) county codes (`county_fips`) and/or 2-digit state FIPS codes (`state_fips`), paired with 4-digit Gregorian calendar years (`year`).
3. **Information Loss Prevention:** We explicitly verify whether any transformation destroyed scientifically valid variance or genuine agricultural anomalies (e.g., severe drought impacts).

---

## 2. Domain-by-Domain Cleaning Audit

### 2.1 USDA NASS Corn Grain Yield (`USDA_Yield_Cleaned.csv`)
- **Input File:** `01_Raw_Data/01_USDA_NASS_Yield/USDA_NASS_Corn_Grain_Yield_9States_2002_2026.csv` (16,956 raw records).
- **Execution Script:** `07_Code/clean_yield.py`
- **Output File:** `03_Cleaned_Data/USDA_Yield_Cleaned.csv` (16,390 rows, 6 columns).
- **Transformation Pipeline:**
  1. *Commodity Filtering:* Strict selection where `Data Item == 'CORN, GRAIN - YIELD, MEASURED IN BU / ACRE'`. Silage, sweet corn, popcorn, and area harvested were excluded.
  2. *Geographic Filtering & Residual Pruning:* 566 records corresponding to `"OTHER COUNTIES"` or `"OTHER (COMBINED) COUNTIES"` (USDA residual balance aggregations for withheld or low-acreage counties) were purged.
  3. *Key Construction:* Standardized 5-digit `county_fips` constructed via `f"{int(row['State ANSI']):02d}{int(row['County ANSI']):03d}"`.
  4. *Target Numeric Coercion:* Formatted text numbers (e.g., `'185.0'`, `'2,100'`) parsed via `.str.replace(',', '').astype(float)`. Non-numeric suppressed codes (e.g., `(D)` for withheld, `(Z)` for negligible) converted to `NaN`.
  5. *CV% Data Quality Variable:* Coefficient of Variation preserved as `cv_percent` (available for 3,730 Census-year records).
- **Outlier & Anomaly Verification:**
  - Range: `0.0` to `253.6` bu/acre (Mean: `157.45` bu/acre, Median: `161.40` bu/acre).
  - The 9 true zero observations (`0.0` bu/acre) in western South Dakota (Harding, Lawrence, Todd counties) represent authentic total crop abandonment/drought loss reported by NASS. They were intentionally preserved.
- **Representative Data Sample:**

```csv
county_fips,state,county,year,corn_yield_bu_acre,cv_percent
17001,Illinois,Adams,2002,143.0,
17001,Illinois,Adams,2003,161.0,
17001,Illinois,Adams,2007,175.0,2.1
17001,Illinois,Adams,2012,102.0,5.8
```

---

### 2.2 NOAA Climate Divisional Database (`NOAA_Climate_Cleaned.csv`)
- **Input Files:** `01_Raw_Data/05_NOAA_Weather/climdiv-pcpncy-v1.0.0-*.txt`, `climdiv-tmpccy-v1.0.0-*.txt`, `climdiv-tmaxcy-v1.0.0-*.txt`, `climdiv-tmincy-v1.0.0-*.txt` (3,100,000 national records each).
- **Execution Script:** `07_Code/clean_climate.py`
- **Output File:** `03_Cleaned_Data/NOAA_Climate_Cleaned.csv` (20,350 rows, 18 columns).
- **Transformation Pipeline:**
  1. *Fixed-Width Parsing:* Parsed NCEI fixed-width record layout: Bytes 0–2 (State Code), Bytes 2–5 (County Code), Bytes 5–7 (Element Code: 01=PCPN, 02=TMPC, 27=TMAX, 28=TMIN), Bytes 7–11 (Year), followed by twelve 7-character monthly values.
  2. *Legacy CIRS State Code Crosswalk:* NOAA CIRS uses historical alphabetical numbering (`11`=IL, `12`=IN, `13`=IA, `21`=MN, `23`=MO, `25`=NE, `33`=OH, `39`=SD, `47`=WI). These were mapped to official 2-digit US FIPS (`17`, `18`, `19`, `27`, `29`, `31`, `39`, `46`, `55`).
  3. *Unit Conversions:*
     - Precipitation: Raw values in hundredths of inches divided by `100.0` -> `Inches`.
     - Temperatures: Raw values in tenths of degrees Fahrenheit divided by `10.0` -> `°F`.
  4. *Missing Sentinel Handling:* NOAA sentinels (`-9.99`, `-99.90`) converted to `NaN`.
  5. *Agronomic Feature Engineering:*
     - `noaa_prcp_may_aug_in`: Cumulative May–August growing season rainfall.
     - `noaa_prcp_may_sep_in`: Full May–September rainfall.
     - `noaa_tmean_may_aug_f`: Mean temperature across vegetative & reproductive phases.
     - `noaa_heat_stress_max_f`: Maximum monthly temperature observed between June and August.
- **Representative Data Sample:**

```csv
county_fips,state,county,year,noaa_prcp_may_aug_in,noaa_tmean_may_aug_f,noaa_tmax_jul_f,noaa_heat_stress_max_f
17001,Illinois,Adams County,2012,8.94,76.4,94.2,94.2
17001,Illinois,Adams County,2014,19.22,71.8,83.5,83.5
```

---

### 2.3 PRISM Climate Group Rasters (`PRISM_Climate_Cleaned.csv`)
- **Input Files:** 119 GeoTIFF rasters in `02_Extracted_Data/03_PRISM_Weather/` (32-bit floating point, 4km resolution).
- **Execution Script:** `07_Code/process_prism.py`
- **Output File:** `03_Cleaned_Data/PRISM_Climate_Cleaned.csv` (20,350 rows, 9 columns).
- **Transformation Pipeline:**
  1. *Spatial Centroid Mapping:* Extracted official county internal coordinates (`INTPTLAT`, `INTPTLON`) from Census TIGER/Line DBF.
  2. *Affine Coordinate Transform:* Mapped geographic WGS84 coordinates into raster pixel space:
     $$\text{pixel\_x} = \frac{\text{lon} - x_{\min}}{\text{pixel\_width}}, \quad \text{pixel\_y} = \frac{y_{\max} - \text{lat}}{\text{pixel\_height}}$$
  3. *Zonal Window Aggregation:* Calculated 3×3 pixel neighborhood zonal means around each county centroid, rejecting PRISM NoData pixels (`-9999.0`).
  4. *Valid Coverage Metric:* Computed `prism_valid_coverage_pct` to track edge-of-domain validity.
- **Critical Data Gap Identified:**
  - Year 2023 precipitation GeoTIFF was omitted in the initial batch download. The pipeline records `NaN` for 2023 PRISM precipitation across all 814 counties, while NOAA precipitation provides complete backup coverage.

---

### 2.4 ISRIC SoilGrids 2.0 (`SoilGrids_Cleaned.csv`)
- **Input Files:** 18 regional GeoTIFF rasters downloaded from ISRIC OGC WCS at `maps.isric.org` covering the Corn Belt bounding box (`lat: 35.0 to 49.5`, `lon: -104.5 to -80.0`).
- **Execution Script:** `07_Code/process_soil.py`
- **Output File:** `03_Cleaned_Data/SoilGrids_Cleaned.csv` (814 rows, 23 columns).
- **Transformation Pipeline:**
  1. *Properties Extracted:* Clay (`clay`), Sand (`sand`), Silt (`silt`), Soil Organic Carbon (`soc`), Soil pH (`phh2o`), Bulk Density (`bdod`).
  2. *Depth Intervals:* 0–5 cm, 5–15 cm, 15–30 cm.
  3. *Profile Mean Integration:* Calculated weighted 0–30 cm profile average:
     $$\bar{X}_{0-30\text{cm}} = \frac{5 \cdot X_{0-5} + 10 \cdot X_{5-15} + 15 \cdot X_{15-30}}{30}$$
  4. *Texture Consistency Check:* Verified that mineral texture fractions sum to 100%:
     $$\text{Mean Clay: } 27.1\%, \quad \text{Mean Sand: } 23.0\%, \quad \text{Mean Silt: } 49.9\% \quad (\Sigma = 100.0\%)$$
- **Masking Finding:**
  - Cook County, IL (`17031`) centroid returned `NaN` due to urban impervious mask / Lake Michigan water boundary. All other 813 counties have complete 100% soil profiles.

---

### 2.5 USDA NASS Crop Condition & Progress (`Crop_Condition_Cleaned.csv`, `Crop_Progress_Cleaned.csv`)
- **Execution Script:** `07_Code/clean_agriculture.py`
- **Output Files:**
  - `Crop_Condition_Cleaned.csv` (225 rows, 13 columns: 9 states × 25 years).
  - `Crop_Progress_Cleaned.csv` (225 rows, 14 columns: 9 states × 25 years).
- **Transformations:**
  - Aggregated weekly state reports (Weeks 20 to 42) into seasonal indices: `% Good + Excellent` and `% Poor + Very Poor`.
  - Computed critical reproductive window condition (Weeks 28 to 34: silking through grain-fill).
  - Calculated phenological timing metrics: calendar week index where cumulative progress reaches 50% for planting, silking, and maturity.
  - Documented federal reporting gap: Week 40 maturity was not published in 2013 due to the October 2013 federal government shutdown.

---

### 2.6 USDA RMA Crop Insurance (`RMA_Cleaned.csv`)
- **Input Files:** 25 raw SOBCOV text files (`sobcov02.txt` through `sobcov26.txt`) comprising 342,447 Corn policy records.
- **Execution Script:** `07_Code/clean_agriculture.py`
- **Output File:** `03_Cleaned_Data/RMA_Cleaned.csv` (18,817 rows, 15 columns).
- **Critical Schema Repair Executed:**
  - Previous extraction scripts used a 21-column header, causing column shifts that misaligned numeric premium/liability fields.
  - `clean_agriculture.py` re-engineered the parser to read the complete 28-field pipe-delimited format directly from the original archives, restoring true dollar values for `Net_Planted_Acres`, `Liability`, `Total_Premium`, `Subsidy`, and `Indemnity_Amount`.
- **Target Leakage Quarantine:**
  - Pre-harvest management features (`rma_net_planted_acres`, `rma_liability_total`, `rma_liability_per_planted_acre`) are partitioned from post-harvest outcome variables (`rma_postharvest_indemnity_amount`, `rma_postharvest_loss_ratio`).

---

## 3. Information Loss Assessment

| Domain Dataset | Transformation Operation | Scientific Information Loss Risk | Audit Verdict |
|---|---|---|---|
| **Yield** | Dropped 566 `"OTHER COUNTIES"` records | Zero loss for county modeling; state residuals do not correspond to geographic polygons. | **BENEFICIAL / CLEAN** |
| **NOAA Weather** | Monthly sum and seasonal aggregation | Loss of sub-monthly extreme daily weather fluctuations (e.g. 3-day heatwaves). | **ACCEPTABLE (Documented)** |
| **PRISM** | Centroid 3×3 pixel buffer vs full polygon mask | Minor spatial smoothing around county edges; preserves central agricultural core. | **ACCEPTABLE** |
| **SoilGrids** | Centroid point buffer vs polygon zonal mean | Potential loss of soil heterogeneity in geographically large counties. | **ACCEPTABLE (Benchmark)** |
| **Crop Progress** | State-level broadcast to counties | Masks intra-state phenological variance between northern and southern counties. | **KNOWN LIMITATION (Documented)** |
| **RMA Insurance** | County-year sum across insurance plans | Blends revenue protection with yield protection; preserves net county exposure. | **BENEFICIAL / CLEAN** |

