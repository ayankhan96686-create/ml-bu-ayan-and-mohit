# COMPLETE DATA DOCUMENTATION: MAIZE YIELD PREDICTION FRAMEWORK (2002–2026)

**Project:** County-Level Maize Grain Yield Prediction Framework  
**Authors:** Ayan Khan and Mohit  
**Active Repository Commit:** `b00a8367c2173be6974004e99d2a3f43ec833d15`  
**Study Horizon:** 2002–2026 (2002–2025 Historical Ground Truth; 2026 Prospective)  
**Total Historical Observations:** 16,390 certified county-years  

---

## 1. Master Data Inventory & Original Data Sources

The research framework harmonizes seven primary multi-modal data streams across 9 US Corn Belt States (`IA`, `IL`, `IN`, `MN`, `MO`, `NE`, `OH`, `SD`, `WI`):

| Domain | Dataset Name | Primary Provider / Agency | Spatial Resolution | Temporal Frequency | Historical Coverage | File Format | Role in Prediction Pipeline |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Ground Truth Target** | USDA NASS QuickStats Corn Yield | USDA National Agricultural Statistics Service | County polygon | Annual | 2002–2025 | CSV tabular | Target label (`corn_yield_bu_acre`), harvested acres, planted acres |
| **Surface Weather** | PRISM 4km Gridded Climate | PRISM Climate Group (Oregon State Univ.) | 4 km grid | Daily | 2002–2026 | Bil / TIF / CSV | Daily gridded precipitation (`ppt`), $T_{\max}$, $T_{\min}$, $T_{\text{mean}}$ |
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
  $$\bar{X}_{c,t} = \frac{\sum_{i \in c} w_i X_{i,t}}{\sum_{i \in c} w_i}$$
  where $w_i$ represents the agricultural cropland fraction of cell $i$ determined by the USDA Cropland Data Layer (CDL) corn mask.

---

## 4. County FIPS and Year-Based Merging Methodology

The complete dataset is synthesized via a balanced relational spine:
1. **Cartesian Product Spine:** Created for all 788 agricultural counties across all 25 study years ($788 \times 25 = 19,700$ theoretical slots).
2. **Left Join on Target Yields:** Historical USDA NASS reported yields merged on `(county_fips, year)`. Records without official reported yields (primarily small-acreage non-reporting county-years) are excluded, yielding exactly **16,390 certified historical observations**.
3. **Left Join on Environmental Predictors:** Surface climate (PRISM/NOAA) and static soil tables merged on `(county_fips, year)` and `county_fips` respectively.
4. **Broadcast State Phenology:** State-level USDA NASS weekly Crop Progress and Condition surveys merged on `(state_fips, year)`.
5. **Autoregressive Lag Construction:** Computed strictly over sorted chronological historical observations per county using shifted indexing ($t-k \ge 1$) to eliminate target leakage.

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
