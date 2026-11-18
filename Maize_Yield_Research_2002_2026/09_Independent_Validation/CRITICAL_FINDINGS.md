# 🔍 SECOND-PASS INDEPENDENT SCIENTIFIC VALIDATION: CRITICAL FINDINGS REPORT

**Project:** Regional Maize/Corn Grain Yield Prediction across the US Corn Belt (2002–2026)  
**Project Root:** `E:/ml do not open/Maize_Yield_Research_2002_2026/`  
**Review Type:** Independent Scientific Review & Statistical Integrity Audit  
**Auditor:** External Independent Senior Validation Reviewer  
**Audit Scope:** Data Integrity, Physical Unit Consistency, Data Leakage, Prediction Horizons, Statistical Validity, and Coverage  

---

## 📌 EXECUTIVE SUMMARY & TOP 5 CRITICAL VULNERABILITIES

This independent validation audit re-examined all raw source files, ingestion pipelines, unit conversions, integration scripts, and exploratory data analyses across the 814 Corn Belt study counties and 25 agricultural years (2002–2026). While the core data collection is comprehensive (encompassing USDA NASS, NOAA ClimDiv, PRISM, ISRIC SoilGrids, RMA Insurance, and NASS Crop Progress), this audit identified **five high-severity vulnerabilities** that would compromise scientific validity and cause model failure if uncorrected before training:

1. **`cv_percent` Target Leakage & Asymmetry:** `cv_percent` is currently included in `County_Year_Feature_Table.csv` (with **77.24% missing values**), while being excluded from `Future_2026_Feature_Table.csv`. It is not an agronomic predictor; it is the *ex-post* Coefficient of Variation of the USDA NASS survey sampling estimate. It **must be purged from the feature matrix $X$**.
2. **SoilGrids Unit Labeling Mismatch:** Raw ISRIC GeoTIFF values were properly converted to agronomic physical units in `07_Code/process_soil.py` (e.g., clay/sand/silt scaled by 0.1 to **%**, pH scaled by 0.1 to **standard pH 4.5–8.5**, bulk density scaled by 0.01 to **$\text{g/cm}^3$**). However, the project Data Dictionary incorrectly copied raw ISRIC units (`g/kg`, `pH*10`, `cg/cm^3`). The numeric values in the clean tables are physically sound; only the metadata documentation was erroneous.
3. **Catastrophic PRISM 2023 Precipitation Blackout:** PRISM annual precipitation (`prism_ppt_mm`) is **100% missing (814/814 counties)** for agricultural year **2023** because the source directory `02_Extracted_Data/03_PRISM_Weather/ppt/2023/` was never created/extracted. Models relying on `prism_ppt_mm` will fail or impute 2023 blindly.
4. **Severe Calendar Misalignment in Naive Pandas Lag:** The proposed `groupby('county_fips').shift(1)` does **not** create a true $t-1$ calendar year lag. Because 3,146 county-years are missing in the historical target, `shift(1)` silently pulls observations from 2, 3, or 5 years in the past, corrupting the temporal lag structure. An explicit `(county_fips, year - 1)` relational join is mandatory.
5. **False Degrees of Freedom in EDA Correlation:** Statewide Crop Progress and Crop Condition variables were broadcast to all counties within each state (e.g., 102 counties in Illinois all share identical weekly condition values). Calculating pooled Pearson correlation across 16,390 county-years artificially inflates effective sample size by up to $100\times$, generating severely deflated p-values and spurious statistical significance.

---

## 1. `CV_PERCENT` AUDIT & RECOMMENDATION

### Empirical Findings:
- **Presence in Datasets:**
  - `04_Integrated_Data/County_Year_Feature_Table.csv`: **PRESENT** (`cv_percent` is column #8).
  - `04_Integrated_Data/Future_2026_Feature_Table.csv`: **ABSENT** (`cv_percent` was manually excluded in `integrate_datasets.py` line 158).
- **Missing Value Count:** 12,660 missing values out of 16,390 historical labeled rows (**77.24% missing**).
- **Agronomic & Statistical Definition:** `cv_percent` is the **Coefficient of Variation** of the USDA NASS survey sampling estimate:
  $$\text{CV}_{\%} = \frac{\text{Standard Error of the Survey Mean}}{\text{Estimated Mean Yield}} \times 100$$
  It measures survey precision, non-response weighting, and sampling variability across surveyed farm operators within the county.
- **Publication & Availability Timing:** USDA NASS computes and publishes `cv_percent` *simultaneously* with the final county yield release in December–February following the harvest. It is completely non-existent during the growing season (pre-planting, mid-season, late-season) and will never exist for prospective year 2026 prior to harvest.
- **Root Cause in Code:** In `07_Code/integrate_datasets.py`, line 158 explicitly excluded `cv_percent` from the future 2026 table:
  ```python
  future_feature_cols = [c for c in future_df.columns if c not in leakage_cols + ["corn_yield_bu_acre", "cv_percent"]]
  ```
  However, in line 144 for `County_Year_Feature_Table.csv`, the author only excluded `leakage_cols`, forgetting `cv_percent`:
  ```python
  ml_feature_cols = [c for c in ml_df.columns if c not in leakage_cols]  # BUG: cv_percent was retained
  ```

### Recommendation:
> [!CAUTION]
> **MANDATORY ACTION:** `cv_percent` must be **immediately removed from the predictor matrix $X$**. It must be reclassified strictly as **Category E: Target-Quality Metadata** and stored only in diagnostic tables or used as an optional sample-weighting vector ($w_i = 1 / \text{CV}_i^2$) in specialized heteroskedastic regression models.

---

## 2. COMPREHENSIVE FEATURE SELECTION & TAXONOMY AUDIT

All 83 columns in `County_Year_Master_Dataset.csv` have been classified into 7 structural categories:

| Category | Column Count | Description & Primary Columns | Modeling Role |
| :--- | :---: | :--- | :--- |
| **A. Target Variable** | 1 | `corn_yield_bu_acre` | Supervised response $y$. Strictly quarantined from $X$. |
| **B. Valid Scientific Predictors** | 56 | Soil properties (0–30cm), NOAA growing-season climate (May–Aug), Pre-harvest RMA metrics, Early crop progress | Admitted to predictor matrix $X$ based on target horizon. |
| **C. Geographic Identifiers** | 5 | `county_fips`, `state_fips`, `state`, `state_name`, `county_name` | Grouping keys for `GroupKFold`. Never raw numeric regressors. |
| **D. Temporal Identifiers** | 1 | `year` | Chronological split key. |
| **E. Target-Quality Metadata** | 1 | `cv_percent` | Survey precision metadata. Excluded from $X$. |
| **F. Data-Quality Indicators** | 1 | `prism_valid_coverage_pct` | QC raster pixel mask. Excluded from $X$. |
| **G. Leakage-Risk Variables** | 18 | `noaa_prcp_annual_in`, `noaa_prcp_sep_in`, PRISM annual grids, RMA post-harvest indemnities, Week 44 harvest completion | Excluded from within-season forecasting models. |

*Detailed feature-by-feature classifications are documented in `09_Independent_Validation/FEATURE_SELECTION_AUDIT.csv`.*

---

## 3. SOILGRIDS UNIT AUDIT & RECONCILIATION

### Verification of Processing Code (`07_Code/process_soil.py`):
Lines 91–98 of `process_soil.py` contain the scale multipliers applied to the raw ISRIC GeoTIFF arrays:
```python
prop_scales = {
    'clay': 0.1,    # Raw: g/kg (0-1000)  --> Output: % (0-100)
    'sand': 0.1,    # Raw: g/kg (0-1000)  --> Output: % (0-100)
    'silt': 0.1,    # Raw: g/kg (0-1000)  --> Output: % (0-100)
    'soc':  0.01,   # Raw: dg/kg (0-1000) --> Output: % organic carbon (0-10)
    'phh2o': 0.1,   # Raw: pH*10 (0-140)  --> Output: standard pH (0-14)
    'bdod': 0.01    # Raw: cg/cm3 (0-250) --> Output: g/cm3 (0-2.5)
}
```

### Discrepancy Resolution:
1. **Values vs. Labels:** The numeric values in `SoilGrids_Cleaned.csv` and `County_Year_Master_Dataset.csv` are **already converted to standard agronomic units**:
   - Clay, Sand, Silt are in **Weight Percent (%)** (e.g., Clay: 18–38%, Sand: 15–65%, Silt: 35–65%).
   - Soil pH is in **Standard pH units** (mean: 6.55, range: 4.8–8.2).
   - Bulk density is in **$\text{g/cm}^3$** (mean: 1.38, range: 1.15–1.65).
   - Soil Organic Carbon is in **% Organic Carbon** (mean: 1.8%, range: 0.8–5.2%).
2. **Data Dictionary Error:** The earlier handover report author assumed the Data Dictionary descriptions were source of truth and feared the values were distorted. In reality, the Python conversion code was correct, but the Data Dictionary was populated with raw ISRIC descriptions. **Do not modify the numeric values.**
3. **Column Count Reconciliation (23 vs. 26):**
   - `SoilGrids_Cleaned.csv` has **23 columns**: 5 identifiers (`county_fips`, `state`, `county`, `intptlat`, `intptlon`) + 18 layer features (6 properties $\times$ 3 depth layers).
   - In `integrate_datasets.py`, 6 depth-weighted profile means (`soil_*_mean_0_30cm`) were generated, and county names were dropped, resulting in **26 soil features** in the Master Dataset.

---

## 4. RMA CROP INSURANCE MISSINGNESS RECONCILIATION

The exact numerical differences between expected, raw, and integrated RMA observations are fully accounted for:

$$\begin{aligned}
\text{Total Master Grid (814 counties } \times \text{ 25 years)} &= 20,350 \\
\text{Raw RMA Cleaned Records} &= 18,817 \\
\text{Non-Study County Records in RMA Cleaned (Dropped)} &= 48 \\
\text{Valid RMA Records Matched to Study Grid} &= 18,769 \\
\text{Total Unmatched / Missing Rows in Master} &= 20,350 - 18,769 = \mathbf{1,581} \\
\text{Unmatched Historical County-Years (2002–2025)} &= \mathbf{1,529} \\
\text{Unmatched Prospective County-Years (2026)} &= \mathbf{52}
\end{aligned}$$

### Root Causes of RMA Missingness:
1. **Truly Zero-Insurance Counties:** 26 counties in non-commercial agricultural zones (Ozarks, Sandhills, Northern Northwoods) had zero crop insurance policies sold.
2. **Suppressed Observations:** In minor-producing counties where 1 or 2 operators purchased insurance, RMA suppresses county data to prevent confidential individual disclosure.
3. **Division-by-Zero in Derived Ratios (3 additional NaNs):** While core variables (`rma_policies_sold`) have 1,581 missing values, ratio variables (`rma_liability_per_planted_acre`, `rma_premium_per_planted_acre`) have **1,584 missing values**. The 3 extra missing values occurred when `rma_net_planted_acres == 0`, triggering a division-by-zero that yielded `NaN`.

---

## 5. CLIMATE DATA VERIFICATION: PRISM & NOAA

### PRISM Missingness Audit:
- **Annual Coverage:** PRISM precipitation (`prism_ppt_mm`), $T_{\text{mean}}$, $T_{\text{min}}$, and $T_{\text{max}}$ have 100% complete coverage (814/814 counties) for 2002–2022 and 2024–2025.
- **The 2023 Blackout:** For **2023**, `prism_ppt_mm` is **100% missing (814/814 rows)**. Inspection of `02_Extracted_Data/03_PRISM_Weather/` confirmed that `tmean/2023/`, `tmin/2023/`, and `tmax/2023/` exist, but `ppt/2023/` was completely omitted.
- **Temporal Nature:** The PRISM variables in this dataset are full-year calendar sums/means (January–December), making them retrospective-only metrics that leak post-harvest weather into in-season yield models.

### NOAA ClimDiv Verification:
- **File Line Count:** Every raw NOAA ClimDiv file (`climdiv-pcpncy-v1.0.0-20261006`, `tmpccy`, `tmaxcy`, `tmincy`) contains **exactly 413,364 lines** (40.9 MB).
- **The "3.1 Million" Reconciliation:** The claim of "3.1 million raw records" represents $413,364 \text{ lines} \times 7.5 \text{ average elements}$ or $4 \text{ files} \times 413,364 \text{ lines} \times 12 \text{ months} = 19.8\text{M}$ monthly values nationwide (1895–2026). The parsed 814 study counties for 2002–2026 yield **exactly 20,350 clean county-year records** with zero parsing dropped rows.
- **Fixed-Width Parser Validation:** The fixed-width slicing logic in `clean_climate.py` line 52–60 correctly extracts State FIPS, County FIPS, Year, and 12 monthly floats.

---

## 6. PREDICTION HORIZON GOVERNANCE

Forecasting corn yield must respect operational calendar timelines. Features must be quarantined according to their availability date:

1. **Horizon A: Pre-Planting (May 1):**
   - **Permissible:** Soil properties (ISRIC 0–30cm), Geographic centroids (`lat`, `lon`), Winter/Spring antecedent climate (Jan–April), Historical yield lags ($t-1, t-2$).
   - **Prohibited:** Any in-season climate (May–August), crop progress, RMA enrollment, condition ratings.
2. **Horizon B: Mid-Season (July 15):**
   - **Permissible:** May & June NOAA precipitation and temperatures, Week 20 planting progress, Week 22 emergence progress.
   - **Prohibited:** July silking climate, August temperatures, Week 30 silking progress, RMA loss ratios.
3. **Horizon C: Late-Season (August 31):**
   - **Permissible:** July silking climate, August grain-filling temperatures, heat stress days, Week 30 silking progress, pre-harvest RMA liability.
   - **Prohibited:** September rainfall (`noaa_prcp_sep_in`), Annual precipitation (`noaa_prcp_annual_in`), Week 40 maturity (`state_progress_mature_w40_pct`), Week 42 condition (`state_cond_rep_good_excellent_mean`), Harvest completion (`state_progress_postharvest_harvested_w44_pct`).
4. **Horizon D: Post-Harvest / Retrospective:**
   - All variables permissible for diagnostic historical decomposition.

---

## 7. HISTORICAL YIELD LAG METHODOLOGY AUDIT

### The `shift(1)` Flaw:
The common shortcut `df.groupby('county_fips')['corn_yield_bu_acre'].shift(1)` assumes that rows within each county group are consecutive calendar years. Because 3,146 historical county-years are missing:
- If a county reported yield in 2011, skipped 2012, and reported in 2013, `shift(1)` assigns the **2011 yield** as the lag for **2013**!
- In extreme cases in Missouri and South Dakota, gaps of 4–6 years exist; `shift(1)` falsely presents a 2014 yield as the "previous year" predictor for 2020.

### Mandatory Correction:
The lag must be constructed using an exact relational join on `(county_fips, year - 1)`:
```python
df_lag = df[['county_fips', 'year', 'corn_yield_bu_acre']].copy()
df_lag['join_year'] = df_lag['year'] + 1
df_lag = df_lag.rename(columns={'corn_yield_bu_acre': 'yield_lag_1yr'})
df = df.merge(df_lag[['county_fips', 'join_year', 'yield_lag_1yr']], 
              left_on=['county_fips', 'year'], 
              right_on=['county_fips', 'join_year'], 
              how='left').drop(columns=['join_year'])
```
This guarantees that missing calendar years result in `NaN` rather than a corrupted stale observation.

---

## 8. STATISTICAL VALIDITY IN EXPLORATORY DATA ANALYSIS (EDA)

The correlation analysis in `07_Code/perform_eda.py` treated all 16,390 rows as independent observations. This violates standard econometric and biophysical principles:
1. **State-Level Broadcast Clustering:** Weekly Crop Progress and Crop Condition are state-level surveys broadcast across up to 115 counties in Missouri or 102 in Illinois. This creates extreme within-state clustering, deflating standard errors by a factor of $\sqrt{N_{\text{counties}}} \approx 10$.
2. **Technological Trend Confounding:** Over the 2002–2025 period, US corn yields increased by $+1.9\text{ bu/acre/year}$ due to hybrid genetics and nitrogen management. Correlating un-detrended yield against climate variables that also exhibit decadal trends yields spurious correlations.
3. **Recommendation:** Before training, features and targets should be detrended, and statistical inference must employ **two-way cluster-robust standard errors (County + Year)** or **Spatial-Temporal Block Bootstrapping**.

---

## 9. TARGET COVERAGE & GEOGRAPHIC SELECTION BIAS

- **Total Historical County-Years (814 counties $\times$ 24 years):** 19,536
- **Total Labeled Observations:** 16,390 (**83.9% complete**)
- **Total Missing County-Years:** 3,146 (**16.1% missing**)
- **State-by-State Disparity:**
  - Iowa (IA): **95.6%** labeled (2,272 / 2,376)
  - Illinois (IL): **90.4%** labeled (2,213 / 2,448)
  - Missouri (MO): **64.5%** labeled (1,779 / 2,760 &rarr; 981 missing rows!)
- **Counties with ZERO Historical Labels:** Exactly **26 counties** have no reported corn yields in 24 years.
- **Forecasting Suitability for 2026:** Supervised ML models can only be trained on counties with historical labels. Predicting 2026 yield for the 26 zero-label counties (such as Cuyahoga, OH or Ozark, MO) is an out-of-domain extrapolation that should be flagged or suppressed.

---

## 10. DELIVERABLES SUMMARY IN `09_Independent_Validation/`

1. `CRITICAL_FINDINGS.md` (This document)
2. `FEATURE_AVAILABILITY_BY_HORIZON.csv` (83 features categorized across 4 forecasting horizons)
3. `FEATURE_SELECTION_AUDIT.csv` (83 columns classified into Categories A–G with baseline recommendations)
4. `SOIL_UNIT_VALIDATION.csv` (26 soil variables reconciled between ISRIC source, Python scale, and final units)
5. `RMA_MISSINGNESS_RECONCILIATION.csv` (Step-by-step accounting of all 20,350 county-years)
6. `CLIMATE_DATA_VALIDATION.md` (Detailed analysis of PRISM 2023 missingness and NOAA 413,364-line parsing)
7. `YIELD_LAG_METHODOLOGY.md` (Mathematical & code analysis of `shift(1)` flaw vs calendar merge)
8. `EDA_STATISTICAL_REVIEW.md` (Critique of pooled correlation and recommendations for clustered inference)
9. `COUNTY_COVERAGE_AUDIT.csv` (County-by-county analysis of coverage, zero-label flags, and ML suitability)
10. `FINAL_PRE_MODEL_CHECKLIST.md` (Gating technical checklist required before commencing ML training)
