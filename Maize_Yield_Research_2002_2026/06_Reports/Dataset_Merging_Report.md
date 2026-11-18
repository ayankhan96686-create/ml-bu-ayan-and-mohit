# Dataset Merging & Integration Technical Report

**Project Title:** Maize (Corn) Grain Yield Prediction Using Machine Learning (2002–2025)  
**Study Region:** 9 US Corn Belt States (814 Counties)  
**Temporal Domain:** 2002–2026  
**Report Directory:** `E:/ml do not open/Maize_Yield_Research_2002_2026/06_Reports/`  

---

## 1. Executive Summary

This report provides the technical specifications, join cardinalities, match rates, and row multiplication checks for the multi-domain dataset integration executed by `07_Code/integrate_datasets.py`.

### Key Integration Metrics:
- **Base Cartesian Spine:** 814 counties $\times$ 25 years = **20,350 county-year units**.
- **Master Dataset Dimensions:** **20,350 rows $\times$ 83 columns**.
- **Historical ML Feature Table Dimensions:** **16,390 rows $\times$ 79 columns** (2002–2025 labeled observations).
- **Future Prediction Table Dimensions:** **814 rows $\times$ 77 columns** (2026 unlabeled prospective set).
- **Row Multiplication Factor:** **Exactly 1.0000** (0 unexpected row expansions).
- **Duplicate Join Keys:** **Exactly 0**.

---

## 2. Step-by-Step Join Lineage & Cardinality

### Step 0: Base Cartesian Spine Construction
- **Sources:** `00_Project_Admin/Study_Area_Counties_FIPS.csv` (814 counties) and calendar years $2002 \dots 2026$ (25 years).
- **Spine Size:** $814 \times 25 = 20,350$ rows.
- **Keys Generated:** `county_fips` (5-digit string), `state_fips` (2-digit string), `state` (2-letter abbreviation), `state_name`, `county_name`, and `year` (int).

### Step 1: Merge USDA Corn Grain Yield
- **Input Table:** `03_Cleaned_Data/USDA_Yield_Cleaned.csv` (16,390 rows).
- **Join Key:** `(county_fips, year)`
- **Cardinality:** 1-to-1 Left Join.
- **Match Rate:** 16,390 matched ($80.54\%$ of spine).
- **Unmatched Rows:** 3,960 historical non-reporting county-years + 814 in 2026. Stored as `NaN`.

### Step 2: Merge NOAA CIRS Climate
- **Input Table:** `03_Cleaned_Data/NOAA_Climate_Cleaned.csv` (20,350 rows).
- **Join Key:** `(county_fips, year)`
- **Cardinality:** 1-to-1 Left Join.
- **Match Rate:** 20,350 matched ($100.0\%$).
- **Features Added:** 14 features (`noaa_prcp_*`, `noaa_tmean_*`, `noaa_tmax_*`, `noaa_tmin_*`, `noaa_heat_stress_*`).

### Step 3: Merge PRISM Gridded Climate
- **Input Table:** `03_Cleaned_Data/PRISM_Climate_Cleaned.csv` (20,350 rows).
- **Join Key:** `(county_fips, year)`
- **Cardinality:** 1-to-1 Left Join.
- **Match Rate:** 19,718 matched ($96.90\%$).
- **Unmatched / Null Rows:** 632 rows in 2023 due to missing raw ppt raster.

### Step 4: Merge ISRIC SoilGrids 2.0 (Static Broadcast)
- **Input Table:** `03_Cleaned_Data/SoilGrids_Cleaned.csv` (814 rows).
- **Join Key:** `county_fips`
- **Cardinality:** 1-to-many Broadcast Join (1 profile row broadcast to 25 years).
- **Match Rate:** 813 counties matched ($99.88\%$).
- **Unmatched / Masked Rows:** 1 county (Cook Co, IL: 25 rows `NaN`).

### Step 5: Merge USDA RMA Federal Crop Insurance
- **Input Table:** `03_Cleaned_Data/RMA_Cleaned.csv` (18,817 rows).
- **Join Key:** `(county_fips, year)`
- **Cardinality:** 1-to-1 Left Join.
- **Match Rate:** 18,817 matched ($92.47\%$).
- **Unmatched Rows:** 1,533 county-years with zero commercial corn insurance policies.

### Step 6 & 7: Merge State Crop Condition & Crop Progress (State Broadcast)
- **Input Tables:** `Crop_Condition_Cleaned.csv` (225 rows) and `Crop_Progress_Cleaned.csv` (225 rows).
- **Join Key:** `(state_fips, year)`
- **Cardinality:** 1-to-many Broadcast Join (9 state records broadcast to internal counties).
- **Match Rate:** 20,350 matched ($100.0\%$).

---

## 3. Output Table Schema Partitioning

```
+-----------------------------------------------------------------------------------+
| OUTPUT TABLE PARTITIONING SCHEME                                                 |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  [ County_Year_Master_Dataset.csv ]                                               |
|  - 20,350 Rows x 83 Columns (2002–2026, All 814 Counties)                        |
|  - Includes: Administrative keys, Target yield, CV%, Climate, Soil,               |
|              Pre-harvest insurance, Crop condition, Crop progress,                |
|              AND Quarantined post-harvest leakage variables.                      |
|                                                                                   |
|  [ County_Year_Feature_Table.csv ]                                                |
|  - 16,390 Rows x 79 Columns (2002–2025 Labeled ML Training Subset)                |
|  - Target: 'corn_yield_bu_acre' (Guaranteed non-null, genuine observations)      |
|  - Strictly PURGED of all 4 post-harvest leakage variables:                       |
|    * rma_postharvest_policies_indemnified                                         |
|    * rma_postharvest_indemnity_amount                                            |
|    * rma_postharvest_loss_ratio                                                   |
|    * state_progress_postharvest_harvested_w44_pct                                 |
|                                                                                   |
|  [ Future_2026_Feature_Table.csv ]                                                |
|  - 814 Rows x 77 Columns (2026 Prospective Prediction Period)                     |
|  - Target 'corn_yield_bu_acre' and 'cv_percent' completely OMITTED                 |
|  - All post-harvest leakage variables completely OMITTED                          |
|  - Preserves prospective blind forecasting integrity                              |
|                                                                                   |
+-----------------------------------------------------------------------------------+
```

---

## 4. Verification Check Script

```python
import pandas as pd

master = pd.read_csv("04_Integrated_Data/County_Year_Master_Dataset.csv", low_memory=False)
feat = pd.read_csv("04_Integrated_Data/County_Year_Feature_Table.csv", low_memory=False)
future = pd.read_csv("04_Integrated_Data/Future_2026_Feature_Table.csv", low_memory=False)

assert len(master) == 20350
assert len(feat) == 16390
assert len(future) == 814
assert master.duplicated(subset=['county_fips', 'year']).sum() == 0
assert feat.duplicated(subset=['county_fips', 'year']).sum() == 0
assert future.duplicated(subset=['county_fips', 'year']).sum() == 0
print("Dataset merging audit: PASSED.")
```

