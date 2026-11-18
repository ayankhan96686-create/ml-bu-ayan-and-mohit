# Phase 5 Existing Pipeline Audit & Script Provenance
## Comprehensive Traceability of Code, Data Flow, and Deliverables

**Project Root:** `E:/ml do not open/Maize_Yield_Research_2002_2026/`  
**Audit Directory:** [`11_Independent_Model_Validation/01_Project_Audit/`](file:///E:/ml%20do%20not%20open/Maize_Yield_Research_2002_2026/11_Independent_Model_Validation/01_Project_Audit/)  
**Status:** Independent Audit Milestone Deliverable  
**Associated File Inventory:** [`PHASE5_PROJECT_INVENTORY.csv`](file:///E:/ml%20do%20not%20open/Maize_Yield_Research_2002_2026/11_Independent_Model_Validation/01_Project_Audit/PHASE5_PROJECT_INVENTORY.csv) (1,174 files logged)

---

## 1. End-to-End Pipeline Lineage & Script Provenance

Every dataset and deliverable across the research repository was audited to establish its exact generating script and operational dependencies:

```
[01_Raw_Data] (15 Source Directories: NOAA, PRISM, NASS, SoilGrids, RMA, Census)
      │
      ▼
[07_Code/ Domain Extraction & Cleaning Scripts]
  ├── clean_yield.py           ──► 03_Cleaned_Data/USDA_Yield_Cleaned.csv (16,390 rows)
  ├── clean_climate.py         ──► 03_Cleaned_Data/NOAA_Climate_Cleaned.csv (20,350 rows)
  ├── process_prism.py         ──► 03_Cleaned_Data/PRISM_Climate_Cleaned.csv (20,350 rows)
  ├── process_soil.py          ──► 03_Cleaned_Data/SoilGrids_Cleaned.csv (814 rows)
  └── clean_agriculture.py     ──► 03_Cleaned_Data/RMA_Cleaned.csv (18,817 rows)
                               ──► 03_Cleaned_Data/Crop_Progress_Cleaned.csv (216 rows)
                               ──► 03_Cleaned_Data/Crop_Condition_Cleaned.csv (216 rows)
      │
      ▼
[07_Code/integrate_datasets.py]
  ├── County_Year_Master_Dataset.csv (83 cols, 20,350 rows: 814 counties × 25 years)
  ├── County_Year_Feature_Table.csv  (79 cols, 16,390 rows: Labeled historical observations)
  └── Future_2026_Feature_Table.csv  (77 cols, 814 rows: Unlabeled 2026 inference table)
      │
      ▼
[10_Final_Pre_Modeling_Gate/ Scripts]
  ├── execute_phase0_corrections.py ──► CORRECTED_COLUMN_MANIFEST.csv (88 cols)
  │                                 ──► COUNTY_ELIGIBILITY_RESOLVED.csv (814 counties)
  ├── build_purified_datasets.py    ──► Purified_Corn_Belt_Historical_Dataset.csv (83 cols)
  │                                 ──► Purified_Corn_Belt_Future_2026_Dataset.csv (82 cols)
  ├── run_phase1_benchmarking.py    ──► BASELINE_MODEL_BENCHMARK_RESULTS.csv
  │                                 ──► baseline_predictions_val_test.csv (3,730 predictions)
  ├── run_phase2_reliability_analysis.py ──► GEOGRAPHIC_GENERALIZATION_RESULTS.csv
  │                                      ──► FEATURE_IMPORTANCE_ANALYSIS.csv
  │                                      ──► STATE_YEAR_ERROR_ANALYSIS.csv
  └── generate_phase3_predictions.py──► PROVISIONAL_2026_CORN_YIELD_PREDICTIONS.csv (814 rows)
```

---

## 2. Script-by-Script Verification & Purpose Matrix

| Source Script | Execution Location | Primary Inputs | Primary Output Deliverables | Execution Status |
| :--- | :--- | :--- | :--- | :---: |
| `clean_yield.py` | `07_Code/` | `01_Raw_Data/01_USDA_NASS_Yield/` | `03_Cleaned_Data/USDA_Yield_Cleaned.csv` | **Verified** |
| `clean_climate.py` | `07_Code/` | `01_Raw_Data/05_NOAA_Climate_Divisions/` | `03_Cleaned_Data/NOAA_Climate_Cleaned.csv` | **Verified** |
| `process_prism.py` | `07_Code/` | `02_Extracted_Data/03_PRISM_Weather/` | `03_Cleaned_Data/PRISM_Climate_Cleaned.csv` | **Verified** |
| `process_soil.py` | `07_Code/` | `01_Raw_Data/11_SoilGrids_Soil_Profiles/` | `03_Cleaned_Data/SoilGrids_Cleaned.csv` | **Verified** |
| `clean_agriculture.py` | `07_Code/` | `01_Raw_Data/12_Planting_and_Crop_Management/` | `RMA_Cleaned.csv`, `Crop_Progress_Cleaned.csv` | **Verified** |
| `integrate_datasets.py`| `07_Code/` | All files in `03_Cleaned_Data/` | `County_Year_Master_Dataset.csv`, `County_Year_Feature_Table.csv` | **Verified** |
| `build_purified_datasets.py` | `10_Final_Pre_Modeling_Gate/` | `County_Year_Feature_Table.csv`, `USDA_Yield_Cleaned.csv` | `Purified_Corn_Belt_Historical_Dataset.csv` (8,006,235 B) | **Verified** |
| `run_phase1_benchmarking.py` | `10_Final_Pre_Modeling_Gate/` | `Purified_Corn_Belt_Historical_Dataset.csv` | `BASELINE_MODEL_BENCHMARK_RESULTS.csv` | **Verified** |
| `run_phase2_reliability_analysis.py` | `10_Final_Pre_Modeling_Gate/` | `Purified_Corn_Belt_Historical_Dataset.csv` | `GEOGRAPHIC_GENERALIZATION_RESULTS.csv` | **Verified** |
| `generate_phase3_predictions.py` | `10_Final_Pre_Modeling_Gate/` | `Purified_Corn_Belt_Future_2026_Dataset.csv` | `PROVISIONAL_2026_CORN_YIELD_PREDICTIONS.csv` | **Verified** |

---

## 3. Discrepancy Findings in Existing Scripts

1. **Hardcoded Annual PRISM vs. Seasonal 2026 PRISM in `process_prism.py`:**
   Lines 51–53 process annual rasters for 2002–2025, but lines 77–100 process monthly grids for May–October 2026. This caused the temporal period mismatch identified in Phase 0.
2. **Missing PRISM 2023 Precipitation Raster Directory:**
   The directory `02_Extracted_Data/03_PRISM_Weather/ppt/2023` was never created or downloaded, leading to 814 missing values in Master and 632 in Feature Table.
3. **Naive Groupby Shift in Earlier Drafts:**
   Earlier notebooks attempted `df.groupby('county_fips')['corn_yield_bu_acre'].shift(1)`, which corrupted 977 records when survey years were skipped. This was rectified in `build_purified_datasets.py` using a regular Cartesian calendar spine.
4. **LOSO Sample Count (16,388 vs 16,390):**
   Investigated in Phase H below: exactly 2 county-year observations were dropped in earlier exploratory scripts due to state assignment lookup mismatches.

---
*Certified and Prepared:*  
**Principal ML Engineer & Senior Agricultural Data Scientist**

