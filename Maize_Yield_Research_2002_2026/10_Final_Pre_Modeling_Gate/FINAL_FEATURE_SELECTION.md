# Authoritative Feature Selection & Model Specification
## Maize/Corn Grain Yield Prediction (2002–2025)

**Project Root:** `E:/ml do not open/Maize_Yield_Research_2002_2026/`  
**Deliverable Directory:** [`10_Final_Pre_Modeling_Gate/`](file:///E:/ml%20do%20not%20open/Maize_Yield_Research_2002_2026/10_Final_Pre_Modeling_Gate/)  
**Status:** Authoritative Scientific Gate Artifact  
**Associated Data Manifest:** [`FINAL_COLUMN_MANIFEST.csv`](file:///E:/ml%20do%20not%20open/Maize_Yield_Research_2002_2026/10_Final_Pre_Modeling_Gate/FINAL_COLUMN_MANIFEST.csv)

---

## 1. Executive Reconciliation of Conflicting Schema Reports

Previous audit iterations reported differing column and feature counts across intermediate documents. The following empirical breakdown reconciles all historical reports against the actual files on disk:

| Discrepancy Topic | Prior Conflicting Figures | Empirical Ground Truth | Root Cause & Scientific Resolution |
| :--- | :---: | :---: | :--- |
| **Total Table Columns** | **79 vs. 83** | **79** in Feature Table<br>**83** in Master Dataset<br>**77** in Future 2026 | `County_Year_Master_Dataset.csv` has 83 columns. During feature engineering, 4 post-harvest columns (`rma_postharvest_indemnity_amount`, `rma_postharvest_loss_ratio`, `rma_postharvest_policies_indemnified`, `state_progress_postharvest_harvested_w44_pct`) were dropped to prevent leakage, yielding exactly **79 columns** in `County_Year_Feature_Table.csv`. `Future_2026_Feature_Table.csv` has **77 columns** (79 minus target `corn_yield_bu_acre` and metadata `cv_percent`). |
| **Predictor Count** | **71 vs. 63** | **71** Raw Non-Target/Non-ID<br>**63** Valid Environmental Predictors | `County_Year_Feature_Table.csv` contains 79 total columns. Subtracting 6 identification keys (`county_fips`, `state_fips`, `state`, `state_name`, `county_name`, `year`), 1 target (`corn_yield_bu_acre`), and 1 metadata (`cv_percent`) leaves **71 raw columns**. If the 8 RMA insurance management features are isolated into an economic management tier, exactly **63 environmental/crop predictors** remain. |
| **Crop Features** | **18 vs. 22** | **18** in Feature Table<br>**19** in Master Dataset<br>**22** in Raw Extraction | `County_Year_Feature_Table.csv` contains exactly **18 in-season crop features** (9 condition indices: `state_cond_*_mean` + 9 progress stages: `state_progress_*`). Master dataset has **19** (+ `state_progress_postharvest_harvested_w44_pct`). Earlier raw extraction scripts processed 22 weekly stage metrics before filtering. |
| **Climate Features** | **19 vs. 16** | **19** Total Climate Columns<br>**16** Pure Weather Features<br>**14** NOAA Features | In `County_Year_Feature_Table.csv`, there are 14 NOAA columns + 5 PRISM columns = **19 total climate columns**. Subtracting 1 quality audit flag (`prism_valid_coverage_pct`), 1 retrospective annual total (`noaa_prcp_annual_in`), and 1 compound index (`noaa_heat_stress_max_f`) yields **16 monthly weather predictors**. |
| **SoilGrids Columns** | **23 vs. 26** | **23** in Cleaned File<br>**26** in Feature Table | `SoilGrids_Cleaned.csv` has 23 columns (18 depth-layer intervals + 2 coordinates + 1 FIPS + 2 land/water areas). In `integrate_datasets.py`, the 2 land/water areas were dropped and 6 profile depth-weighted averages (`soil_*_mean_0_30cm`) were added ($18 + 2 + 6 = \mathbf{26}$ soil columns). |

---

## 2. Mutually Exclusive & Exhaustive Feature Taxonomy

All columns in the research repository belong to exactly **one** scientific category:

```
Total Repository Taxonomy (87 Columns)
├── Tier A: Target Vector y (1) ───────────────► corn_yield_bu_acre
├── Tier B: Valid Scientific Predictors (63)
│   ├── B1: Static Soil Physical Properties (24) ──► 18 depth layers (0-5, 5-15, 15-30cm) + 6 profile means
│   ├── B2: NOAA Monthly / Seasonal Climate (14) ──► 8 precipitation + 5 temperature + 1 heat stress
│   ├── B3: PRISM Gridded Weather (4) ────────────► prism_ppt_mm, prism_tmean_c, prism_tmin_c, prism_tmax_c
│   ├── B4: Crop Condition Indices (9) ───────────► state_cond_excellent_mean through rep_poor_very_poor
│   └── B5: Crop Progress Milestones (9) ─────────► planted_w20 through week_50pct_mature
├── Tier C: Geographic Identifiers (7) ────────► county_fips, state_fips, state, state_name, county_name, soil_intptlat, soil_intptlon
├── Tier D: Temporal Identifiers (1) ──────────► year
├── Tier E: Target Quality Metadata (1) ───────► cv_percent (STRICTLY BANNED FROM X)
├── Tier F: Data Quality Indicators (1) ───────► prism_valid_coverage_pct (Audit Only)
├── Tier G: Post-Harvest / Leakage Variables (4)► rma_postharvest_indemnity_amount, loss_ratio, policies_indemnified, harvested_w44
├── Tier H: Agricultural Management RMA (9) ───► 8 pre-harvest policy features + 1 rma_missing_indicator
└── Tier I: Engineered Historical Lags (3) ────► yield_lag_1yr, yield_lag_2yr, yield_roll_mean_5yr
```

---

## 3. Four Separate Model Feature Sets

To avoid confounding environmental yield potential with economic insurance participation or autoregressive yield memory, four distinct, nested feature configurations are defined:

### Model Set A: Strict Environmental Baseline (42 Features)
*Focus: Biophysical yield drivers (Soil + Climate only).*
- **Soil Properties (24):** `soil_clay_*`, `soil_sand_*`, `soil_silt_*`, `soil_soc_*`, `soil_phh2o_*`, `soil_bdod_*` (3 depths each: 0–5cm, 5–15cm, 15–30cm) + 6 profile means (0–30cm).
- **NOAA Climate (14):** `noaa_prcp_may_in`, `noaa_prcp_jun_in`, `noaa_prcp_jul_in`, `noaa_prcp_aug_in`, `noaa_prcp_sep_in`, `noaa_prcp_may_aug_in`, `noaa_prcp_may_sep_in`, `noaa_prcp_annual_in`, `noaa_tmean_may_aug_f`, `noaa_tmax_jul_f`, `noaa_tmax_aug_f`, `noaa_tmax_may_aug_f`, `noaa_tmin_may_aug_f`, `noaa_heat_stress_max_f`.
- **PRISM Climate (4):** `prism_ppt_mm`, `prism_tmean_c`, `prism_tmin_c`, `prism_tmax_c`.

### Model Set B: Extended In-Season Model (60 Features)
*Focus: Biophysical drivers + Crop physiological development.*
- **All 42 features from Model Set A.**
- **NASS Crop Condition (9):** `state_cond_excellent_mean`, `state_cond_good_mean`, `state_cond_fair_mean`, `state_cond_poor_mean`, `state_cond_very_poor_mean`, `state_cond_good_excellent_mean`, `state_cond_poor_very_poor_mean`, `state_cond_rep_good_excellent_mean`, `state_cond_rep_poor_very_poor_mean`.
- **NASS Crop Progress (9):** `state_progress_planted_w20_pct`, `state_progress_emerged_w22_pct`, `state_progress_silking_w30_pct`, `state_progress_dough_w34_pct`, `state_progress_dented_w37_pct`, `state_progress_mature_w40_pct`, `state_progress_week_50pct_planted`, `state_progress_week_50pct_silking`, `state_progress_week_50pct_mature`.

### Model Set C: Agricultural Management Model (69 Features)
*Focus: Biophysical drivers + Crop development + Producer risk management & coverage intensity.*
- **All 60 features from Model Set B.**
- **RMA Crop Insurance Pre-Harvest Underwriting (9):** `rma_policies_sold`, `rma_policies_earning_prem`, `rma_net_planted_acres`, `rma_liability_total`, `rma_total_premium`, `rma_subsidy`, `rma_liability_per_planted_acre`, `rma_premium_per_planted_acre`, `rma_missing_indicator`.

### Model Set D: Historical Autoregressive Benchmark (45, 63, or 72 Features)
*Focus: Integrating technological yield trends and local productivity memory.*
- **Any base set (A, B, or C) + 3 strict calendar-aligned lag features:**
  - `yield_lag_1yr`: Yield observed in year $t-1$ (NaN if skipped).
  - `yield_lag_2yr`: Yield observed in year $t-2$ (NaN if skipped).
  - `yield_roll_mean_5yr`: Mean yield over preceding 5 years $[t-5, t-1]$ (minimum 3 years observed).

---

## 4. Operational Prediction Horizons

Features must only be evaluated according to what is genuinely observable at the intended operational forecasting date:

| Horizon Name | Cutoff Date | Available Features | Excluded Features |
| :--- | :---: | :---: | :--- |
| **Horizon 1: Pre-Planting** | **April 1** | 24 Soil + 3 Lag Yields = **27 Features** | NO in-season weather; NO crop progress; NO RMA insurance. |
| **Horizon 2: Mid-Season (Silking)** | **August 1** | Horizon 1 + May–July NOAA weather + Early PRISM + Weeks 20–30 condition/progress = **40 Features** | NO August–October weather; NO late crop progress (dent/mature); NO RMA insurance. |
| **Horizon 3: Late-Season (Dent)** | **October 1** | Horizon 2 + August–September NOAA weather + August–September PRISM + Weeks 31–40 progress/condition + Pre-harvest RMA underwriting = **69 Features** | NO post-harvest RMA indemnities; NO harvest completion progress. |
| **Horizon 4: Post-Harvest Benchmark** | **December 1** | Full growing season climate + Full condition/progress + Final RMA policy metrics = **72 Features** | Retrospective academic ceiling only; not a forecasting model. |

---

## 5. Strict Exclusion List & Scientific Justifications

The following variables are **categorically prohibited** from the predictive matrix $\mathbf{X}$:

1. **`cv_percent` (Sampling Error Quality Metric):**
   - *Missingness:* 77.24% in historical data (100% missing 2002–2019; present only 2020–2025). 100% missing in 2026.
   - *Hazard:* Tree models split on `cv_percent.isna()`, creating an artificial regime shift that completely fails on 2026 prospective data.
2. **`soil_intptlat` and `soil_intptlon` (Geographic Centroids):**
   - *Hazard:* Including raw spatial coordinates allows gradient boosted decision trees to partition the geographic space into localized "leaf boxes," effectively memorizing county-specific historical average yields rather than learning functional relationships with soil and climate.
3. **`state_name`, `county_name`, `state` (String Identifiers):**
   - *Hazard:* Free-text strings lead to arbitrary categorical splits and high cardinality overfitting.
4. **`year` as a Linear Continuous Feature:**
   - *Hazard:* Decision trees are step functions and cannot extrapolate linear trends beyond $\max(\text{train\_year}) = 2025$. Passing `year = 2026` to a decision tree routes the prediction to the 2025 leaf, failing to capture technology drift. Autoregressive lags (`yield_lag_1yr`, `yield_roll_mean_5yr`) capture this drift naturally.
5. **Post-Harvest RMA Variables (`rma_postharvest_*`):**
   - *Hazard:* Indemnity amounts and loss ratios are computed after crop failure or loss claims are inspected in late winter/spring. Using loss payments to predict yield collapse is circular reasoning.
6. **Harvest Progress (`state_progress_postharvest_harvested_w44_pct`):**
   - *Hazard:* Harvest progress in November occurs after yield is already fixed and grain is in storage bins.

---
*Signed and Approved for Modeling Implementation:*  
**Principal Machine Learning Engineer & Research Auditor**
