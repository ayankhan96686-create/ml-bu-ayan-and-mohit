# Data Quality & Missingness Technical Audit Report

**Project Title:** Maize (Corn) Grain Yield Prediction Using Machine Learning (2002–2025)  
**Study Region:** 9 US Corn Belt States (814 Counties)  
**Temporal Domain:** 2002–2025 Labeled Observations; 2026 Unlabeled Prospective Set  
**Report Directory:** `E:/ml do not open/Maize_Yield_Research_2002_2026/06_Reports/`  

---

## 1. Executive Summary of Data Quality

This report evaluates the completeness, validity, consistency, and statistical distribution of the integrated datasets. 

### Key Data Quality Findings:
1. **Target Variable Authenticity:** 16,390 historical county grain yield observations across 788 counties and 24 years. Zero synthetic values. Exactly 0 nulls in the target column of `County_Year_Feature_Table.csv`.
2. **True Agricultural Zeroes:** 9 observations have `corn_yield_bu_acre == 0.0` in western South Dakota (Harding, Lawrence, Todd counties). These reflect genuine total crop failures and pasture conversions reported by NASS, and are intentionally preserved.
3. **Missing Value Profiles:** In `County_Year_Feature_Table.csv`, 43 of 79 features have 0.0% missingness (100% complete). Only `cv_percent` (77.2% missing by survey design), `state_progress_mature_w40_pct` (7.35% missing due to 2013 federal shutdown), and `prism_ppt_mm` (3.86% missing due to 2023 download omission) exceed 3% missingness.
4. **Duplicate Record Count:** Exactly **0 duplicate keys** exist across all integrated dataset tables.

---

## 2. Missing Value Audit in ML Feature Table (`County_Year_Feature_Table.csv`)

| Column Name | Domain | Missing Count | Missing (%) | Reason & Operational Impact |
|---|---|---|---|---|
| `cv_percent` | Quality Flag | 12,660 | 77.24% | Published only during Census of Agriculture years (every 5 yrs). Retained as quality variable; excluded from modeling. |
| `state_progress_mature_w40_pct` | Phenology | 1,204 | 7.35% | Week 40 was not published in 2013 due to the October 2013 US Federal Government shutdown. |
| `prism_ppt_mm` | Climate | 632 | 3.86% | Year 2023 precipitation raster was omitted during batch download. NOAA precipitation provides 100% complete backup. |
| `rma_net_planted_acres` | Insurance | 366 | 2.23% | Non-participating or urban/peri-urban counties with 0 commercial corn insurance policies (e.g. DuPage Co, IL). |
| `rma_liability_total` | Insurance | 366 | 2.23% | Non-participating or urban/peri-urban counties. |
| `rma_total_premium` | Insurance | 366 | 2.23% | Non-participating or urban/peri-urban counties. |
| `rma_subsidy` | Insurance | 366 | 2.23% | Non-participating or urban/peri-urban counties. |
| `rma_liability_per_planted_acre` | Insurance | 366 | 2.23% | Non-participating or urban/peri-urban counties. |
| `rma_premium_per_planted_acre` | Insurance | 366 | 2.23% | Non-participating or urban/peri-urban counties. |
| `state_progress_week_50pct_mature` | Phenology | 60 | 0.37% | Northern states in cool years where 50% maturity was reached after survey end. |
| Soil Profile Columns (26 cols) | Soil Physics | 11 | 0.07% | Cook County, IL (`17031`) centroid fell in Chicago urban / Lake Michigan shoreline mask. |
| All NOAA Climate Columns (14 cols) | Climate | 0 | 0.00% | 100% complete across all 16,390 county-years. |
| All Crop Condition Columns (9 cols) | Crop Health | 0 | 0.00% | 100% complete across all 16,390 county-years. |
| Target (`corn_yield_bu_acre`) | Target | 0 | 0.00% | 100% complete across all 16,390 county-years. |

---

## 3. Outlier & Range Validation

| Variable | Physical / Plausible Range | Observed Min | Observed Max | Observed Mean | Anomaly Assessment |
|---|---|---|---|---|---|
| `corn_yield_bu_acre` | 0 – 300 bu/acre | 0.0 | 253.6 | 157.45 | Verified authentic. Max in Macon Co, IL (2021); Min in SD (drought/forage). |
| `noaa_prcp_may_aug_in` | 2 – 50 inches | 3.12 | 38.64 | 16.78 | Verified. Severe drought in 2012 (3.12 in); historic floods in 2010/2019 (38.64 in). |
| `noaa_tmax_jul_f` | 65 – 115 °F | 73.8 | 98.6 | 84.82 | Verified. Peak heat observed in July 2012 across Missouri and Nebraska. |
| `soil_soc_mean_0_30cm` | 10 – 600 dg/kg | 46.0 | 386.0 | 185.34 | Verified. Highest organic matter in northern Iowa/Minnesota Mollisols. |
| `soil_ph_mean_0_30cm` | 40 – 90 (pH*10) | 48.0 | 79.0 | 64.21 | Verified. Neutral to slightly acidic typical for US Corn Belt. |
| `rma_net_planted_acres` | 0 – 600,000 acres | 10.0 | 464,120.0 | 92,450.0 | Verified. Matches commercial row-crop acreage distributions. |

---

## 4. Key Uniqueness & Referential Integrity

- Base Cartesian Spine: $814 \text{ counties} \times 25 \text{ years} = 20,350$ rows.
- Master Dataset Rows: $20,350$. Duplicate keys: $0$.
- Feature Table Rows: $16,390$. Duplicate keys: $0$.
- Future 2026 Table Rows: $814$. Duplicate keys: $0$.
- **Integrity Verdict:** **PASSED.** Complete mathematical key consistency across all tables.

