# Final Dataset Version Manifest & Cryptographic Ledger
## Maize/Corn Grain Yield Prediction (2002–2025)

**Project Root:** `E:/ml do not open/Maize_Yield_Research_2002_2026/`  
**Deliverable Directory:** [`10_Final_Pre_Modeling_Gate/`](file:///E:/ml%20do%20not%20open/Maize_Yield_Research_2002_2026/10_Final_Pre_Modeling_Gate/)  
**Status:** Authoritative Cryptographic Inventory  

---

## 1. Master Inventory & Cryptographic Ledger

Every dataset utilized in this research project is cryptographically tracked to guarantee scientific reproducibility and prevent silent data tampering:

| Dataset / Artifact File | Directory Location | Rows | Cols | File Size | SHA-256 Checksum | Pipeline Role |
| :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| **`County_Year_Master_Dataset.csv`** | `04_Integrated_Data/` | 20,350 | 83 | 9,811,436 B | `b1afee99f22708afb702763f78c3eb61c5a03915fce1fadd0a94dd6b57bf2e44` | Complete Cartesian Spine ($814 \text{ counties} \times 25 \text{ years}$) |
| **`County_Year_Feature_Table.csv`** | `04_Integrated_Data/` | 16,390 | 79 | 7,598,114 B | `4b8df5846ed38ec46f3e193f5b8f878149a614c2a032fc74b18c417f3b64a47f` | Raw Historical Labeled Dataset (2002–2025) |
| **`Future_2026_Feature_Table.csv`** | `04_Integrated_Data/` | 814 | 77 | 369,239 B | `d11a6daa32f056cdaa9fdd448032adcb45d632936b69b61ae074db110654879c` | Raw Prospective Forecast Table (2026) |
| **`Purified_Corn_Belt_Historical_Dataset.csv`** | `10_Final_Pre_Modeling_Gate/` | **16,390** | **83** | **8,006,235 B** | `0b0b45a7bffadf2c7b1e738ad0f7dded23b2fb1fe98e5a5b3ae723d1ffd65c51` | **Authoritative Modeling Dataset** (Lags engineered, `cv_percent` purged) |
| **`Purified_Corn_Belt_Future_2026_Dataset.csv`** | `10_Final_Pre_Modeling_Gate/` | **814** | **82** | **390,081 B** | `b69a3d5c09fb951304f05bc13dbdc19b33d70f01f97e01c287ab03a760511d7a` | **Authoritative 2026 Prediction Table** (Eligible counties tagged) |
| **`FINAL_COLUMN_MANIFEST.csv`** | `10_Final_Pre_Modeling_Gate/` | 87 | 13 | 15,737 B | `886ec93f7bb8b9b5b0da6d5c1e67c9297513d248abeb3203f18c8776f5b16aae` | Complete 87-column taxonomy & feature dictionary |
| **`FINAL_TEMPORAL_SPLIT_REPORT.csv`** | `10_Final_Pre_Modeling_Gate/` | 10 | 10 | 2,498 B | `df6295d3dbcd48f413a191b1670f62b73c131aa2fb0059ae77879ea775dfb9ec` | Reconciles Versions A, B, and C with empirical proof |
| **`FINAL_RMA_RECONCILIATION.csv`** | `10_Final_Pre_Modeling_Gate/` | 4 | 11 | 1,507 B | `49a660ffe8a167852543bbd23cfa5ce8706b7ee02839781872baae21a321dfa2` | Reconciles RMA rows, 1,581 nulls, and 3 div-by-zero rows |
| **`FINAL_WEATHER_AVAILABILITY.csv`** | `10_Final_Pre_Modeling_Gate/` | 19 | 11 | 5,366 B | `db95cd49f6162c43b8bab17475c8a60429be4ea4306ca51ee1ed0c96a1a68e12` | Weather metadata, PRISM 2023 gap, 2026 provisional M3 rasters |
| **`COUNTY_PREDICTION_ELIGIBILITY.csv`** | `10_Final_Pre_Modeling_Gate/` | 814 | 13 | 212,684 B | `add27511d3e94c224d0bd3699107680d93e9fa1115bb7321baf33064decbd30c` | 814 counties: 788 eligible, 26 excluded with agronomic justification |

---

## 2. Structural Schema Comparison

| Schema Dimension | `County_Year_Feature_Table.csv` (Raw) | `Purified_Corn_Belt_Historical_Dataset.csv` (Purified) | Operational Change |
| :--- | :---: | :---: | :--- |
| **Row Count** | 16,390 | 16,390 | Zero rows dropped. 100% labeled records retained. |
| **Column Count** | 79 | 83 | Net $+4$ columns ($+4$ engineered features, $-1$ purged `cv_percent`). |
| **`cv_percent`** | Present (77.24% missing) | **REMOVED** | Purged to prevent catastrophic temporal leakage and split artifacting. |
| **Engineered Lags** | None (Raw yields only) | **ADDED** | `yield_lag_1yr`, `yield_lag_2yr`, `yield_roll_mean_5yr` derived via strict calendar spine. |
| **RMA Missing Indicator** | None | **ADDED** | `rma_missing_indicator` (1 if no RMA policy record, 0 otherwise). |
| **County Scope Tag** | None | **ADDED** | `eligible_for_2026_forecast` (True for 788, False for 26 non-agricultural). |

---
*Verified and Certified:*  
**Data Integrity Auditor & Research Engineer**
