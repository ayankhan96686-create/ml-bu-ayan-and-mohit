# PUBLICATION EVIDENCE AND RESEARCH REPRODUCIBILITY CHECKLIST

## Comprehensive Verification Manifest for Peer Review and Academic Certification
**Project:** County-Level Maize/Corn Grain Yield Prediction (2002–2026)  
**Output Directory:** `15_Final_Confirmatory_Validation/07_Manuscript/`  
**Certified Cohort:** Out-of-Time Prospective Panel (2023–2025, $N = 1,699$)  
**Date:** November 2026

---

### 1. Research Protocol and Data Integrity Checklist

| Invariant / Audit Standard | Requirement Description | Verification Evidence | Audit Status |
| :--- | :--- | :--- | :---: |
| **No Data Leakage** | Zero post-August covariates used in yield prediction | Verified in `run_task_g_final_models.py` line 31–53: All 69 predictors observable by Aug 31 | **PASS** |
| **Historical Calendar Spine** | Historical lags generated on complete calendar spine prior to any modeling | Verified: 2002–2025 contiguous panel; zero dropped years in feature table | **PASS** |
| **2012 Drought Retention** | Genuine 2012 drought data preserved in authoritative dataset | Verified: All 711 county records in 2012 retained; zero rows pruned from historical panel | **PASS** |
| **Exact FIPS Identification** | 5-digit zero-padded string FIPS formatting | Verified: `%05d` string encoding applied across all historical, test, and forecast tables | **PASS** |
| **Target Min/Max Validity** | Target yield range physiologically grounded | Verified: Minimum 0.0 bu/ac (9 valid disaster zero-yields in SD), Maximum 253.6 bu/ac | **PASS** |
| **Identical Test Cohort** | All candidate models evaluated on exact same test partition | Verified: Strictly identical 1,699 rows ($2023–2025$) evaluated across all 13 models | **PASS** |
| **Mathematical Metric Invariant** | Exact formula $R^2 = 1 - \text{SSE}/\text{SST}$ with unweighted test cohort mean | Verified: Formula confirmed in Python metric calculation functions | **PASS** |
| **Deterministic Seeds** | Master random seed pinned to 42 across all training routines | Verified: `random_state=42`, `random_seed=42` pinned in all scripts | **PASS** |
| **Non-Destructive Execution** | Zero modifications to prior certified directories (`10_`, `11_`, `12_`, `13_`, `14_`) | Verified: All Phase 10 artifacts written strictly into `15_Final_Confirmatory_Validation/` | **PASS** |

---

### 2. Discrepancy Resolution Checklist

| Item | Prior Discrepancy | Forensic Audit Resolution | Evidence File |
| :--- | :--- | :--- | :--- |
| **2012 Annual Mean** | 108.9 vs 111.75 bu/acre | Discovered typographical string in `perform_eda.py` Line 124 annotation. Authoritative unweighted county mean is **111.75 bu/ac**; acreage-weighted yield is **116.71 bu/ac**. | `01_Independent_Audit/2012_ANNUAL_AGGREGATION_RECONCILIATION.csv` |
| **Model Champion** | Ridge Stacking ($R^2=0.7429$) vs Quad-Tree Blend ($R^2=0.7221$) | Reconciled: Ridge Stacking is #1 Overall Champion (RMSE 14.92 bu/ac); Quad-Tree Blend is #1 Pure-Tree Drought Champion (Severe Tail RMSE 29.62 bu/ac). Both evaluated on identical $N=1,699$ test cohort. | `01_Independent_Audit/MODEL_CHAMPION_RECONCILIATION.csv` |
| **Drought Tail Penalty** | Assertion of "+1.98 bu/ac" penalty | Conflation resolved: $+1.9821$ bu/acre is the exact LightGBM severe tail penalty from excluding 2012 training observations ($31.3643 - 29.3822 = 1.9821$). | `01_Independent_Audit/DROUGHT_FEATURE_ABLATION_RECONCILIATION.csv` |
| **Robust Loss Functions** | Early report that Huber/Fair loss "failed" | Proved mathematical failure of default unscaled parameters ($c=1.0$ vanishing Hessian). Properly scaled Huber/Fair loss ($c=20.0$) achieves $R^2 \approx 0.716$, but L2 remains optimal for tail risk. | `02_Loss_Function_Validation/LOSS_FUNCTION_SELECTION_REPORT.md` |
| **Residual Correction** | Proposal to fit secondary model on residuals | Strict 5-fold OOF training residual experiments proved that second-stage models overfit residual noise and degrade test variance. Ridge Stacking directly on predictions remains superior. | `04_Residual_Optimization/RESIDUAL_CORRECTION_REPORT.md` |

---

### 3. Certification Sign-Off

The entire analytical package has been independently executed, cross-checked against underlying datasets, and certified as publication-ready.

