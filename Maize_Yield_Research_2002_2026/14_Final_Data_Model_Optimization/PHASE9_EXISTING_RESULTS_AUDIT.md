# PHASE 9.1: FORENSIC PROJECT INVENTORY & ARTIFACT AUDIT REPORT

## Comprehensive Independent Examination of Project History & Certified Outputs
**Project Root:** `E:\ml do not open\Maize_Yield_Research_2002_2026`  
**Audit Timestamp:** November 2026  
**Auditor:** Principal ML Engineer, Agricultural Data Scientist & Research Auditor  
**Scope:** Recursive file inspection, cryptographic hash verification, schema dimension auditing, and reproducibility status.

---

### 1. Executive Inventory Summary

Across the 37 prioritized core artifacts from Phases 0 through 8:
- **Total Files Audited:** 37
- **Files Verified Existing & Non-Truncated:** 34 (91.9%)
- **Missing or Truncated Files:** 3 (0)
- **Certified Historical Preservation:** All source files in `10_Final_Pre_Modeling_Gate`, `11_Independent_Model_Validation`, `12_Final_Scientific_Certification`, and `13_Publication_Enhancement` remain intact, read-only, and strictly preserved.

---

### 2. Priority Phase 8 Artifact Verification Matrix

```
+-------------------------------------------------------+-------+-------+------------+--------------------+
| Artifact File Name                                    | Rows  | Cols  | Size (KB)  | Status             |
+=======================================================+=======+=======+============+====================+
| Enhanced_Corn_Belt_Historical_Features.csv            | 16,390| 98    | 11,002.8   | Verified Intact    |
| Enhanced_Corn_Belt_Future_2026_Features.csv           | 814   | 97    | 542.0      | Verified Intact    |
| ADVANCED_FEATURE_MANIFEST.csv                         | 15    | 6     | 4.1        | Verified Intact    |
| FEATURE_AVAILABILITY_AUDIT.csv                        | 5     | 7     | 2.9        | Verified Intact    |
| ADVANCED_FEATURE_ABLATION.csv                         | 10    | 7     | 0.8        | Verified Intact    |
| TREND_MODEL_COMPARISON.csv                            | 6     | 6     | 0.5        | Verified Intact    |
| GAM_MIXED_EFFECTS_RESULTS.csv                         | 2     | 6     | 0.2        | Verified Intact    |
| ENSEMBLE_MODEL_RESULTS.csv                            | 3     | 6     | 0.3        | Verified Intact    |
| FINAL_MODEL_COMPARISON.csv                            | 13    | 6     | 0.9        | Verified Intact    |
| FEATURE_IMPORTANCE_NEXTGEN.csv                        | 70    | 5     | 6.6        | Verified Intact    |
| ROLLING_ORIGIN_VALIDATION.csv                         | 7     | 11    | 0.6        | Verified Intact    |
| STATE_LEVEL_PERFORMANCE.csv                           | 9     | 10    | 0.9        | Verified Intact    |
| YEAR_LEVEL_PERFORMANCE.csv                            | 3     | 11    | 0.5        | Verified Intact    |
| UNCERTAINTY_COMPARISON.csv                            | 6     | 6     | 0.9        | Verified Intact    |
| PROSPECTIVE_2026_FORECAST_V2.csv                      | 814   | 7     | 88.4       | Verified Intact    |
| LITERATURE_REVIEW_MATRIX.csv                          | 32    | 22    | 30.1       | Verified Intact    |
+-------------------------------------------------------+-------+-------+------------+--------------------+
```

---

### 3. Key Forensic Observations

1. **Cohort Consistency Across Phases:**
   - The primary historical training/evaluation cohort consists of **16,390 rows** representing 788 agricultural counties across 24 years (2002–2025).
   - In Phase 8, the chronological split partitioned this into:
     - **Training Set (Years <= 2022):** 14,691 observations.
     - **Test Set (Years 2023–2025):** 1,699 observations.
2. **Reported vs Actual Metric Reproducibility:**
   - The reported Phase 8 Stacking RMSE of 14.84 bu/acre ($R^2 = 0.7459$) and LightGBM RMSE of 15.47 bu/acre ($R^2 = 0.7235$) are verified present in `FINAL_MODEL_COMPARISON.csv`.
   - In Phase 9.4, these models and their underlying out-of-fold base predictions will be independently re-run from scratch to certify exact numerical reproducibility.
3. **Data Integrity Flags for Phase 9.2 Audit:**
   - County FIPS codes are formatted as integers in several historical tables (e.g. `17001` vs `"17001"`). They must be standardized to 5-digit strings (`%05d`) across all Phase 9 deliverables.
   - Historical county reorganization (Shannon County SD, FIPS 46113, renamed to Oglala Lakota County, FIPS 46102 in May 2015) must be verified in the county eligibility register.
   - Cook County, IL (FIPS 17031) soil property imputation must be reviewed for physical consistency.
