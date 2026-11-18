# FINAL MANUSCRIPT EDITORIAL AUDIT REPORT

## Rigorous Verification of Mathematical Notation, Metric Consistency, Units, and Citations
**Project:** County-Level Maize Grain Yield Prediction (2002–2026)  
**Output Directory:** `16_Final_Publication_Package/09_Final_Audit/`  
**Date:** November 2026

---

### 1. Verification of Metric Consistency Across Document Sections

We cross-checked reported metrics between the Abstract, Methodology, Results Tables, Discussion, and Conclusion:
- **Ridge Stacking Overall Champion:** RMSE = 14.9206 bu/ac, MAE = 11.2845 bu/ac, R² = 0.7429, Bias = +0.2418 bu/ac. *(Consistently reported across all sections).*
- **Calibrated Quad-Tree Blend:** RMSE = 15.5129 bu/ac, MAE = 11.8243 bu/ac, R² = 0.7221, Bias = -2.7972 bu/ac, Severe Tail RMSE (<120 bu/ac) = 29.6209 bu/ac. *(Consistently reported across all sections).*
- **2012 Severe Tail Penalty:** With 2012 = 29.38 bu/ac; Without 2012 = 31.36 bu/ac (Penalty = +1.9821 bu/ac). *(Exact match).*
- **Acreage Aggregation (2026):** Unweighted county mean = 175.49 bu/ac; Harvested-grain-acre weighted = 181.12 bu/ac; Projected total production = 12.38 billion bushels across 68.36 million acres. *(Exact match).*

---

### 2. Editorial and Scientific Terminology Guardrails

- **No "Percentage Accuracy" Misnomers:** R² is formally defined as the *coefficient of determination* ($1 - \text{SSE}/\text{SST}$); never described as "percentage accuracy".
- **Prediction vs. Confidence Intervals:** Individual county bounds ($\pm 23.42$ bu/ac) are correctly designated as *conformal prediction intervals*, while regional aggregate mean bounds ($\pm 6.84$ bu/ac) are designated as *aggregate forecast bounds*.
- **Harvested Grain vs. Planted Acres:** Distinct state weights are explicitly reported for harvested grain acreage (68.36M acres) versus planted footprint (75.15M acres).
- **Association vs. Causality:** SHAP attributions are explicitly described as predictive feature importance within tree structures, not mechanistic physiological proof.
