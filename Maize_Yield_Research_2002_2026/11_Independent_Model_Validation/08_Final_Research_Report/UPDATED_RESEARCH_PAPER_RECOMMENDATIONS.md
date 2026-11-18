# UPDATED RESEARCH PAPER RECOMMENDATIONS: SCIENTIFIC JOURNAL MANUSCRIPT STRATEGY

**Project:** County-Level Maize Grain Yield Prediction Research Project (2002–2026)  
**Author:** Senior Agricultural Data Scientist & Machine Learning Research Reviewer  
**Target Submission Period:** 2026/2027 Academic Year  

---

## 1. Recommended Target Journals

Based on the interdisciplinary focus on machine learning, remote sensing, soil science, and agro-climatology:

1. **Primary Target (Tier 1 Applied AI / Agriculture):**
   - **Computers and Electronics in Agriculture** (*Elsevier*, Impact Factor $\approx 7.7$)
   - *Fit:* Perfect alignment with applied machine learning, large-scale spatial-temporal modeling, open-source agricultural pipelines, and interpretable AI.
2. **Alternative Target A (Tier 1 Agro-Climatology):**
   - **Agricultural and Forest Meteorology** (*Elsevier*, Impact Factor $\approx 5.6$)
   - *Fit:* Strong emphasis on the biophysical heat-stress cliff (86°F July threshold), drought interactions, and atmospheric evaporative demand.
3. **Alternative Target B (Tier 1 Agronomy & Crop Modeling):**
   - **Field Crops Research** (*Elsevier*, Impact Factor $\approx 5.8$)
   - *Fit:* High interest in county-level yield gap analysis, irrigation decoupling in Nebraska, and acreage-weighted aggregation methodology.

---

## 2. Proposed Working Title and Abstract Outline

### Proposed Title:
> *"Disentangling Historical Memory and Biophysical Drivers in County-Level Maize Yield Prediction: A Rigorous Multi-Decadal Benchmark Across the US Corn Belt (2002–2026)"*

### Abstract Structure:
1. **Context & Motivation:** Accurate county-level crop yield forecasting is vital for food security and commodity market stability. However, existing literature frequently conflates long-term technological trends with in-season climate sensitivity, leaving the true incremental value of biophysical features unresolved.
2. **Methodology:** We construct a leak-free, 24-year spatial-temporal dataset covering 788 counties across 9 US Corn Belt states (16,390 historical observations). We conduct a controlled ablation benchmark isolating historical yield memory from 37 high-resolution soil and climate variables, followed by Leave-One-State-Out spatial cross-validation and conformal prediction calibration.
3. **Key Findings:**
   - Environmental features provide a statistically significant **18.06% RMSE reduction** ($p = 9.85 \times 10^{-24}$) and a **+$0.1771$ gain in $R^2$** beyond historical memory alone.
   - July maximum temperature (`noaa_tmax_jul_f`) is the dominant biophysical driver, exhibiting a sharp non-linear yield decline starting at **86.0°F ($30.0°C$)**.
   - Spatial transfer reveals a profound **irrigation decoupling** in Nebraska, where pivot irrigation from the Ogallala Aquifer insulates crops from thermal stress, causing models trained on rainfed states to underpredict by 28.6 bu/acre.
   - Acreage-weighted state aggregation resolves a **$11.83$ bu/acre (7.09%) distortion** caused by naive county averaging, representing a 728-million-bushel swing across the Corn Belt.
4. **Significance:** This work establishes a reproducible benchmark and decision framework for operational agricultural forecasting.

---

## 3. Methodological Guidance: Validated Claims vs. Cautions

### Claims Validated with Strong Empirical Proof:
- **Incremental Value of Climate and Soil:** Proved via paired block bootstrap (500 resamples, 95% CI $[3.24, 4.48]$ bu/acre RMSE improvement).
- **The 86°F July Thermal Stress Cliff:** Validated via TreeSHAP non-linear response curves across 1,699 test points.
- **The Acreage Aggregation Bias:** Quantified across 61.54M planted acres, proving why weighted aggregation is mathematically essential.
- **Conformal Interval Coverage:** Validated on completely unseen test years (89.5% empirical coverage for nominal 90% confidence).

### Cautions & Guardrails to Observe in Manuscript:
- **Avoid Overclaiming Causality on Irrigation:** State clearly that irrigation decoupling is an empirical, agronomic hypothesis supported by regional data (USGS/USDA), but our model does not explicitly observe field-level water application. Frame this as a limitation and future research direction.
- **Explain Cook County, IL Transparently:** Disclose that Cook County's urban lakefront centroid caused missing ISRIC SoilGrids data, and justify why tree models handle this natively while linear models require median imputation.
- **Distinguish Baseline Memory from Trend:** Emphasize that rolling 5-year averages act as a baseline productivity proxy, and future research could incorporate explicit parametric genetic-gain trends (+1.8 bu/acre/yr).

---

## 4. Required Manuscript Figures and Tables

The manuscript should directly utilize the artifacts generated in Phase 5:

1. **Figure 1 (Study Area Map & Pipeline Architecture):** Conceptual schematic illustrating the 9 Corn Belt states, 788 eligible counties, and temporal blocking ($t \le 2019$, $2020–2022$, $2023–2025$, $2026$).
2. **Figure 2 (Ablation Benchmark Barplot):** Visualizing Model A, B, C, and D test RMSE and $R^2$ with bootstrap error bars (`CONTROLLED_ABLATION_BENCHMARK.csv`).
3. **Figure 3 (TreeSHAP Global Summary):** Embedding `11_Independent_Model_Validation/06_Interpretability/SHAP_GLOBAL_IMPORTANCE.png`.
4. **Figure 4 (Non-Linear Biophysical Responses):** Embedding `11_Independent_Model_Validation/06_Interpretability/SHAP_DEPENDENCE_ANALYSIS.png` showing the 86°F July TMax inflection point and August precipitation saturation.
5. **Figure 5 (Spatial LOSO Error Map):** Choropleth map illustrating state-level biases under Scenario 1 vs Scenario 2, highlighting the Nebraska anomaly (`CORRECTED_LOSO_RESULTS.csv`).
6. **Figure 6 (State Aggregation Discrepancy):** Bar comparison of Unweighted vs. Acreage-Weighted State Yields (`STATE_AGGREGATION_COMPARISON.csv`).
7. **Table 1 (Comprehensive Model Comparison):** Presenting the 11-algorithm benchmark leaderboard (`ADVANCED_ML_MODEL_COMPARISON.csv`).
8. **Table 2 (Conformal Prediction Calibration):** Documenting empirical coverage across 80%, 90%, and 95% nominal levels (`PREDICTION_INTERVAL_VALIDATION.csv`).

