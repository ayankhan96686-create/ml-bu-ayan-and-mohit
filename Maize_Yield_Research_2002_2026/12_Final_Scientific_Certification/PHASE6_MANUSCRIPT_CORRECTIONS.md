# PHASE 6 MANUSCRIPT CORRECTIONS AND REVISION DIRECTIVES

**Project:** County-Level Maize Grain Yield Prediction Research Project (2002–2026)  
**Intended Target Manuscripts:** Applied Machine Learning & Agronomy Submissions  
**Auditor:** Principal Academic Research Reviewer & Agricultural Data Scientist  

---

## 1. Executive Summary & Purpose

This document provides a line-item inventory of corrections, adjustments, and language modifications required to transition the project from internal technical reports into a peer-reviewed academic journal article.

Every revision is designed to eliminate vulnerable claims, replace over-optimistic statistics with cluster-robust metrics, correct mathematical discrepancies, and align agricultural terms with international standards.

---

## 2. Line-Item Manuscript Corrections

### Correction 1: Resampling Uncertainty and p-values
- **Original Wording in Phase 5 Draft:**  
  *"The incremental value of environmental features was proven with p = 9.85e-24 (paired t-test) and p = 2.92e-31 (Wilcoxon test)."*
- **Revised Manuscript Text:**  
  *"Environmental features provided a statistically significant improvement over historical yield memory alone, reducing out-of-sample RMSE from 21.60 to 17.70 bu/acre (-18.06%; Delta RMSE = 3.90 bu/acre). Under a conservative two-way block bootstrap clustering by state and year (2,000 replicates), the 95% Confidence Interval for the RMSE reduction was [2.44, 5.37] bu/acre, confirming robust predictive gains after accounting for spatial and temporal panel dependence."*
- **Rationale:** Standard $i.i.d.$ paired $t$-tests assume independent errors, which is violated in spatial panels. Cluster-robust bootstrap intervals provide a defensible statistical foundation.

---

### Correction 2: Correction of Spatial LOSO Average R²
- **Original Wording in Phase 5 Draft:**  
  *"Average spatial holdout R² across the 9 states is 0.6879 in Scenario 1 and 0.3703 in Scenario 2."*
- **Revised Manuscript Text:**  
  *"In Leave-One-State-Out spatial cross-validation, the macro-average state R² was 0.6434 across all 9 states in Scenario 1 (pooled total R² = 0.6499). Excluding Nebraska, which exhibits distinct irrigation dynamics, the average state R² reached 0.7018. In Scenario 2 (cold-start environmental transfer), the macro-average state R² was 0.3259 (0.4783 excluding Nebraska)."*
- **Rationale:** Corrects the arithmetic discrepancy ($0.6879 \to 0.6434$ and $0.3703 \to 0.3259$) and explicitly documents the impact of Nebraska.

---

### Correction 3: Scope of Geographic Validation (Spatial vs. Spatiotemporal)
- **Original Wording in Phase 5 Draft:**  
  *"Leave-One-State-Out spatial cross-validation demonstrates prospective forecasting generalization."*
- **Revised Manuscript Text:**  
  *"Leave-One-State-Out (LOSO) cross-validation evaluates purely spatial generalization across geographic boundaries under the assumption that multi-year regional training data are available. Because training folds contain concurrent and future calendar years from neighboring states, LOSO is not a prospective space-time forecasting test. Prospective forecasting ability is evaluated strictly on the out-of-sample 2023–2025 temporal holdout."*
- **Rationale:** Prevents peer reviewers from critiquing LOSO as a pseudo-forecast.

---

### Correction 4: Agronomic Caution on Nebraska Irrigation
- **Original Wording in Phase 5 Draft:**  
  *"Nebraska cold-start failure is caused by Ogallala aquifer irrigation decoupling."*
- **Revised Manuscript Text:**  
  *"The substantial negative bias observed in Nebraska under cold-start transfer (-28.56 bu/acre) strongly aligns with regional irrigation patterns. According to USDA Census and USGS data, over 62% of Nebraska corn acreage is irrigated from the High Plains (Ogallala) Aquifer, decoupling crop productivity from summer rainfall deficits. However, because field-level water application was not directly observed in the feature matrix, this irrigation decoupling represents an empirically supported agronomic explanation rather than a directly proved causal mechanism."*
- **Rationale:** Prevents causal overreach and invites reviewers to appreciate the author's nuanced agronomic knowledge.

---

### Correction 5: Harvested vs. Planted Acreage & Grain Production
- **Original Wording in Phase 5 Draft:**  
  *"Projected 2026 Corn Belt grain production is 10,987.8 million bushels across 61.54M planted acres."*
- **Revised Manuscript Text:**  
  *"Applying our prospective 2026 acreage-weighted yield estimate (178.48 bu/harvested acre) across the 59.39 million estimated harvested grain acres projects total Corn Belt maize grain production of 10,600.0 million bushels. Using total planted acreage (61.54 million acres) directly as a multiplier overestimates grain output by 387.8 million bushels due to the exclusion of corn harvested as silage—most notably in Wisconsin, where approximately 23.5% of planted acreage is dedicated to livestock silage rather than grain."*
- **Rationale:** Directly satisfies agricultural economics conventions (USDA NASS yield is strictly defined per *harvested* acre).

---

### Correction 6: Prediction Intervals vs. Confidence Intervals
- **Original Wording in Phase 5 Draft:**  
  *"Conformal confidence intervals provide distribution-free guarantees on 2026 forecasts."*
- **Revised Manuscript Text:**  
  *"We construct conformal prediction intervals for future county observations. While empirical coverage on the 2023–2025 test set closely matches nominal targets (78.40% for 80%, 89.46% for 90%, and 94.94% for 95%), year-by-year evaluation indicates that macro-climatic shifts induce modest fluctuations in empirical coverage (e.g., 86.95% in 2025 for the nominal 90% interval). These intervals represent calibrated empirical prediction bounds rather than exact distribution-free guarantees under temporal climate drift."*
- **Rationale:** Technical precision: intervals for random observations are *prediction intervals*; guarantees require exchangeability.

---

### Correction 7: Technology Trend Decomposition
- **Original Wording in Phase 5 Draft:**  
  *"Historical yields increase at ~+1.8 bu/acre/year due to hybrid genetic breeding."*
- **Revised Manuscript Text:**  
  *"Linear regression across the 24-year study period (2002–2025) indicates a regional productivity gain of +2.45 bu/acre/year (95% CI: [1.73, 3.17] bu/ac/yr; R² = 0.6926). This historical drift reflects a combined technological trajectory—encompassing commercial hybrid breeding, increased planting density, precision fertilizer application, improved field drainage, and management enhancements—which is effectively captured by historical rolling yield features."*
- **Rationale:** Avoids attributing multi-faceted agricultural yield gains to genetics alone.

---

### Correction 8: July Thermal Cliff and Compound Stress
- **Original Wording in Phase 5 Draft:**  
  *"July maximum temperature causes severe physiological crop loss above 86°F."*
- **Revised Manuscript Text:**  
  *"TreeSHAP analysis identified an inflection threshold at 86.0°F (30.0°C) for July maximum temperature, beyond which marginal yield attributions decline steeply (-3.5 to -5.0 bu/acre per degree). In observational climate data, high summer temperatures exhibit moderate negative correlation with rainfall (r = -0.25 to -0.45). Consequently, this inflection reflects a compound atmospheric heat and vapor pressure deficit stress threshold during the critical maize silking and pollination window."*
- **Rationale:** Grounds machine learning interpretability in established agro-meteorological principles.

---

### Correction 9: Multi-Model Competition and Ensemble Recommendation
- **Original Wording in Phase 5 Draft:**  
  *"HistGradientBoosting is declared the best model on the test set."*
- **Revised Manuscript Text:**  
  *"On the 2023–2025 holdout set, HistGradientBoosting (17.24 bu/acre), MLPRegressor (17.26 bu/acre), and LightGBM (17.33 bu/acre) demonstrated superior performance over baseline XGBoost (17.70 bu/acre). However, paired block bootstrap tests indicate that the top three algorithms are statistically indistinguishable within sampling noise (p > 0.35; difference 95% CIs encompass zero). To leverage both the non-linear interaction splitting of gradient-boosted trees and the continuous boundary regularisation of neural networks, we recommend an equal-weighted ensemble for operational forecasting."*
- **Rationale:** Demonstrates scientific maturity and avoids overfitting to minor test-set differences.

---

## 3. Summary of Master Equations for Manuscript

1. **Acreage-Weighted State Yield:**
   $$\bar{Y}_s = \frac{\sum_{i \in \mathcal{C}_s} \hat{y}_i \times A_{i}^{\text{harvested}}}{\sum_{i \in \mathcal{C}_s} A_{i}^{\text{harvested}}}$$
2. **Total Grain Production:**
   $$P_{\text{total}} = \sum_{i=1}^{N_{\text{eligible}}} \hat{y}_i \times A_{i}^{\text{harvested}}$$
3. **Conformal Prediction Interval:**
   $$\hat{\mathcal{C}}(X) = \left[ \hat{y}(X) - q_{1-\alpha}(|y - \hat{y}|_{\text{calib}}), \quad \hat{y}(X) + q_{1-\alpha}(|y - \hat{y}|_{\text{calib}}) \right]$$
4. **Two-Way Block Bootstrap Test:**
   $$\Delta \text{RMSE}_b = \text{RMSE}_b(\text{Model B}) - \text{RMSE}_b(\text{Model D}), \quad b \in \{1, \dots, 2000\}$$

