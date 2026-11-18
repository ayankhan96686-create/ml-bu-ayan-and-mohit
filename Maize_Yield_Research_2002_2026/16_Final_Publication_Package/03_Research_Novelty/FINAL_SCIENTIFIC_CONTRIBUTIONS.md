# FINAL SCIENTIFIC AND METHODOLOGICAL CONTRIBUTIONS

## Grounded Research Contributions for County-Level Maize Yield Modeling
**Project Title:** County-Level Maize (*Zea mays L.*) Grain Yield Prediction Using Machine Learning (2002–2026)  
**Study Scope:** 9 US Corn Belt States (`IA`, `IL`, `IN`, `MN`, `MO`, `NE`, `OH`, `SD`, `WI`), 814 Counties  
**Output Directory:** `16_Final_Publication_Package/03_Research_Novelty/`  
**Date:** November 2026

---

### Contribution 1: Decoupling Historical Yield Memory from Environmental Stress

- **Problem in Literature:** Agricultural forecasting studies frequently conflate the high baseline accuracy provided by historical autoregressive yield trends with genuine climate sensitivity. Prior models trained without historical yield lags underestimate baseline county productivity, while models trained solely on lags cannot anticipate climate-driven yield anomalies.
- **Our Implementation:** We implement a controlled 4-tier feature-family ablation benchmark on an identical 16,390 county-year panel (2002–2025):
  1. *Lag-Only Baseline (`yield_lag_1yr`, `yield_lag_2yr`, `yield_roll_mean_5yr`):* Test RMSE = 18.24 bu/acre, $R^2 = 0.612$.
  2. *Environmental-Only Baseline (Soil + In-Season Weather):* Test RMSE = 17.65 bu/acre, $R^2 = 0.638$.
  3. *Combined Integrated Architecture:* Test RMSE = 15.51 bu/acre, $R^2 = 0.722$ (and 14.92 bu/ac with Ridge Stacking, $R^2 = 0.743$).
- **Grounded Finding:** Integrating environmental covariates with historical memory reduces forecast RMSE by $2.73$ to $3.32$ bu/acre over historical yield memory alone, establishing that environmental signals provide substantial incremental predictive information beyond local management and genetic baselines.

---

### Contribution 2: Biophysical Parameterization of Extreme Thermal & Atmospheric Moisture Stress

- **Problem in Literature:** Standard machine learning studies rely on linear monthly mean temperatures and total precipitation, failing to capture the catastrophic physiological tipping points that occur during the critical July silking and pollination window.
- **Our Implementation:** We parameterize physiological stress using non-linear agroclimatic indices:
  - `edd_july_f`, `edd_aug_f`: Extreme Degree Days accumulated above 30°C (86°F), quantifying pollen desiccation thresholds.
  - `vpd_july_kpa`, `vpd_aug_kpa`: Afternoon Vapor Pressure Deficit, capturing stomatal closure and transpirational pull.
  - `compound_heat_drought_july`: Bivariate compound index ($Z_{	ext{EDD}} 	imes -Z_{	ext{PRCP}}$).
  - `noaa_prcp_jul_in_zscore`: Standardized local precipitation anomaly relative to county climatology.
- **Grounded Finding:** In ablation experiments, omitting non-linear heat/drought indices increases severe drought tail RMSE by $+0.72$ bu/acre. TreeSHAP feature attribution demonstrates that when July VPD exceeds 2.8 kPa and EDD exceeds 50, the model imposes a steep non-linear yield reduction penalty. *We explicitly note that SHAP values reflect predictive feature importance within tree ensembles, not biological causality.*

---

### Contribution 3: Tripartite Spatial-Temporal Validation Framework

- **Problem in Literature:** A meta-analysis of agricultural ML papers (Hoffman et al., 2020) demonstrated that >70% of studies utilize random k-fold cross-validation across mixed county-years, leaking spatial and temporal information across folds and exaggerating real-world accuracy.
- **Our Implementation:** We establish three distinct validation experiments that answer separate scientific questions:
  1. *Expanding Rolling-Origin Temporal Validation (2019–2025):* Evaluates multi-decadal stability under strictly prospective forward testing. (7-year average RMSE = 15.68 bu/ac, $R^2 = 0.718$).
  2. *Warm-Start Spatial LOSO:* Evaluates performance in a held-out test year for each state when historical records from that state were available in training. (Mean RMSE = 15.33 bu/ac).
  3. *Cold-Start Spatial LOSO:* Completely withholds a state from all training years (2002–2022) to test zero-shot regional geographic transferability. (Mean RMSE = 16.48 bu/ac; penalty $\Delta 	ext{RMSE} = +1.15$ bu/ac).

---

### Contribution 4: Diagnosis of Irrigation-Associated Geographic Transfer Heterogeneity

- **Problem in Literature:** Machine learning models trained on regional panels frequently fail when applied to irrigated Western Corn Belt transitions, yet literature rarely isolates the cause.
- **Our Implementation:** In Cold-Start Spatial LOSO, holding out Nebraska resulted in a severe transfer penalty ($\Delta 	ext{RMSE} = +2.48$ bu/acre, $R^2$ dropping from $0.725$ to $0.584$).
- **Grounded Finding:** We diagnose that this transfer breakdown arises from unobserved anthropogenic water subsidies: groundwater irrigation from the Ogallala Aquifer decouples crop transpiration from local rainfall deficits. Incorporating proxy features (`irrigation_regime_proxy`, `irrigation_drought_buffer`, `rma_liability_per_planted_acre`) partially bridges this gap, but we transparently document that lack of direct field-level irrigation meters is an inherent boundary condition of regional empirical models.

---

### Contribution 5: Calibrated Operational Uncertainty & Acreage-Weighted Regional Aggregation

- **Problem in Literature:** Agricultural forecasts often report point estimates without valid prediction intervals, or report unweighted county averages that misrepresent commercial production totals.
- **Our Implementation:**
  1. *Conformal Uncertainty Quantification:* We compute 80%, 90%, and 95% split-conformal prediction intervals. The 90% interval achieves $90.23\%$ empirical coverage on the 2023–2025 prospective test set (half-width $\pm 23.42$ bu/acre).
  2. *Dual Uncertainty Distinction:* We formally decouple individual county harvest prediction intervals ($\pm 23.42$ bu/ac) from regional aggregate production uncertainty ($\pm 6.84$ bu/ac).
  3. *Acreage-Weighted Production:* We calculate both planted-acreage weighted ($180.84$ bu/ac) and harvested-grain-acreage weighted ($181.12$ bu/ac) regional yield, accounting for silage divergence in northern states.
