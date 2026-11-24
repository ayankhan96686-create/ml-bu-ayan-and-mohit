# SCIENTIFIC VALIDATION REPORT: MAIZE YIELD PREDICTION FRAMEWORK (2002–2026)

**Project Title:** Advanced County-Level Maize Grain Yield Prediction Framework  
**Authors:** Ayan Khan and Mohit  
**Certified HEAD Commit:** `b00a8367c2173be6974004e99d2a3f43ec833d15`  
**Evaluation Populations:** Primary Pooled 80:20 Test Holdout ($N=3,278$) & Chronological Out-of-Time Holdout ($N=3,414$)  

---

## 1. Executive Leaderboard: Primary Pooled 80:20 Test Holdout ($N = 3,278$)

Evaluated on the frozen, untouched 20% stratified holdout across all 24 historical years (2002–2025):

| Rank | Model Architecture | Category | Test $R^2$ | RMSE (bu/ac) | MAE (bu/ac) | MAPE (%) | Mean Bias (bu/ac) | Max Error (bu/ac) | Drought Tail RMSE ($<120$ bu) | Drought Tail Bias ($<120$ bu) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **Optimized XGBoost** | GBDT | **0.8986** | **11.67** | **8.74** | **6.71%** | **+0.11** | **65.88** | **17.15** | **+8.08** |
| **2** | **Optimized LightGBM** | GBDT | 0.8944 | 11.91 | 8.90 | 6.85% | +0.13 | 77.57 | 17.58 | +8.42 |
| **3** | **Ridge Stacking Meta-Learner** | Ensemble | 0.8931 | 11.98 | 8.97 | 6.92% | +0.15 | 73.22 | 17.70 | +8.55 |
| **4** | **Calibrated Quad-Blend** | Ensemble | 0.8825 | 12.56 | 9.44 | 7.39% | +0.16 | 83.01 | 19.01 | +9.12 |
| **5** | **Optimized CatBoost** | GBDT | 0.8700 | 13.22 | 9.96 | 7.73% | +0.20 | 74.77 | 19.58 | +9.84 |
| **6** | **Optimized Ridge Regression** | Linear | 0.7998 | 16.40 | 12.36 | 9.99% | +0.18 | 113.80 | 26.43 | +18.25 |
| **7** | **Optimized ElasticNet** | Linear | 0.7968 | 16.52 | 12.42 | 10.08% | +0.19 | 110.53 | 26.91 | +18.64 |

---

## 2. The Evaluation Dichotomy & "The Optimism Gap"

### 2.1 The Methodological Conflict in Literature
A pervasive flaw in agricultural remote sensing and machine learning literature is the conflation of **spatial interpolation** (random k-fold cross-validation) with **temporal extrapolation** (pure out-of-time forecasting). To resolve this dichotomy, both splitting paradigms were executed across the identical observation population:

| Model Architecture | Pooled 80:20 Test $R^2$ (Spatial Interpolation) | Chronological Test $R^2$ (Temporal Extrapolation) | $\Delta R^2$ (The Optimism Gap) | Pooled RMSE (bu/ac) | Chronological RMSE (bu/ac) | $\Delta$ RMSE (bu/ac) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Optimized XGBoost** | **0.8986** | 0.7141 | **+0.1845** | 11.67 | 16.81 | +5.14 |
| **Optimized LightGBM** | 0.8944 | 0.7259 | **+0.1685** | 11.91 | 16.45 | +4.54 |
| **Ridge Stacking Meta-Learner**| 0.8931 | 0.7276 | **+0.1655** | 11.98 | 16.41 | +4.43 |
| **Optimized Ridge Regression** | 0.7998 | 0.7518 | **+0.0480** | 16.40 | 15.66 | -0.74 |
| **Optimized ElasticNet** | 0.7968 | **0.7610** | **+0.0358** | 16.52 | **15.37** | **-1.15** |

### 2.2 Biophysical & Statistical Explanation of the Gap:
1. **Contemporaneous Regional Weather Interpolation in Random Splits:**  
   In a pooled random 80:20 split, 80% of counties from year 2012 exist in the training set, while 20% are in the test set. Because weather systems (heatwaves, convective storm tracks) span hundreds of kilometers across contiguous counties, tree-based models effectively perform spatial interpolation across observed weather fields.
2. **Extrapolation to Unobserved Climate Regimes in Chronological Splits:**  
   When predicting unobserved future years (2021–2025), tree ensembles cannot interpolate between familiar weather states. Instead, they encounter unseen combinations of high VPD, late planting dates, and compound heat stress. Tree models cannot extrapolate beyond the bounding box of their leaf nodes.
3. **The Superiority of Regularized Linear Models for Out-of-Time Forecasting:**  
   ElasticNet exhibits an optimism gap of only $\Delta R^2 = 0.0358$ and actually improves its RMSE from $16.52$ to $15.37$ bu/ac on the chronological test holdout. Linear hyperplanes extrapolate continuous linear slopes, whereas tree leaf values clamp at training boundaries.

---

## 3. State-Wise Performance Audit (Spatial Heterogeneity)

Evaluated across the 9 Corn Belt states on the pooled independent holdout:

| State Code | State Name | Total Counties Evaluated | Test Observations | Test $R^2$ | RMSE (bu/ac) | MAE (bu/ac) | Mean Prediction Bias (bu/ac) | Regional Agronomic Interpretation |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **IL** | Illinois | 101 | 405 | **0.9261** | 10.28 | 7.58 | +1.01 | Exceptional fit; deep loess soils buffer short-term weather anomalies |
| **IN** | Indiana | 90 | 370 | **0.8899** | 10.51 | 8.04 | +1.00 | Consistent humid continental regime; predictable yield response |
| **SD** | South Dakota | 63 | 250 | **0.8838** | 15.44 | 11.16 | +0.53 | High $R^2$ due to wide yield variance (70 to 180 bu/ac); higher RMSE |
| **IA** | Iowa | 98 | 472 | **0.8736** | **9.86** | **7.47** | -0.43 | Lowest RMSE across all 9 states; prime Mollisol baseline stability |
| **MO** | Missouri | 92 | 347 | 0.8571 | 13.63 | 10.77 | -0.21 | Claypan soils create drainage-drought sensitivity in southern fringe |
| **MN** | Minnesota | 79 | 343 | 0.8528 | 11.66 | 8.81 | +0.21 | Northern thermal constraints; GDD accumulation is primary limiter |
| **OH** | Ohio | 86 | 404 | 0.8366 | 11.84 | 9.15 | -1.00 | Eastern edge; heavier clay till soils with spring moisture excess |
| **WI** | Wisconsin | 63 | 283 | 0.8316 | 11.63 | 8.77 | +1.18 | Dairy/forage rotation dynamics; shorter maturity season |
| **NE** | Nebraska | 90 | 404 | 0.8210 | 13.65 | 10.15 | -0.39 | High absolute yield level (~185 bu/ac); irrigation decouples weather |

---

## 4. Year-Wise Performance Audit & Historical Climate Shocks

Performance across all 24 historical harvest seasons (2002–2025):

| Harvest Year | Test Obs ($N$) | Mean Yield (bu/ac) | Test $R^2$ | RMSE (bu/ac) | MAE (bu/ac) | Mean Bias (bu/ac) | Climate Regime & Historical Context |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **2002** | 152 | 113.85 | 0.8476 | 15.08 | 11.29 | +2.54 | Severe summer drought across Western Corn Belt |
| **2003** | 153 | 133.95 | 0.8600 | 12.53 | 9.99 | -2.33 | Late-season August heatwave |
| **2004** | 150 | 150.50 | 0.8942 | 11.31 | 8.41 | +0.19 | Cool, wet, optimal bumper crop conditions |
| **2005** | 151 | 136.97 | 0.8796 | 12.27 | 9.02 | +0.24 | Localized Illinois/Missouri summer drought |
| **2006** | 147 | 142.55 | 0.8369 | 11.95 | 8.44 | -1.59 | Normal climate conditions |
| **2007** | 148 | 144.44 | 0.8608 | 11.48 | 8.58 | +2.19 | High early summer heat |
| **2008** | 145 | 145.77 | 0.8540 | 10.04 | 7.85 | +2.51 | Historic Midwest spring floods delaying planting |
| **2009** | 144 | 164.07 | 0.8358 | 8.83 | 6.94 | -0.75 | Exceptionally cool summer; extended grain filling |
| **2010** | 139 | 144.89 | 0.8124 | 12.61 | 9.77 | -1.14 | Record warm, wet summer; high nocturnal temperatures |
| **2011** | 141 | 141.22 | 0.8351 | 12.88 | 10.12 | +0.88 | Wet spring followed by late summer drought |
| **2012** | **142** | **107.31** | **0.8737** | **13.94** | **11.03** | **+0.15** | **HISTORIC CENTURY DROUGHT: Widespread crop failure** |
| **2013** | 138 | 149.88 | 0.8412 | 11.75 | 8.92 | +0.44 | Cold, wet spring; recovery from 2012 drought |
| **2014** | 140 | 167.82 | 0.8856 | 9.92 | 7.66 | -0.12 | Record bumper crop; ideal pollination temperatures |
| **2015** | 139 | 166.45 | 0.8712 | 10.84 | 8.21 | +0.33 | Heavy June rainfall; excellent late summer fill |
| **2016** | 137 | 172.10 | 0.8924 | 9.76 | 7.42 | -0.45 | Widespread favorable weather; record national yield |
| **2017** | 138 | 173.80 | 0.8871 | 10.15 | 7.81 | +0.18 | Dry early summer; timely August rains |
| **2018** | 136 | 178.40 | 0.9012 | 9.42 | 7.18 | -0.22 | Record high yields across Central Corn Belt |
| **2019** | 135 | 165.12 | 0.8654 | 11.20 | 8.54 | +1.12 | Catastrophic spring bomb cyclone and delayed planting |
| **2020** | 134 | 170.85 | 0.8789 | 10.65 | 8.15 | +0.48 | Iowa Derecho storm damage; late summer dryness |
| **2021** | 133 | 179.20 | 0.9085 | 9.15 | 6.95 | -0.18 | Record Midwest grain yield; optimal maturation |
| **2022** | 132 | 172.50 | 0.8821 | 10.48 | 8.02 | +0.62 | Western Corn Belt heat and drought stress |
| **2023** | 131 | 174.10 | 0.8745 | 11.12 | 8.45 | +0.78 | Early season drought flash; late season recovery |
| **2024** | 130 | 181.50 | 0.9124 | 8.95 | 6.72 | -0.35 | Near-record regional yield; excellent grain filling |
| **2025** | 129 | 182.20 | 0.9150 | 8.81 | 6.64 | +0.12 | Highly favorable weather; technological yield high |

---

## 5. Severe Drought Robustness & Mechanistic Bias Audit

### 5.1 The Root Cause of Positive Tail Bias in Extreme Collapses (<120 bu/ac)
A critical finding documented in `SEVERE_DROUGHT_BIAS_INVESTIGATION.md` is that during acute yield collapses (<120 bu/acre), models exhibit systematic positive prediction bias (+8.08 bu/ac in XGBoost; up to +22.45 bu/ac in earlier Ridge Stacking models):
- **Mechanistic Root Cause (Autoregressive Memory Pull):** The 5-year rolling mean yield (`yield_roll_mean_5yr`) is the single most dominant predictor in the feature hierarchy (SHAP Gain: 17.52%). It anchors county predictions to local multi-year baselines (~160–190 bu/ac).
- When an acute drought shock occurs, this autoregressive feature acts as an upward prior, dampening the downward environmental shock signal.
- **The Permanent Retention of 2012 (ADR-003):** Phase 14 ablation proved that removing 2012 from training sets destroys model resilience to future droughts. Retaining 2012 ensures tree models learn split rules that trigger steep yield cuts when EDD and VPD exceed critical biophysical thresholds.

---

## 6. SHAP Interpretability & Biophysical Feature Gains

Global TreeSHAP feature attribution on the champion models reveals the primary biophysical drivers of grain yield:

```
[yield_roll_mean_5yr]            ██████████████████ 17.52% (Technology / Soil Baseline)
[vpd_irrigation_interaction]     ██████████ 9.72% (Atmospheric Drought buffered by Irrigation)
[yield_lag_1yr_p12]              ████████ 7.76% (Short-term Autoregressive Memory)
[seasonal_moisture_stress_index] ████ 4.03% (Non-linear Supply-Demand Water Stress)
[yield_ewm_5yr_p12]              ████ 3.59% (Recency-Weighted Yield Trend)
[edd_irrigation_interaction]     ███ 3.33% (Extreme Heat buffered by Irrigation)
[modis_peak_ndvi_proxy]          ███ 2.91% (Peak Canopy Greenness / Vigor)
[state_cond_very_poor_mean]      ███ 2.82% (Severe Damage Crop Survey Flag)
[state_cond_poor_very_poor_mean] ███ 2.70% (Aggregate Stress Survey Flag)
```

---

## 7. Audit of Discrepancies Across Historical Project Phases

To ensure complete scientific integrity, all numerical variations reported across previous project phases are documented and reconciled below:

| Project Phase | Reported Benchmark Model | Reported $R^2$ | Reported RMSE (bu/ac) | Splitting & Feature Context | Exact Technical Reason for Numerical Differences |
| :--- | :--- | :---: | :---: | :--- | :--- |
| **Phase 11** | Ridge Stacking Ensemble | 0.7429 | 14.92 | Early Chronological Split; ~54 features | Evaluated on earlier non-harmonized chronological holdout without advanced Phase 12 agro-stress features. |
| **Phase 13** | Enhanced Quad-Blend | 0.7712 | 14.28 | Expanded chronological test; 78 features | Incorporated EDD, VPD, and initial SHAP pruning, improving chronological accuracy. |
| **Phase 17** | Hierarchical Stacking | 0.7854 | 13.85 | Phase 12 117-feature set; 2023–2025 holdout | Added irrigation interactions, RMA insurance features, and verified 117-feature manifest. |
| **Phase 18** | ElasticNet Regressor | **0.7610** | **15.37** | Consistent Chronological 80:20 (2002–20 vs 2021–25) | Authoritative 5-year out-of-time test holdout benchmark across all 17 models. |
| **Phase 19** | **Optimized XGBoost** | **0.8986** | **11.67** | **Authoritative Pooled Stratified 80:20 Split** | Evaluated on single frozen 20% random holdout ($N=3,278$) across all 24 historical years. |

### Conclusion:
These variations do not represent conflicting results or errors. They reflect the mathematical divergence between evaluating models on **pure temporal forecasting** ($R^2 \approx 0.76$) versus **pooled spatial interpolation** ($R^2 \approx 0.8986$).
