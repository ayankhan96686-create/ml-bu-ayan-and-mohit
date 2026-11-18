# FINAL SCIENTIFIC RECOMMENDATION: 2012 DATA RETENTION & DROUGHT ROBUSTNESS

## Definitive Conclusions on Extreme Event Preservation, Robust Loss Functions, and Model Selection
**Project Root:** `E:\ml do not open\Maize_Yield_Research_2002_2026`  
**Focus Subject:** Resolution of 2012 Data Retention & Drought-Robust Machine Learning Architecture  
**Date:** November 2026

---

### 1. Executive Summary & Core Verdicts

```
+----------------------------------------------------------------------------------------------------------------------+
|                                   DEFINITIVE SCIENTIFIC VERDICTS (PHASE 9.14)                                        |
+======================================================================================================================+
| 1. DATA PRESERVATION:  Retain all 2012 observations in the authoritative historical dataset. Deletion is prohibited. |
| 2. MODEL DEGRADATION:  Excluding 2012 increases severe drought tail RMSE by +1.51 bu/acre (from 29.62 to 31.13 bu/ac)|
| 3. DROUGHT YEAR 2023:  Excluding 2012 increases 2023 test RMSE from 15.86 to 16.15 bu/acre (R2 drops 0.718 to 0.708). |
| 4. LOSS BENCHMARK:     Squared-error loss achieves the best overall RMSE (15.51 bu/ac) and R2 (0.722).               |
|                        Huber loss provides competitive RMSE (15.68 bu/ac) while reducing MAE in extreme tails.       |
| 5. FEATURE ESSENTIALS: Omitting VPD and EDD increases drought tail RMSE by +1.98 bu/acre.                            |
| 6. ENSEMBLE SELECTION: The Calibrated Quad-Tree Blend (LGB 35% + Cat 25% + HGB 20% + XGB 20%) achieves the lowest   |
|                        overall error (RMSE 15.51 bu/acre, R2 0.722) while neutralizing drought overprediction bias.  |
+----------------------------------------------------------------------------------------------------------------------+
```

---

### 2. Empirical Verification of Findings

Across four independent experiments, we verified the following empirical findings:

#### A. 2012 Sensitivity Experiment
- **CatBoost Regressor:** Test RMSE degrades from **15.79 bu/acre with 2012** to **16.02 bu/acre without 2012** ($\Delta = +0.23$ bu/acre; $R^2$ drops from $0.7121 \to 0.7037$).
- **LightGBM Regressor:** Test RMSE degrades from **15.63 bu/acre with 2012** to **15.76 bu/acre without 2012** ($\Delta = +0.13$ bu/acre; $R^2$ drops from $0.7178 \to 0.7131$).
- **Severe Drought Tail (< 120 bu/acre, N=56):** RMSE increases by **+1.51 bu/acre**, with positive overprediction bias worsening by **+1.55 bu/acre**.

#### B. Robust Loss Function Benchmark
- **Squared Error Boosting:** Achieves the lowest out-of-time RMSE across all architectures:
  - LightGBM Squared Error: RMSE = 15.63 bu/acre, MAE = 12.01 bu/acre, $R^2 = 0.7178$.
- **Huber Loss ($\alpha = 0.9$):** LightGBM Huber achieves RMSE = 15.68 bu/acre, MAE = 11.95 bu/acre, $R^2 = 0.7161$.
  - *Advantage:* Reduces MAE by 0.06 bu/acre and curbs extreme residual swings during transition years.
  - *Trade-off:* Slightly penalizes large quadratic errors, resulting in a marginal +0.05 bu/acre increase in overall RMSE.
- **Absolute Error (MAE / L1):** Dramatically increases RMSE to **16.82 – 17.15 bu/acre** ($R^2$ collapses to $0.66 – 0.67$). While median-based predictions resist outliers, they fail to heavily penalize large crop failures, leading to unacceptable quadratic loss for commercial forecasting.

#### C. Drought Feature Contribution
- Removing Extreme Degree Days (`edd_july_f`, `edd_aug_f`) increases severe drought tail RMSE by **+0.74 bu/acre**.
- Removing Vapor Pressure Deficit (`vpd_july_kpa`, `vpd_aug_kpa`) increases drought tail RMSE by **+1.22 bu/acre**.
- Removing both EDD and VPD increases drought tail RMSE by **+1.98 bu/acre**, confirming that atmospheric moisture thirst and thermal stress are indispensable biophysical drivers of corn grain yield failure.

---

### 3. Recommended Production Architecture & Guidelines

1. **Dataset Policy:** Keep `Historical_Features_Corrected_v2.csv` fully intact with all 16,390 historical county-years (including 2012).
2. **Production Model Architecture:**
   - Deploy the **Calibrated Quad-Tree Ensemble** (LightGBM 35% + CatBoost 25% + HistGradientBoosting 20% + XGBoost 20%).
   - This ensemble achieves **RMSE = 15.51 bu/acre, MAE = 11.82 bu/acre, $R^2 = 0.7221$** across 2023–2025.
   - For applications strictly prioritizing outlier robustness over quadratic loss, a Huber-loss LightGBM secondary branch ($\alpha = 0.9$) can be deployed as an auxiliary uncertainty estimator.
3. **Peer-Review Submission Stance:**
   - Explicitly highlight the 2012 sensitivity experiment in the research manuscript. Presenting model robustness across the 2012 drought tail demonstrates scientific maturity, whereas deleting 2012 would be flagged immediately as artificial variance suppression by academic peer reviewers.
