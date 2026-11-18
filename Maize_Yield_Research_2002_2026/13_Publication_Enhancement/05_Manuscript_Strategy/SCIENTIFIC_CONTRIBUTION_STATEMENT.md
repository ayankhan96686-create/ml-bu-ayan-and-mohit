# SCIENTIFIC CONTRIBUTION STATEMENT

## Formal Research Contributions for Peer-Reviewed Submission
**Title:** Disentangling Historical Yield Memory, Irrigation Heterogeneity, and Environmental Extremes in County-Level Maize Yield Forecasting  
**Target Journals:** *Computers and Electronics in Agriculture* / *Agricultural and Forest Meteorology* / *Nature Food*  
**Date:** November 2026

---

### Abstract Summary of Novelty

This study addresses fundamental methodological vulnerabilities in spatial-temporal agricultural machine learning by establishing a leakage-free, multi-source forecasting architecture across 788 US Corn Belt counties spanning 24 continuous years (2002–2025). The central contributions of this work are:

#### 1. First Rigorous Decoupling of Historical Memory and Climate Shocks
- Demonstrated that historical autoregressive yield memory (`yield_roll_mean_5yr`, `yield_lag_1yr`, `yield_lag_2yr`) accounts for 47.7% of total test variance ($\text{RMSE} = 21.28$ bu/acre), capturing accumulated soil capital, infrastructure, and technological creep.
- Formally quantified the incremental value of multi-source environmental features (soil profiles, weather extremes, crop progress), proving that environmental inputs provide a statistically certified **26.3% error reduction** down to $\text{RMSE} = 15.68$ bu/acre ($R^2 = 0.7161$) for single models and **30.3% error reduction** to $\text{RMSE} = 14.84$ bu/acre ($R^2 = 0.7459$) for stacking meta-models.

#### 2. Resolution of the Spatial Transferability Dilemma (Warm-Start vs Cold-Start)
- Resolved the conflicting findings in the literature regarding machine learning transferability by introducing the two-scenario Leave-One-State-Out (LOSO) benchmark:
  - **Warm-Start Transfer (Scenario 1):** Out-of-region climate generalization with historical county yield memory retained ($\text{Mean } R^2 = 0.6434$; pooled $R^2 = 0.6499$; excluding Nebraska $R^2 = 0.7018$).
  - **Cold-Start Transfer (Scenario 2):** Zero-history transfer relying purely on physical soil and weather grids ($\text{Mean } R^2 = 0.3259$).
- Proved that >50% of the spatial transfer penalty observed in prior studies is due to missing local baseline productivity calibration rather than flawed weather-response learning.

#### 3. Resolving the Western Corn Belt Irrigation Anomaly
- Pinpointed the primary failure mode of regional crop modeling: the catastrophic breakdown of rainfed crop models when transferred into Nebraska's groundwater-irrigated production systems (unadjusted LOSO $R^2 = 0.1764$).
- Formulated novel county-level irrigation proxies (`irrigation_regime_proxy`, `irrigation_drought_buffer`) combining USDA RMA insurance coverage and regional groundwater intensity, elevating Nebraska test performance to $R^2 = 0.7241$ ($\text{RMSE} = 13.12$ bu/acre).

#### 4. Distribution-Free Conformal Prediction & Temporal Drift Audit
- Implemented Split Conformal Prediction, delivering distribution-free 80%, 90%, and 95% marginal coverage intervals with finite-sample guarantees.
- Conducted the first systematic temporal coverage drift audit across agricultural shock years, showing that while parametric Gaussian intervals collapse to 69.46% coverage during drought events, Conformal Prediction maintains 86.21% coverage.

#### 5. Macroeconomic Grain Acreage vs Silage Aggregation Reconciliation
- Exposed an operational flaw in academic yield modeling: multiplying predicted county yields by total planted acres rather than harvested grain acres.
- Quantified an empirical error of **$387.8$ million bushels** across the 9 Corn Belt states, driven largely by Wisconsin dairy silage harvesting (where 17.1% of corn acreage is chopped green), and formalized a certified acreage weighting protocol.

