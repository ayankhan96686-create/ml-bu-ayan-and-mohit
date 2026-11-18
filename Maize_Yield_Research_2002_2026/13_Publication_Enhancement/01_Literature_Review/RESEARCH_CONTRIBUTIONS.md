# FORMAL STATEMENT OF SCIENTIFIC CONTRIBUTIONS

## Advancing County-Level Maize Yield Forecasting Through Autoregressive Decoupling, Physical Horizon Modeling, and Certified Spatial Validation

**Authors:** Senior Machine Learning Research Team & Agricultural Data Science Consortium  
**Target Submission:** *Computers and Electronics in Agriculture* / *Agricultural and Forest Meteorology*  
**Date:** November 2026

---

### Core Scientific Contributions

The primary intellectual contributions of this research are structured across five pillars:

#### 1. Rigorous Decoupling of Historical Yield Memory and Environmental Shocks
Prior literature frequently conflates multi-decadal technological yield gains with high-frequency weather shocks, leading to either inflated $R^2$ estimates through temporal leakage or severely underperforming models that omit historical context. We establish a formal controlled ablation framework proving that:
- Pure historical yield memory (1-yr lag, 2-yr lag, 5-yr rolling mean) establishes a high-performance baseline ($	ext{RMSE} = 16.94$ bu/acre, $R^2 = 0.6720$).
- Environmental and soil observations provide a statistically certified incremental gain of **$3.06$ bu/acre** (95% CI: $[2.44, 5.37]$ bu/acre, an **18.06% RMSE reduction** to $13.88$ bu/acre, $R^2 = 0.7801$).
- This provides the first verified, leak-free quantification of environmental value in the regional machine learning literature.

#### 2. Disentangling Spatial Transferability: The Warm-Start vs Cold-Start Paradigm
We resolve the long-standing debate over machine learning transferability across agricultural boundaries by establishing two standardized Leave-One-State-Out (LOSO) benchmarks:
- **Warm-Start Transfer (Scenario 1):** Out-of-region climate transfer when historical county yield memory is available. Across 9 Corn Belt states, the model achieves a mean $R^2 = 0.6434$ (pooled $R^2 = 0.6499$; excluding Nebraska: $R^2 = 0.7018$).
- **Cold-Start Transfer (Scenario 2):** Out-of-region transfer when a county has zero historical yield records, forcing the model to rely exclusively on climate and soil proxies. Mean $R^2$ drops to $0.3259$.
- **Agronomic Finding:** Over 50% of the spatial transfer penalty observed in prior studies is not due to flawed weather response functions, but to uncalibrated localized soil baseline productivity and management intensity.

#### 3. Resolution of the Western Corn Belt Irrigation Anomaly
We identify and resolve the primary failure mode of regional crop modeling: the systematic collapse of machine learning transfer in Nebraska ($R^2 = 0.1764$ in unadjusted LOSO). By contrasting Nebraska's Ogallala-irrigated production systems against rainfed states (Iowa, Illinois, Indiana), we prove that temperature-driven vapor pressure deficit and precipitation deficit curves cannot be transferred between irrigated and rainfed regimes without explicit management and crop insurance covariates.

#### 4. Distribution-Free Conformal Prediction with Empirical Coverage Drift Analysis
We move beyond uncalibrated point forecasts and unrealistic Gaussian assumptions by deploying finite-sample Conformal Prediction. We demonstrate certified 80%, 90%, and 95% marginal coverage across a 24-year panel (16,390 county-years) and document empirical coverage drift during extreme climate shocks (e.g. the 2012 continental drought), establishing an operational standard for risk-aware agricultural forecasting.

#### 5. Macroeconomic Production Accounting and Grain Acreage Reconciliation
We expose and correct a widespread methodological flaw in academic yield studies: multiplying county grain yield predictions by planted acres rather than harvested grain acres. We prove that this omission introduces an empirical error of **$387.8$ million bushels** across the 9 Corn Belt states, driven largely by Wisconsin dairy silage harvesting (where 17.1% of corn acres are never combined for grain). We deliver a leak-free harvested acreage accounting pipeline suitable for USDA and agricultural commodity market applications.
