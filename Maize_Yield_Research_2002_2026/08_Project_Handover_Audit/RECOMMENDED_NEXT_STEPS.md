# Recommended Next Steps & Research Roadmap

**Project:** Maize/Corn Grain Yield Prediction Using Machine Learning (2002–2025)  
**Root Directory:** `E:/ml do not open/Maize_Yield_Research_2002_2026/`  
**Audit Date:** 2026-11-10  
**Auditor:** Senior Data Engineer, Machine Learning Lead & Agricultural Data Scientist  

---

## 1. Phased Research & Implementation Roadmap

This roadmap provides an actionable, prioritized execution plan designed for the next data scientist, machine learning engineer, or independent reviewer taking stewardship of this project.

```
+----------------------------------------------------------------------------------------------------+
| RESEARCH ROADMAP PHASES                                                                            |
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|  [ PHASE 1: IMMEDIATE DATA PATCHING (Est. Time: 1–2 Hours) ]                                       |
|  - Download missing PRISM 2023 precipitation raster                                                |
|  - Resolve Cook County, IL soil missingness                                                        |
|  - Re-run process_prism.py & integrate_datasets.py                                                 |
|                                                                                                    |
|  [ PHASE 2: ADVANCED FEATURE ENGINEERING (Est. Time: 3–4 Hours) ]                                  |
|  - Compute autoregressive yield lags (Y_{t-1}, 5-yr historical mean)                               |
|  - Calculate Growing Degree Days (GDD) and Extreme Degree Days (EDD > 86°F)                        |
|  - Construct detrended target yield anomalies (ΔY_t = Y_t - Trend_t)                               |
|                                                                                                    |
|  [ PHASE 3: SATELLITE REMOTE SENSING INTEGRATION (Est. Time: 1–2 Days) ]                           |
|  - Acquire MODIS MOD13A2 16-day 1km NDVI & EVI county time series via Google Earth Engine          |
|  - Extract annual pure-corn crop mask from CDL via GEE zonal statistics                            |
|                                                                                                    |
|  [ PHASE 4: MACHINE LEARNING BENCHMARKING (Est. Time: 2–3 Days) ]                                  |
|  - Implement Time-Series Blocked CV pipeline (Train: 2002–2019, Val: 2020–2022, Test: 2023–2025)  |
|  - Benchmark Ridge/ElasticNet, Random Forest, XGBoost, LightGBM, and CatBoost                      |
|  - Optimize hyperparameters using expanding-window time-series CV                                  |
|                                                                                                    |
|  [ PHASE 5: MODEL EXPLAINABILITY & PROSPECTIVE FORECASTING (Est. Time: 1–2 Days) ]                 |
|  - Compute SHAP (SHapley Additive exPlanations) values for biophysical feature importance          |
|  - Generate prospective 2026 county-level grain yield predictions with uncertainty intervals       |
|                                                                                                    |
+----------------------------------------------------------------------------------------------------+
```

---

## 2. Detailed Action Plan

### Phase 1: Immediate Data Patching
1. **Download Missing PRISM 2023 GeoTIFF:**
   - URL: `https://ftp.prism.oregonstate.edu/normals_4km/ppt/2023/` or annual PRISM portal.
   - Save to: `02_Extracted_Data/03_PRISM_Weather/ppt/2023/prism_ppt_us_25m_2023.tif`.
   - Re-run `07_Code/process_prism.py` and `07_Code/integrate_datasets.py`.
   - *Outcome:* Eliminates all 632 missing values in `prism_ppt_mm`.
2. **Resolve Cook County, IL Soil Mask:**
   - In `07_Code/process_soil.py`, assign Cook County soil properties using the mean of adjacent collar agricultural counties (`Will`: `17197`, `Kane`: `17089`), or flag Cook County for exclusion in purely rural agronomic training.

---

### Phase 2: High-Impact Feature Engineering
1. **Autoregressive Yield Lag Features:**
   ```python
   # Add to 07_Code/integrate_datasets.py
   ml_df['yield_lag_1yr'] = ml_df.groupby('county_fips')['corn_yield_bu_acre'].shift(1)
   ml_df['yield_rolling_5yr_mean'] = ml_df.groupby('county_fips')['corn_yield_bu_acre'].transform(
       lambda s: s.shift(1).rolling(5, min_periods=2).mean()
   )
   ```
2. **Technological Trend Decomposition:**
   - Fit linear county or state trends: $\hat{Y}_{it} = \alpha_i + \beta_i \cdot \text{year}$.
   - Define yield anomaly: $\tilde{Y}_{it} = Y_{it} - \hat{Y}_{it}$.
   - Predicting yield anomaly removes secular technological drift and allows algorithms to isolate pure meteorological and soil responses.
3. **Biophysical Degree Days:**
   - Approximate cumulative GDD (May–Aug) from monthly temperatures:
     $$\text{GDD}_{\text{month}} = \max\left(0, \frac{T_{\max} + T_{\min}}{2} - 50^\circ\text{F}\right) \times \text{Days}$$

---

### Phase 3: Satellite Remote Sensing Integration
1. **Google Earth Engine (GEE) Python API Pipeline:**
   - Query collection `MODIS/061/MOD13A2` (16-day vegetation indices).
   - Filter by USDA CDL corn pixels (`crop_code == 1`) within each county polygon.
   - Extract:
     - `ndvi_peak_value`: Maximum canopy greenness (typically late July).
     - `ndvi_integral_season`: Cumulative area under the NDVI curve (proxy for total biomass).
     - `evi_reproductive_mean`: Enhanced Vegetation Index during silking.
   - Export directly as a clean tabular CSV: `03_Cleaned_Data/Satellite_MODIS_Cleaned.csv`.

---

### Phase 4: Machine Learning Model Benchmarking
1. **Establish Baseline Suite:**
   - Linear Baseline: ElasticNet Regressor (with standard scaling inside CV pipeline).
   - Tree Ensembles: Random Forest, LightGBM Regressor, XGBoost Regressor, CatBoost Regressor.
2. **Enforce Strict Temporal Validation:**
   - Never use random train/test splits.
   - Evaluate model robustness specifically on extreme climate shock years:
     - 2012 Severe Drought Shock
     - 2019 Historic Midwest Planting Floods
3. **Evaluation Metrics Reporting:**
   - Tabulate Test RMSE, MAE, $R^2$, and Normalized RMSE (%) across held-out years 2023–2025.

---

### Phase 5: Prospective 2026 Forecasting & Research Publication
1. **Prospective 2026 Prediction:**
   - Feed `04_Integrated_Data/Future_2026_Feature_Table.csv` into the trained ensemble.
   - Output predicted county yield $\hat{Y}_{2026}$ with $90\%$ prediction intervals.
2. **Model Interpretability (XAI):**
   - Run TreeSHAP to verify that the model has learned authentic agronomic physics:
     - High July $T_{\max} > 90^\circ\text{F} \rightarrow$ Negative SHAP value (yield reduction).
     - Adequate May–August rainfall ($15\text{–}20\text{ inches}$) $\rightarrow$ Positive SHAP value.
     - Low topsoil SOC $\rightarrow$ Depressed baseline yield expectation.

