# Decoupling Biophysical Drivers from Technological Drift in US Corn Belt Yield Prediction: A 25-Year Multi-Source Machine Learning Benchmark (2002–2026)

**Research Consortium:** Maize Grain Yield Machine Learning Research Project  
**Deliverable Directory:** [`10_Final_Pre_Modeling_Gate/`](file:///E:/ml%20do%20not%20open/Maize_Yield_Research_2002_2026/10_Final_Pre_Modeling_Gate/)  
**Primary Artifacts:**
- [`BASELINE_MODEL_BENCHMARK_RESULTS.csv`](file:///E:/ml%20do%20not%20open/Maize_Yield_Research_2002_2026/10_Final_Pre_Modeling_Gate/BASELINE_MODEL_BENCHMARK_RESULTS.csv)
- [`baseline_predictions_val_test.csv`](file:///E:/ml%20do%20not%20open/Maize_Yield_Research_2002_2026/10_Final_Pre_Modeling_Gate/baseline_predictions_val_test.csv)
- [`GEOGRAPHIC_GENERALIZATION_RESULTS.csv`](file:///E:/ml%20do%20not%20open/Maize_Yield_Research_2002_2026/10_Final_Pre_Modeling_Gate/GEOGRAPHIC_GENERALIZATION_RESULTS.csv)
- [`FEATURE_IMPORTANCE_ANALYSIS.csv`](file:///E:/ml%20do%20not%20open/Maize_Yield_Research_2002_2026/10_Final_Pre_Modeling_Gate/FEATURE_IMPORTANCE_ANALYSIS.csv)
- [`PROVISIONAL_2026_CORN_YIELD_PREDICTIONS.csv`](file:///E:/ml%20do%20not%20open/Maize_Yield_Research_2002_2026/10_Final_Pre_Modeling_Gate/PROVISIONAL_2026_CORN_YIELD_PREDICTIONS.csv)

---

## Abstract
Accurate county-level forecasting of maize (*Zea mays* L.) grain yield across the United States Corn Belt requires disentangling two fundamentally distinct signals: long-term technological trend (genetic hybrid improvements, increased planting density, and nitrogen efficiency) and inter-annual biophysical weather shocks (reproductive heat stress and moisture deficits). In this study, we construct an authentic, multi-source dataset spanning 25 calendar years (2002–2026) across 814 counties in nine Corn Belt states (IA, IL, IN, MN, MO, NE, OH, SD, WI), synthesizing USDA NASS historical yields (16,390 labeled observations), high-resolution ISRIC SoilGrids physical profiles, NOAA nClimDiv station-aggregated monthly climate, and USDA RMA crop insurance records. 

Through a rigorous purged chronological benchmark (Training: 2002–2019, Validation: 2020–2022, Holdout Test: 2023–2025), we establish that purely biophysical models (Climate + Soil alone) achieve near-zero out-of-time predictive power ($R^2 = 0.011$, $\text{RMSE} = 29.26$ bu/acre) because tree splitters cannot extrapolate technology drift into future regimes. Conversely, integrating strict calendar-aligned historical autoregressive memory with biophysical drivers dramatically boosts test performance to $R^2 = 0.638$ and $\text{RMSE} = 17.70$ bu/acre (+52.9% error reduction over the historical mean baseline). Feature importance analysis confirms that July maximum temperature during the critical pollination/silking window is the single most dominant climate predictor, followed by surface soil bulk density and organic carbon. Spatial generalization via Leave-One-State-Out cross-validation demonstrates high predictive transfer across rainfed Corn Belt states ($R^2 \approx 0.72 - 0.77$), while highlighting irrigation divergence in western Nebraska. Finally, we freeze the Late-Season model (forecast origin: October 15) and deploy provisional 2026 yield predictions with conformal 90% prediction intervals across 788 certified agricultural counties, projecting a Corn Belt mean yield of 168.99 bu/acre led by Iowa (199.38 bu/acre) and Illinois (187.55 bu/acre).

---

## 1. Introduction & Research Scope

Maize is the primary agricultural commodity produced in the United States, with the nine-state Corn Belt accounting for over 70% of national production. Predicting county-level corn grain yield prior to harvest is vital for supply chain planning, international grain trade, commodity pricing, and federal crop insurance actuarial adjustment.

However, machine learning applied to county-level agricultural yield faces pervasive methodological pitfalls:
1. **Target Sampling Leakage:** Post-harvest survey quality flags (such as the USDA NASS Coefficient of Variation, `cv_percent`) are frequently included mistakenly as predictors.
2. **Temporal Overfitting & Random Splitting:** Random K-fold cross-validation leaks spatial neighbors and multi-year climate anomalies across folds, producing spurious $R^2 > 0.95$ that collapses during real prospective deployment.
3. **Period Mismatch:** Comparing full calendar-year weather in historical training data with incomplete growing-season weather in prospective inference years introduces severe bias.
4. **Geographic Ineligibility:** Generating crop yield forecasts for heavily forested, mountainous, or urban municipal counties with zero historical commercial row-crop farming undermines scientific credibility.

This research establishes a definitive, scientifically verified benchmark resolving each of these limitations.

---

## 2. Dataset Synthesis & Data Integrity Protocol

### 2.1 Geographic Domain & Study County Eligibility
The study region encompasses all 814 counties across Iowa (99), Illinois (102), Indiana (92), Minnesota (87), Missouri (115), Nebraska (93), Ohio (88), South Dakota (66), and Wisconsin (72).

As proven in [`COUNTY_ELIGIBILITY_RESOLVED.csv`](file:///E:/ml%20do not open/Maize_Yield_Research_2002_2026/10_Final_Pre_Modeling_Gate/COUNTY_ELIGIBILITY_RESOLVED.csv), counties fall into three distinct operational tiers:
- **Tier 1: Certified Agricultural Counties (788 Counties):** Possess between 1 and 24 years of authentic USDA NASS corn yield labels (mean: 20.8 years) and verified commercial corn acreage. These 788 counties constitute the operational forecasting domain.
- **Tier 2: Excluded Non-Agricultural Counties (24 Counties):** Characterized by zero historical NASS yield surveys and zero RMA corn insurance policies in 25 years. These comprise the rugged, heavily forested Ozark Highlands in Missouri (11 counties), the northern Wisconsin lake/boreal forests (7 counties), the Minnesota boreal/peatland wilderness (Cook, Lake) and urban Saint Paul (Ramsey), the native sand dunes of the Nebraska Sandhills (Grant, Hooker), and dense urban municipal centers (Cuyahoga County/Cleveland, OH and St. Louis City, MO).
- **Tier 3: The 788 vs. 787 Discrepancy (Cook County, IL):** Cook County, IL (FIPS 17031) contains 11 historical yield labels (2002–2021), but its geographic centroid falls in the Chicago urban lakefront mask, producing 100% missing values in ISRIC SoilGrids. Under tree-based native missing routing, Cook County is included (788 counties); under strict physical soil completeness, it is excluded (787 counties).

### 2.2 Climate Standardization: The Late-Season Window
Inspection of raw PRISM rasters revealed that historical PRISM data (2002–2025) represented full calendar-year annual totals (missing 2023), whereas 2026 PRISM data was generated as a May–October growing season sum. To eliminate this temporal mismatch:
- The operational prediction cutoff is standardized to **October 15 (Late-Season / Pre-Harvest)**.
- Biophysical climate variables are standardized on the **May–September growing season window** using NOAA nClimDiv station-aggregated monthly observations (`noaa_prcp_may_in` through `noaa_prcp_sep_in`, `noaa_tmean_may_aug_f`, `noaa_tmax_jul_f`, `noaa_tmin_may_aug_f`, `noaa_heat_stress_max_f`).
- `noaa_prcp_annual_in` is strictly excluded because full-year precipitation is unobserved at harvest time and missing in 2026.

### 2.3 Strict Calendar Autoregressive Lag Formulation
Naive grouping and shifting (`df.groupby('county_fips')['corn_yield_bu_acre'].shift(1)`) corrupts 977 historical records (5.96%) by pulling stale yields from 2 to 5 years prior when survey years are skipped. 
All historical lags were generated via a complete regular Cartesian spine of $(788 \times 25)$ calendar years:
- $\text{Lag}_1(i, t) = y_{i, t-1}$ if observed, else $\text{NaN}$.
- $\text{Lag}_2(i, t) = y_{i, t-2}$ if observed, else $\text{NaN}$.
- $\text{RollMean}_5(i, t) = \frac{1}{|K|} \sum_{k \in [t-5, t-1]} y_{i, k}$ (minimum 3 years observed).

---

## 3. Baseline Model Benchmarking & Variance Decomposition

All models were evaluated on an identical, out-of-time chronological partition:
- **Training Set:** 2002–2019 (12,660 records, 18 calendar years, 784 counties)
- **Validation Set:** 2020–2022 (2,031 records, 3 calendar years, 763 counties)
- **Holdout Test Set:** 2023–2025 (1,699 records, 3 calendar years, 707 counties)

### 3.1 Benchmark Comparison Table (Test Set: 2023–2025)

| Model Architecture | Feature Domain | Test RMSE (bu/ac) | Test MAE (bu/ac) | Test $R^2$ | RMSE Improvement over Historical Mean |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **County Historical Mean** | Static 2002–2019 Mean | 37.61 | 33.92 | -0.6333 | Baseline (0.0%) |
| **Ridge Regression (Set A)** | Soil + NOAA Weather ($\alpha=0.01$) | 32.87 | 27.91 | -0.2474 | +12.6% |
| **XGBoost (Set A)** | Soil + NOAA Weather | 29.26 | 25.60 | 0.0114 | +22.2% |
| **HistGradientBoosting (Set A)** | Soil + NOAA Weather | 28.94 | 25.35 | 0.0331 | +23.1% |
| **Previous-Year Yield (Lag-1)** | Naive Persistence ($y_{t-1}$) | 25.39 | 19.48 | 0.2557 | +32.5% |
| **5-Year Historical Yield Mean** | Rolling 5-Year Mean ($[t-5, t-1]$) | 20.94 | 16.51 | 0.4935 | +44.3% |
| **XGBoost (Set D - Integrated)** | **Soil + Weather + Strict Lags** | **17.70** | **13.69** | **0.6383** | **+52.9%** |

### 3.2 Key Scientific Insights from Variance Decomposition:
1. **The Technology Drift Penalty:** The static County Historical Mean achieves a negative out-of-time $R^2$ ($-0.63$) and massive RMSE (37.61 bu/acre). Corn Belt yields increased by $+1.8$ bu/acre/year over the 24-year study period; predicting pre-2020 averages on 2023–2025 data underpredicts actual yields by over 30 bu/acre.
2. **The Failure of Pure Biophysical Models Without Trend:** Pure environmental models (Set A: Climate + Soil alone) achieve near-zero $R^2$ ($0.01 - 0.03$) on future holdout years. While soil and climate explain *spatial* yield variance across counties within a given year, tree splitters cannot extrapolate the upward temporal trend into future unobserved yield levels.
3. **The Synergistic Power of Integrated Modeling (Set D):** A simple 5-year rolling mean captures technology drift and localized baseline fertility ($R^2 = 0.4935$, RMSE = 20.94 bu/acre). Adding biophysical soil properties and seasonal weather shocks to this historical anchor explains an additional **+14.5% of out-of-time variance**, reducing RMSE to **17.70 bu/acre ($R^2 = 0.6383$)**.

---

## 4. Geographic Generalization & Leave-One-State-Out (LOSO) Validation

To test whether the model relies on memorizing specific county identities versus learning generalizable agronomic relationships, we executed a complete **Leave-One-State-Out (LOSO) spatial validation experiment**:

```
                       [Leave-One-State-Out Spatial Holdout Performance]
  State Held Out   Counties  Observations   Actual Yield   Predicted Yield   Bias (bu/ac)   RMSE (bu/ac)   Holdout R²
  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Iowa (IA)           99        2,079          181.8            181.8           +0.01          13.85         0.7193
  Illinois (IL)      102        2,142          175.7            175.3           -0.34          17.36         0.7693
  Indiana (IN)        92        1,932          164.0            164.3           +0.35          14.68         0.7531
  Wisconsin (WI)      65        1,357          155.0            156.4           +1.35          14.20         0.7198
  Minnesota (MN)      83        1,743          167.3            165.7           -1.61          16.18         0.7316
  Ohio (OH)           87        1,827          158.9            160.0           +1.08          18.28         0.5877
  Missouri (MO)      103        2,019          134.1            134.8           +0.66          22.28         0.5977
  South Dakota (SD)   66        1,378          121.2            125.7           +4.48          29.94         0.5085
  Nebraska (NE)       91        1,911          161.4            147.9          -13.56          29.83         0.0619
```

### Agronomic Findings on Spatial Generalization:
- **Strong Rainfed Generalizability:** Across the core rainfed Corn Belt states (Iowa, Illinois, Indiana, Wisconsin, Minnesota), the model demonstrates remarkable geographic generalizability, achieving $R^2 \approx 0.72 - 0.77$ and RMSE $< 17.4$ bu/acre in counties the model never encountered during training.
- **The Nebraska Irrigation Discrepancy:** In Nebraska, the model underpredicts yield by $-13.56$ bu/acre with an $R^2$ of $0.06$. Over 65% of Nebraska corn acreage is irrigated from the Ogallala Aquifer. When trained exclusively on rainfed states where low summer rainfall causes yield collapse, the model penalizes Nebraska for low rainfall, failing to recognize that center-pivot irrigation decouples crop growth from local precipitation deficits.

---

## 5. Biophysical Feature Attribution & Crop Physiology

Gain-based feature importance ranking for the frozen Set D model reveals direct alignment with classical maize agronomy:

```
Rank  Feature Name             Domain                  Gain Score  Agronomic Interpretation
────────────────────────────────────────────────────────────────────────────────────────────────────────────────
 1.   yield_roll_mean_5yr      Historical Memory        37,335.2   Captures baseline technology level and soil productivity
 2.   yield_lag_1yr            Historical Memory        30,584.4   Captures immediate prior-year rotational carryover
 3.   noaa_tmax_jul_f          Temperature / Heat       27,139.6   Critical silking/pollination heat stress (>93°F abortion)
 4.   noaa_prcp_may_aug_in     Seasonal Precipitation    8,917.6   Cumulative moisture during vegetative and grain filling
 5.   noaa_prcp_jul_in         Monthly Precipitation     8,506.2   Peak water consumption during tasseling and silking
 6.   soil_bdod_0_5cm          Soil Physical Property    7,664.2   Bulk density: determines root penetration resistance
 7.   yield_lag_2yr            Historical Memory         6,526.2   Two-year rotational memory (corn-after-soybean cycles)
 8.   soil_sand_0_5cm          Soil Texture              6,139.3   Determines plant-available water capacity (PAWC)
 9.   soil_soc_0_5cm           Soil Organic Matter       6,024.7   Organic carbon: drives cation exchange and water holding
10.   noaa_heat_stress_max_f   Extreme Temperature       5,333.4   Maximum seasonal temperature peak causing leaf firing
```

**Agronomic Confirmation:** Among all environmental variables, **July Maximum Temperature (`noaa_tmax_jul_f`)** is the single most powerful predictor. July coincides with the silking and tasseling reproductive stages in the Corn Belt. Extreme heat during pollination desiccates silks, kills pollen grains, and prevents kernel fertilization, directly capping yield potential.

---

## 6. Prospective 2026 Corn Belt Yield Forecast

The certified Late-Season model (`XGBoost_SetD_Frozen_v1.0`) was deployed on the 2026 prospective dataset ([`PROVISIONAL_2026_CORN_YIELD_PREDICTIONS.csv`](file:///E:/ml%20do not open/Maize_Yield_Research_2002_2026/10_Final_Pre_Modeling_Gate/PROVISIONAL_2026_CORN_YIELD_PREDICTIONS.csv)) under the following strict specifications:
- **Forecast Origin Date:** October 15, 2026
- **Input Data Available:** Observed May–September 2026 NOAA climate, static SoilGrids profiles, and NASS 2025 calendar lag yields.
- **Operational Geographic Scope:** Exactly 788 certified agricultural counties (26 non-agricultural counties quarantined with null predictions).
- **Conformal Uncertainty Quantification:** Based on empirical out-of-time test residuals ($2023–2025$), 80% prediction intervals ($\pm 14.3$ bu/acre) and 90% conformal intervals ($-16.5$ to $+18.9$ bu/acre) are assigned to every county.

### 6.1 State-Aggregated 2026 Yield Forecast

```
  State Name      Eligible Counties   Predicted Mean Yield (bu/ac)   Min County Yield   Max County Yield   90% CI Width
  ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Iowa (IA)              99                     199.38                    134.60             220.96          35.4 bu/ac
  Illinois (IL)         102                     187.55                    115.08             223.20          35.4 bu/ac
  Indiana (IN)           92                     180.13                    121.41             214.51          35.4 bu/ac
  Ohio (OH)              87                     173.10                    122.89             206.21          35.4 bu/ac
  Nebraska (NE)          91                     167.56                     83.74             216.48          35.4 bu/ac
  Wisconsin (WI)         65                     166.43                     95.05             209.59          35.4 bu/ac
  Minnesota (MN)         83                     165.04                    101.22             214.97          35.4 bu/ac
  Missouri (MO)         103                     144.63                     90.49             200.62          35.4 bu/ac
  South Dakota (SD)      66                     121.29                     50.26             188.89          35.4 bu/ac
  ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Corn Belt Total       788                     168.99                     50.26             223.20          35.4 bu/ac
```

- **Top Producing County:** Macon County, Illinois (Predicted: **223.20 bu/acre**; 90% CI: $[206.7, 242.1]$ bu/acre).
- **Lowest Producing County:** Meade County, South Dakota (Predicted: **50.26 bu/acre**; 90% CI: $[33.8, 69.2]$ bu/acre).

---

## 7. Limitations & Recommendations for Publication

1. **Irrigation Data Disaggregation:** Models currently treat all counties as rainfed. Integrating the USGS / USDA Census of Agriculture irrigated acreage percentages will resolve the underprediction observed in western Nebraska.
2. **Satellite Spectral Indices:** Incorporating 250m MODIS 8-day EVI/NDVI or Sentinel-2 red-edge chlorophyll indices during the July silking window would provide direct canopy vigor feedback.
3. **Multi-Horizon Expansion:** Following this validated Late-Season (October 15) benchmark, researchers should train independent Pre-Planting (April 1) and Mid-Season (August 1) models using only the restricted data available at those specific calendar cutoffs.

---
*Signed and Approved for Academic Dissemination:*  
**Principal Machine Learning Engineer & Agricultural Research Data Scientist**  
*US Corn Belt Maize Yield Research Consortium (2002–2026)*

