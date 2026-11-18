# FINAL SCIENTIFIC METHODS AND EXPERIMENTAL DESIGN

## Comprehensive Description of Data Pipelines, Biophysical Feature Engineering, and Validation Protocols
**Project:** County-Level Maize/Corn Grain Yield Prediction (2002–2026)  
**Output Directory:** `15_Final_Confirmatory_Validation/07_Manuscript/`  
**Date:** November 2026

---

### 1. Study Area, Geographic Scope, and Panel Structure

The study covers nine core US Corn Belt states: **Iowa (IA), Illinois (IL), Indiana (IN), Minnesota (MN), Missouri (MO), Nebraska (NE), Ohio (OH), South Dakota (SD), and Wisconsin (WI)**.
- **Total Counties in Region:** 814 counties.
- **Historically Eligible Agricultural Counties:** 788 counties meeting the certified threshold of at least 1,000 harvested corn grain acres across historical USDA NASS surveys.
- **Quarantined Non-Agricultural Counties:** 26 counties (primarily urban centers such as Cook County, IL / Chicago; Ramsey County, MN / St. Paul; Milwaukee County, WI; and non-arable forest regions in northern Minnesota and Wisconsin) with negligible maize production, preserved for geographic completeness in prospective forecasting.
- **Temporal Panel:** 24 historical calendar years (2002–2025) comprising $N = 16,390$ verified county-year observations.
- **Partitions:**
  - **Model Training Cohort:** 2002–2022 ($N = 14,691$ records). Retains all 711 observations from the historic 2012 drought.
  - **Prospective Out-of-Time Test Cohort:** 2023–2025 ($N = 1,699$ records). Encompasses the 2023 drought shock ($N=632$), the 2024 record harvest ($N=546$), and the 2025 near-normal season ($N=521$).
  - **Prospective Forecast Horizon:** 2026 ($N = 814$ counties).

---

### 2. Multi-Source Environmental Feature Engineering (69 In-Season Predictors)

Predictors were constructed strictly using data observable on or before August 31 of each growing season, eliminating temporal data leakage:

1. **Historical Yield Autoregressive Memory (3 Predictors):**
   - `yield_lag_1yr`: Harvested yield in year $t-1$ (bu/acre).
   - `yield_lag_2yr`: Harvested yield in year $t-2$ (bu/acre).
   - `yield_roll_mean_5yr`: 5-year rolling harmonic/arithmetic average yield ($\bar{y}_{t-5:t-1}$), capturing local technological progress and management baseline.

2. **Physiographic and Soil Hydraulic Properties (SoilGrids 250m, 26 Predictors):**
   - Organic carbon stock, cation exchange capacity, pH, bulk density, sand, silt, and clay fractions across standard soil depth layers (0–5 cm, 5–15 cm, 15–30 cm, 30–60 cm, 60–100 cm).
   - `soil_awc_index`: Profile-integrated Available Water Capacity (mm) in the top 100 cm root zone.
   - `subsoil_clay_contrast`: Clay gradient between surface and subsoil layers, reflecting drainage impedance and perched water table tendencies.

3. **In-Season Gridded Meteorology (PRISM 4km & NOAA NCEI, 26 Predictors):**
   - Monthly and cumulative precipitation totals (May, June, July, August, May–August).
   - Temperature extremes (mean, minimum, maximum) across critical phenological windows.
   - `gdd_may_aug_c`, `gdd_july_c`: Accumulated Growing Degree Days (base 10°C, cutoff 30°C).
   - `edd_july_f`, `edd_aug_f`: Extreme Degree Days accumulated above 30°C (86°F), quantifying cellular thermal shock.
   - `vpd_july_kpa`, `vpd_aug_kpa`: Mean afternoon Vapor Pressure Deficit (kPa), quantifying transpirational evaporative stress.
   - `compound_heat_drought_july`: Bivariate interaction index combining normalized July heat accumulation with standardized precipitation deficits ($Z_{\text{EDD}} \times -Z_{\text{PRCP}}$).
   - `noaa_prcp_jul_in_zscore`: County-specific standardized July precipitation anomaly relative to 24-year climatological normal.

4. **Irrigation and Risk Attenuation Proxies (3 Predictors):**
   - `irrigation_regime_proxy`: Groundwater exploitation index distinguishing groundwater-irrigated acreage (e.g. Ogallala Aquifer in central/western Nebraska) from dryland rainfed production.
   - `irrigation_drought_buffer`: Interaction between irrigation intensity and July atmospheric vapor pressure deficit.
   - `rma_liability_per_planted_acre`: USDA Risk Management Agency crop insurance liability density ($/acre), capturing local capital investment and risk protection intensity.

5. **Crop Phenology and Canopy Condition (USDA NASS Weekly Progress, 11 Predictors):**
   - State-level planting progress (Week 20), emergence (Week 22), silking (Week 30), and dough stage (Week 34).
   - `silking_thermal_interaction`: Coincidence index between July silking peak and extreme heat days.
   - `delayed_planting_penalty`: Quadratic penalty for corn planted past optimal agronomic windows (post-May 20).
   - NASS crop condition ratings (percentage Excellent, Good, Poor) averaged through August.

---

### 3. Model Architecture and Stacking Framework

1. **Base Learners:**
   - **LightGBM:** Tree depth 7, 45 leaves, learning rate 0.04, feature subsampling 0.75, sample subsampling 0.85, 350 trees.
   - **XGBoost:** Tree depth 6, learning rate 0.04, feature subsampling 0.75, sample subsampling 0.80, 350 trees.
   - **CatBoost:** Oblivious tree depth 6, learning rate 0.04, 400 iterations.
   - **HistGradientBoosting:** Max depth 7, 300 iterations, learning rate 0.04, min samples per leaf 25.
   - **Multilayer Perceptron (MLP):** Two hidden layers (128, 64), ReLU activation, Adam optimizer, early stopping on validation loss.

2. **Ensemble Metamodels:**
   - **Ridge Stacking:** 5-fold cross-validated out-of-fold predictions from all five base learners stacked into a meta-feature matrix $\mathbf{X}_{\text{meta}} \in \mathbb{R}^{N \times 5}$. A regularized Ridge meta-regressor with penalty $\alpha = 150.0$ learns optimal linear combination weights, mitigating base learner collinearity and individual prediction variance.
   - **Calibrated Quad-Tree Blend:** Deterministic convex combination of the four tree models:
     $$\hat{y}_{\text{quad}} = 0.35 \hat{y}_{\text{LGB}} + 0.25 \hat{y}_{\text{Cat}} + 0.20 \hat{y}_{\text{HGB}} + 0.20 \hat{y}_{\text{XGB}}$$

---

### 4. Mathematical Validation Criteria

Model evaluation strictly conforms to non-negotiable algebraic definitions:
$$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2}, \quad \text{MAE} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$$
$$R^2 = 1 - \frac{\sum_{i=1}^N (y_i - \hat{y}_i)^2}{\sum_{i=1}^N (y_i - \bar{y})^2}, \quad \text{Mean Bias} = \frac{1}{N} \sum_{i=1}^N (\hat{y}_i - y_i)$$
Where $\bar{y}$ is strictly the actual mean of the evaluated test cohort.

