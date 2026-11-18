# COMPREHENSIVE RESEARCH METHODOLOGY SPECIFICATION

## Mathematical Formulations, Validation Protocols & Experimental Architecture
**Project:** County-Level Maize/Corn Grain Yield Prediction Research Project (2002–2026)  
**Output Target:** Peer-Reviewed Academic Manuscript  
**Date:** November 2026

---

### 1. Mathematical Problem Formulation

Let $i \in \{1, \dots, N\}$ index agricultural counties ($N = 788$) and $t \in \{2002, \dots, T\}$ index crop production years ($T = 2025$). Let $Y_{i, t} \in \mathbb{R}^+$ denote the true USDA NASS corn grain yield (bushels per harvested acre) for county $i$ in year $t$.

The prediction task is formulated as:
$$\hat{Y}_{i, t} = f\left(\mathbf{X}_{i, t}^{\text{mem}}, \mathbf{X}_{i}^{\text{soil}}, \mathbf{X}_{i, t}^{\text{clim}}, \mathbf{X}_{i, t}^{\text{adv\_clim}}, \mathbf{X}_{i, t}^{\text{irrig}}, \mathbf{X}_{s(i), t}^{\text{phen}}\right)$$

Where:
1. $\mathbf{X}_{i, t}^{\text{mem}} \in \mathbb{R}^3$: Autoregressive yield memory (1-year lag $Y_{i, t-1}$, 2-year lag $Y_{i, t-2}$, and 5-year rolling mean $\bar{Y}_{i, t-1:t-5}$).
2. $\mathbf{X}_{i}^{\text{soil}} \in \mathbb{R}^{26}$: Static ISRIC SoilGrids multi-depth horizons (0–5cm, 5–15cm, 15–30cm, and 0–30cm depth-weighted mean) for bulk density, clay, sand, silt, pH, and soil organic carbon.
3. $\mathbf{X}_{i, t}^{\text{clim}} \in \mathbb{R}^{18}$: Primary monthly meteorological series from NOAA NCEI Climatic Divisions and PRISM 4km rasters (monthly precipitation, monthly maximum, minimum, and mean temperatures).
4. $\mathbf{X}_{i, t}^{\text{adv\_clim}} \in \mathbb{R}^{9}$: Advanced agrometeorological features:
   - **Growing Degree Days (GDD):**
     $$\text{GDD}_{\text{season}} = \sum_{d=1}^{123} \max\left(0, \min(30^\circ\text{C}, T_{\text{mean}, d}) - 10^\circ\text{C}\right)$$
   - **Extreme Degree Days (EDD):**
     $$\text{EDD}_{\text{July}} = 31 \times \max\left(0, T_{\max, \text{July}} - 86.0^\circ\text{F}\right)$$
   - **Vapor Pressure Deficit (VPD):**
     $$\text{VPD} = e_s(T_{\max}) - e_s(T_{\min})$$
     Using Tetens equation for saturation vapor pressure:
     $$e_s(T) = 0.61078 \exp\left(\frac{17.27 T}{T + 237.3}\right) \text{ [kPa]}$$
   - **Compound Heat-Drought Stress:**
     $$\text{Compound\_Stress} = \left(\frac{\text{EDD}_{\text{July}}}{50}\right) \times \max\left(0, \frac{4.5 - P_{\text{July}}}{4.5}\right)$$
   - **Precipitation & Temperature Z-scores:**
     $$Z_{i, t} = \frac{X_{i, t} - \hat{\mu}_{i, \text{pre-2023}}}{\hat{\sigma}_{i, \text{pre-2023}}}$$
5. $\mathbf{X}_{i, t}^{\text{irrig}} \in \mathbb{R}^3$: Irrigation and management intensity features:
   - **Irrigation Regime Indicator:**
     $$\text{Irrig\_Proxy}_i = \begin{cases} 1.0 & \text{if } \text{State} = \text{NE} \\ \text{clip}\left(\frac{\text{Liability}_{i, t} - \text{Med}_s}{2 \times \text{IQR}_s}, 0, 0.4\right) & \text{otherwise} \end{cases}$$
   - **Groundwater Drought Buffer:**
     $$\text{Buffer}_{i, t} = \text{Irrig\_Proxy}_i \times \max(0, 4.0 - P_{\text{July}, i, t})$$
   - **USDA RMA Liability per Net Planted Acre:** Capturing economic capitalization and crop insurance guarantee level.
6. $\mathbf{X}_{s(i), t}^{\text{phen}} \in \mathbb{R}^9$: USDA NASS weekly crop progress and condition curves for state $s(i)$:
   - Silking-thermal interaction: $\text{Week}_{50\%\text{ silking}} \times T_{\max, \text{July}}$.
   - Delayed planting penalty: $\max(0, \text{Week}_{50\%\text{ planted}} - 20) \times P_{\text{May}}$.

---

### 2. Leakage-Free Temporal Validation Architecture

To eliminate cross-year and spatial information leakage:
1. **Prospective Primary Holdout:** Years 2002–2022 ($N = 14,691$ county-years) serve strictly as training data; years 2023–2025 ($N = 1,699$ county-years) serve as an independent, untouched evaluation period.
2. **Expanding-Window Rolling Origin:**
   $$\mathcal{T}_k = \{t \le 2018 + k\}, \quad \text{Test}_k = \{2019 + k\}, \quad k \in \{0, \dots, 6\}$$
3. **Out-of-Fold (OOF) Stacking Pipeline:** Base learners generate out-of-fold predictions strictly within past temporal folds; the meta-learner is fitted on historical OOF predictions and evaluated on prospective test years.

---

### 3. Model Architecture Specifications

#### A. Gradient Boosted Decision Trees (HistGB, LightGBM, XGBoost, CatBoost)
- Objective: Minimize Mean Squared Error $\mathcal{L}(y, \hat{y}) = \frac{1}{2}(y - \hat{y})^2$.
- Tree Depth: $6 - 7$; Learning Rate: $0.05$; Subsample: $0.80$; Column subsample by tree: $0.80$.
- Early stopping monitored on expanding temporal validation sets.

#### B. Generalized Additive Model (GAM) with Cubic B-Splines
- Non-linear spline basis expansion for thermal and moisture variables:
  $$\hat{Y}_{i, t} = \alpha_i + \sum_{j=1}^{p} f_j(X_{i, t, j}) + \mathbf{Z}_{i, t}^\top \boldsymbol{\beta}$$
  Where $f_j(\cdot)$ is parameterized via 5-knot cubic B-splines with an $L_2$ roughness penalty $\lambda = 10.0$.

#### C. Stacking Meta-Learner (Non-Negative Least Squares & Ridge)
- Meta-model formulation:
  $$\hat{Y}_{i, t}^{\text{ensemble}} = \beta_0 + \sum_{m=1}^{M} w_m \hat{Y}_{i, t}^{(m)}, \quad \text{s.t. } w_m \ge 0, \quad \sum w_m \approx 1$$
  Fitted using expanding-window OOF predictions to prevent meta-model overfitting.

---

### 4. Finite-Sample Conformal Prediction Mathematics

For any test point $(\mathbf{X}_{n+1}, Y_{n+1})$, we construct distribution-free prediction intervals $C(\mathbf{X}_{n+1}) = [\hat{Y}_{n+1} - q, \hat{Y}_{n+1} + q]$ such that:
$$\mathbb{P}\left(Y_{n+1} \in C(\mathbf{X}_{n+1})\right) \ge 1 - \alpha$$

1. Split training data into proper training set $\mathcal{I}_1$ and calibration set $\mathcal{I}_2$ ($N_{\text{cal}} = |\mathcal{I}_2|$).
2. Fit model $\hat{f}$ on $\mathcal{I}_1$.
3. Compute non-conformity scores: $R_i = |Y_i - \hat{f}(\mathbf{X}_i)|$ for $i \in \mathcal{I}_2$.
4. Compute the empirical $(1 - \alpha)$-quantile:
   $$q = \text{Quantile}\left(R_i, \frac{\lceil(N_{\text{cal}} + 1)(1 - \alpha)\rceil}{N_{\text{cal}}}\right)$$

---

### 5. Macroeconomic Grain Acreage Conversion Protocol

To calculate total regional corn production $P_{\text{regional}}$ (bushels):
$$P_{\text{regional}} = \sum_{i=1}^{N} \hat{Y}_{i, t} \times A_{i, t}^{\text{harvested\_grain}}$$

Where harvested grain acreage is obtained from USDA NASS county summaries or estimated via the county historical conversion ratio:
$$\kappa_i = \frac{A_{i}^{\text{harvested\_grain}}}{A_{i}^{\text{planted}}}$$
In dairy-intensive states (e.g. Wisconsin, $\kappa_{\text{WI}} = 0.829$), multiplying by planted acreage overstates grain production by $>17\%$.

