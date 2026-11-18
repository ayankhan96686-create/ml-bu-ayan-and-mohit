# PREPROCESSING AND ENCODING AUDIT: MAIZE YIELD PREDICTION (2002–2026)

**Project:** County-Level Maize (Corn) Grain Yield Prediction Research Project  
**Author:** Independent Statistical & Machine Learning Research Audit Team  
**Audit Target Directory:** `10_Final_Pre_Modeling_Gate/`  
**Certified Datasets Evaluated:**
- `Purified_Corn_Belt_Historical_Dataset.csv` (16,390 observations $\times$ 83 features)
- `Purified_Corn_Belt_Future_2026_Dataset.csv` (814 county observations $\times$ 82 features)

---

## 1. Executive Summary & Objective

This audit establishes the rigorous, leak-free preprocessing, feature scaling, missing value imputation, and categorical encoding protocols required across all baseline, ablation, and advanced machine learning models in Phase 5.

Every data transformation must satisfy the **Strict Temporal Precedence Invariant**:
$$\mathcal{T}_{\text{fit}} \subseteq \{t \mid t \le 2019\}$$
Parameters for centering, scaling, quantile calculation, or median imputation must be computed **solely** on historical training observations ($t \le 2019$). Under no circumstances may validation ($2020 \le t \le 2022$), test ($2023 \le t \le 2025$), or prospective forecast ($t = 2026$) records be included in transformation fitting.

---

## 2. In-Depth Missingness Investigation: The Cook County, Illinois Phenomenon

### 2.1 Empirical Observation
During dataset verification across the 788 historically certified agricultural counties, a critical anomaly was isolated:
- **Cook County, Illinois (`fips_code = 17031`, Chicago Metropolitan Area)** contains **11 historical NASS corn yield records** between 2002 and 2021 (mean yield: 124.5 bu/acre; range: 88.0 to 163.0 bu/acre).
- However, all **24 SoilGrids physical and chemical soil properties** are **100.0% null (`NaN`)** for Cook County in both historical and 2026 datasets.
- Neighboring agricultural counties (DuPage `17043`, Will `17197`, Kane `17089`) have 100% complete SoilGrids data.

### 2.2 Root Cause Analysis
SoilGrids (ISRIC 250m global digital soil mapping) applies an urban, water, and artificial surface mask. The geographic spatial centroid of Cook County falls within the dense urban fabric of Chicago near Lake Michigan:
1. When spatial point extractions or small-buffer zonal centroids were sampled against the SoilGrids raster layers, the sampled coordinates fell onto masked "NoData" / water / urban pixels.
2. In contrast, NASS agricultural reporting for Cook County captures remaining peri-urban parcels in the southern and western periphery (e.g., Barrington, Schaumburg, Palos, and Bloom townships).

### 2.3 Scientific Implications: The 788 vs. 787 County Eligibility Fork
This finding cleanly resolves the historical disagreement in prior audit iterations regarding county eligibility:
- **Operational Policy A (788 Counties - Inclusive):**
  Include Cook County, relying on tree-based models' native missing value routing (e.g., XGBoost / LightGBM / CatBoost direction splitting on `NaN`).
- **Operational Policy B (787 Counties - Strict Biophysical Complete-Case):**
  Exclude Cook County from core biophysical training due to missing soil features, leaving exactly **787 strictly valid counties**.
- **Audit Decision:**
  For our benchmark suite:
  1. Tree ensembles natively ingest all **788 counties** (16,390 rows) to ensure complete alignment with the Phase 1 certified benchmarks.
  2. For linear models, kernel SVR, and neural networks, Cook County soil values are imputed using the **Illinois statewide training median** (fitted strictly on $t \le 2019$), accompanied by explicit tracking in the audit logs.

---

## 3. Missing Value Taxonomy and Imputation Hierarchy

Across the 83 columns in `Purified_Corn_Belt_Historical_Dataset.csv`, missing values fall into three distinct structural categories:

| Feature Family | Columns Affected | Nature of Missingness | Imputation Protocol for Linear / Neural Models | Tree-Based Model Protocol |
| :--- | :--- | :--- | :--- | :--- |
| **SoilGrids** | 24 soil physical/chemical properties | Spatial mask failure (Cook County, IL: 11 rows) | Median of state-level training observations ($t \le 2019$) | Native default split routing (`missing=np.nan`) |
| **RMA Indemnity** | `rma_indemnity_total_loss`, `rma_indemnity_drought_loss`, etc. | Structural zero (counties without crop failure payouts) | Imputed to `0.0`; explicit boolean indicator `rma_missing_indicator = 1` preserved | Native `0.0` with boolean indicator column |
| **Yield Lags** | `yield_lag_1yr`, `yield_lag_2yr`, `yield_roll_mean_5yr` | Boundary truncation ($2002$ has no $t-1$; $2002-2003$ has no $t-2$; $2002-2006$ has partial rolling) | Strict forward historical window; missing historical lags filled via county-specific long-term historical mean ($t \le 2019$) | Forward filled from authentic NASS records; zero look-ahead |
| **Climate (NOAA)** | 13 growing-season features (May–Sep) | $0.0\%$ missing across all 16,390 rows | None required | None required |

### Strict Anti-Leakage Imputation Implementation
$$\hat{\mu}_{\text{soil}, j}^{\text{train}} = \text{Median}\left(\{ x_{i,j} \mid t_i \le 2019 \land x_{i,j} \text{ is not NaN} \}\right)$$
This median vector $\hat{\mu}^{\text{train}}$ is serialized into `11_Independent_Model_Validation/02_Reproducibility/imputation_medians.json` and applied to Validation ($2020-2022$), Test ($2023-2025$), and Future ($2026$).

---

## 4. Feature Scaling and Normalization Protocols

Tree-based algorithms (Random Forest, Extra Trees, Gradient Boosting, XGBoost, LightGBM, CatBoost) are invariant to monotonic transformations of continuous features. However, Distance-, Gradient-, and Kernel-based models (Ridge, ElasticNet, SVR, MLP) are critically sensitive to feature magnitude and variance.

### 4.1 Scaling Method Selection
Given that agricultural weather extremes (e.g., severe precipitation deluges, drought heatwaves) and insurance losses exhibit pronounced right-skewness and heavy tails, standard Z-score standardization can compress extreme anomalies.
- **RobustScaler** (interquartile range centering):
  $$z = \frac{x - Q_2(x_{\text{train}})}{Q_3(x_{\text{train}}) - Q_1(x_{\text{train}})}$$
- **StandardScaler**:
  $$z = \frac{x - \bar{x}_{\text{train}}}{s_{\text{train}}}$$
- **Protocol:**
  In Phase 5, all parametric models use `StandardScaler` fitted strictly on $t \le 2019$ after median imputation. Features with zero variance in the training set are dropped.

---

## 5. Categorical Feature Encoding

The dataset contains the following non-numeric / geographic metadata columns:
1. `fips_code`: 5-digit integer county FIPS.
2. `county_name`: String county identifier.
3. `state_alpha`: 2-letter state abbreviation (`IA`, `IL`, `IN`, `MN`, `MO`, `NE`, `NE`, `OH`, `SD`, `WI`).
4. `state_fips`: 2-digit integer state code.

### 5.1 County Identifier Treatment
County FIPS represents 788 distinct entities.
- Direct integer encoding as a continuous feature is strictly prohibited (introduces false ordinal topology).
- Direct one-hot encoding introduces 788 sparse columns, causing severe tree fragmentation and high dimensionality.
- Target encoding of county FIPS carries severe out-of-fold leakage risks unless strictly nested in cross-validation loops.
- **Protocol:** County FIPS and county name are **excluded** from predictor feature sets $X$, acting strictly as index and group keys for spatial Leave-One-State-Out (LOSO) and temporal blocking.

### 5.2 State Identifier Treatment
State ANSI codes represent 9 macro-regional political and agronomic zones.
- For non-LOSO experiments, `state_alpha` is encoded using standard One-Hot Encoding (9 binary indicators, drop-first if linear model).
- For Leave-One-State-Out (LOSO) cross-validation, state indicators are **excluded** from $X$ to prevent target collapse during out-of-state evaluation.

---

## 6. Preprocessing Audit Sign-Off

This audit certifies that all preprocessing routines designed for Phase 5 strictly preserve temporal causality, isolate out-of-sample data, and provide complete reproducibility across all benchmark runs.

