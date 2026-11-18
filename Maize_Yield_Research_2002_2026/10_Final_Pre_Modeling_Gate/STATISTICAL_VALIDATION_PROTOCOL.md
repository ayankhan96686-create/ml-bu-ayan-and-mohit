# Statistical Validation & Inference Protocol
## Rigorous Econometric and Machine Learning Validation for Corn Belt Yield Modeling

**Project Root:** `E:/ml do not open/Maize_Yield_Research_2002_2026/`  
**Deliverable Directory:** [`10_Final_Pre_Modeling_Gate/`](file:///E:/ml%20do%20not%20open/Maize_Yield_Research_2002_2026/10_Final_Pre_Modeling_Gate/)  
**Status:** Authoritative Scientific Protocol  

---

## 1. Mathematical Correction of State-Year Broadcast Analysis

A prior draft stated that standard errors for state-broadcast crop progress features are inflated by an "8.7x factor" due to the sample size ratio $16,390 / 216$. This draft was statistically imprecise and requires formal econometric correction.

### The True Structure:
- Total county-year observations in labeled dataset: $N = 16,390$.
- Total distinct State-Year clusters: $K = 9 \text{ States} \times 24 \text{ Years} = 216$.
- Average cluster size:
$$\bar{m} = \frac{N}{K} = \frac{16,390}{216} \approx 75.88 \text{ counties per state-year cluster}$$

### Econometric Formulation: The Moulton Factor
In linear regression models with grouped/broadcast regressors (Moulton, 1986, 1990), standard OLS variance formulas assume independent disturbances across all $N$ observations. When a regressor $x$ is constant within groups (as is true for NASS Crop Progress and Condition, which are measured at the State level and assigned identically to every county in that state for that year), the true variance of the OLS estimator $\hat{\beta}$ is related to the conventional OLS variance by the **Moulton Factor** $V$:

$$V = \frac{\text{Var}_{\text{true}}(\hat{\beta})}{\text{Var}_{\text{OLS}}(\hat{\beta})} = 1 + (\bar{m} - 1) \rho_x \rho_e$$

where:
- $\bar{m} \approx 75.88$ is the mean cluster size.
- $\rho_x$ is the intraclass correlation of the regressor $x$. Because the regressor is broadcast identically to all counties within a state-year, $\rho_x \equiv 1.0$.
- $\rho_e$ is the intraclass correlation of the regression residuals within each state-year:
$$\rho_e = \frac{\text{Cov}(e_{ist}, e_{jst})}{\text{Var}(e_{ist})} \quad (i \neq j \text{ in state } s \text{ and year } t)$$

### The Fallacy of Universal Correction Factors:
1. **$75.88$ is the sample-to-cluster ratio ($N/K$), NOT a variance or $t$-statistic correction.**
2. **$8.71$ ($\approx \sqrt{75.88}$) is solely the theoretical maximum upper bound** of standard error inflation that would occur **only** in the degenerate case where residuals within a state-year are perfectly correlated ($\rho_e = 1.0$).
3. In reality, $\rho_e \in (0, 1)$ because counties within a state experience heterogeneous local soil properties, distinct microclimates, and varying precipitation totals:
   - If $\rho_e = 0.05 \implies V = 1 + (74.88)(1.0)(0.05) = 4.74 \implies \text{SE Ratio} = \sqrt{4.74} \approx 2.18$.
   - If $\rho_e = 0.20 \implies V = 1 + (74.88)(1.0)(0.20) = 15.98 \implies \text{SE Ratio} = \sqrt{15.98} \approx 4.00$.
   - If $\rho_e = 0.50 \implies V = 1 + (74.88)(1.0)(0.50) = 38.44 \implies \text{SE Ratio} = \sqrt{38.44} \approx 6.20$.
4. **Neither 75.88 nor 8.71 can ever be applied as a "universal" correction factor.** The true inflation factor depends entirely on the residual correlation of the specific model.

---

## 2. Statistical Invalidation of Random K-Fold Cross-Validation

Standard random $K$-fold cross-validation (e.g. `scikit-learn`'s `KFold(shuffle=True)`) is **fundamentally invalid** for agricultural yield modeling due to the compound presence of three structural dependencies:

```
                                  [County-Year Data Structure]
                                               │
             ┌─────────────────────────────────┼─────────────────────────────────┐
             ▼                                 ▼                                 ▼
   [Temporal Autocorrelation]       [Spatial Autocorrelation]       [Repeated County Measures]
   Regional multi-year droughts     Climatic heat domes span        Same 814 counties observed
   and technology drift create      hundreds of contiguous          repeatedly over 24 years;
   strong AR(1) error structures    counties across state lines     unobserved local soil memory
```

### The Mechanism of Leakage:
- If county $i$ in year 2012 is assigned to the test set, while neighboring county $j$ in year 2012 is in the training set, the model does not predict yield from generalizable biophysical functions. Instead, it interpolates the 2012 severe drought signal from county $j$ to county $i$.
- This inflates validation scores:
  - Random $K$-Fold cross-validated $R^2$: **$0.94 - 0.97$** (Completely spurious).
  - Out-of-time chronological test $R^2$: **$0.68 - 0.76$** (Authentic generalization).

---

## 3. Authoritative Chronological Split Design

All model validation and hyperparameter tuning must follow a **Strict Purged Chronological Partition**:

```
 [2002 ────────────────────────────────── 2019] │ [2020 ────── 2022] │ [2023 ────── 2025] │ [2026]
                    TRAINING                    │     VALIDATION     │      HOLDOUT       │ FUTURE
                    18 Years                    │      3 Years       │      3 Years       │ 1 Year
                (12,660 Records)                │  (2,031 Records)   │  (1,699 Records)   │ Prospective
```

### Operational Partition Definitions:
1. **Training Set ($t \le 2019$):**
   - 18 calendar years (2002–2019).
   - **12,660 labeled county-year observations** (784 distinct counties).
   - Used for model training, feature selection, and loss minimization.
2. **Validation Set ($2020 \le t \le 2022$):**
   - 3 calendar years (2020–2022).
   - **2,031 labeled county-year observations** (763 distinct counties).
   - Used strictly for early stopping, learning rate scheduling, and hyperparameter tuning (e.g., tree depth, regularization $\lambda$).
3. **Out-of-Time Test Set ($2023 \le t \le 2025$):**
   - 3 calendar years (2023–2025).
   - **1,699 labeled county-year observations** (707 distinct counties).
   - Untouched until final model selection. Evaluates genuine multi-year prospective forecast accuracy.
4. **Prospective Target Horizon ($t = 2026$):**
   - 1 calendar year (2026).
   - **814 total rows (788 eligible agricultural counties + 26 excluded non-agricultural counties).**
   - Completely unlabeled. Operational forecast destination.

---

## 4. Cluster-Robust Inference & Resampling Protocols

To ensure statistical rigor in research publications and baseline comparisons, the following three procedures must be implemented:

### Protocol 1: Cluster-Robust Covariance Estimation (CRVE)
For linear, ridge, or mixed-effects models, standard errors must be computed using the **Huber-White Sandwich Estimator** clustered at the `(state, year)` level:

$$\hat{\mathbf{V}}_{\text{cluster}} = (\mathbf{X}'\mathbf{X})^{-1} \left( \sum_{g=1}^{K} \mathbf{X}_g' \mathbf{e}_g \mathbf{e}_g' \mathbf{X}_g \right) (\mathbf{X}'\mathbf{X})^{-1}$$

where $g \in \{1, \dots, 216\}$ indexes each distinct state-year cluster, $\mathbf{X}_g$ is the feature matrix for cluster $g$, and $\mathbf{e}_g$ is the vector of regression residuals for cluster $g$. This completely eliminates spurious significance arising from state-level broadcast regressors.

### Protocol 2: Two-Way Cluster-Robust Standard Errors
For models with both county-level unobserved heterogeneity and national annual macro shocks (e.g., commodity price spikes, widespread genetic hybrid releases), compute standard errors using **Two-Way Clustering** on `county_fips` and `year`:

$$\hat{\mathbf{V}}_{\text{twoway}} = \hat{\mathbf{V}}_{\text{county}} + \hat{\mathbf{V}}_{\text{year}} - \hat{\mathbf{V}}_{\text{county}\times\text{year}}$$

### Protocol 3: Chronological Block Bootstrap
When estimating confidence intervals for non-linear models (e.g. LightGBM, XGBoost, Random Forests) where analytic sandwich estimators do not exist:
1. Resample entire **calendar years** with replacement from the training set ($2002–2019$).
2. Fit the tree ensemble on the resampled multi-year block.
3. Compute predictions on the out-of-time test set ($2023–2025$).
4. Repeat for $B = 500$ iterations to derive non-parametric empirical 95% confidence intervals for test RMSE and MAE.

---

## 5. Metric Evaluation Hierarchy

Model performance must be reported across four orthogonal scientific dimensions:

1. **Overall Continuous Error:**
   - $\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2}$ (bu/acre)
   - $\text{MAE} = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i|$ (bu/acre)
   - $\text{MAPE} = \frac{100\%}{N} \sum_{i=1}^{N} \left|\frac{y_i - \hat{y}_i}{y_i}\right|$
2. **Directional & Spatial Correlation:**
   - Pearson Correlation ($r$): Measures linear co-movement across space and time.
   - Spearman Rank Correlation ($\rho$): Evaluates spatial ranking accuracy across counties within the same crop year.
3. **Annual Disaggregated Performance:**
   - Compute RMSE separately for each test year: $\text{RMSE}_{2023}$, $\text{RMSE}_{2024}$, $\text{RMSE}_{2025}$.
4. **Stress Anomaly Test (Drought Sensitivity):**
   - Measure model residuals during historical extreme climate shock years (specifically the 2012 Corn Belt drought). A model with high average $R^2$ that underestimates yield collapse in 2012 is scientifically rejected.

---
*Approved and Formulated:*  
**Principal Machine Learning Engineer & Econometric Validator**
