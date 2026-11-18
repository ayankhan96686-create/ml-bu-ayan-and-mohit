# 📊 STATISTICAL VALIDITY REVIEW OF EXPLORATORY DATA ANALYSIS (EDA)

**Project:** Regional Maize/Corn Grain Yield Prediction (2002–2026)  
**Location:** `E:/ml do not open/Maize_Yield_Research_2002_2026/09_Independent_Validation/`  
**Topic:** Critical Statistical Review of Correlation Analyses, Clustering, and Autocorrelation  

---

## 1. CRITIQUE OF POOLED PEARSON CORRELATION IN `perform_eda.py`

In `07_Code/perform_eda.py`, bivariate correlations were computed across the 16,390 rows of `County_Year_Feature_Table.csv` using:
```python
corr_series = feat_df[numeric_cols].corr()["corn_yield_bu_acre"]
```
This naive calculation treats each row as an independent, identically distributed ($i.i.d.$) observation. In spatio-temporal agronomic panel data, this assumption is fundamentally violated in four ways:

### 1.1 State-Level Broadcast Clustering (Artificial Degrees of Freedom)
- **The Mechanism:** USDA NASS Crop Progress (e.g., `state_progress_silking_w30_pct`) and Crop Condition ratings (e.g., `state_cond_good_mean`) are **statewide survey statistics**.
- **The Defect:** In Illinois (102 counties) or Missouri (115 counties), all counties within the state receive the **exact same numerical value** for each year.
- **Statistical Consequence:** When 102 counties in Illinois receive the identical condition rating in year 2012, calculating correlation across all 16,390 rows treats this as 102 independent data points. In reality, there is only **1 independent state observation** per year ($N_{\text{eff}} \approx 9 \text{ states} \times 24 \text{ years} = 216$ independent points).
- **Impact:** Effective degrees of freedom are artificially inflated by up to **$75\times$**, drastically compressing standard errors and yielding spuriously high $t$-statistics and false p-values ($p < 0.0001$).

### 1.2 Repeated Measures (Panel Clustering by County)
- The dataset comprises the same 814 geographic counties observed repeatedly across 24 years.
- County unobserved heterogeneities (soil micro-topography, grower management practices, baseline drainage) induce strong **positive intra-county correlation**:
  $$\text{Corr}(\epsilon_{i, t}, \epsilon_{i, s}) > 0 \quad \text{for } t \neq s$$
- Standard OLS and Pearson correlation standard errors underestimate true uncertainty by $40\%\text{ to }60\%$.

### 1.3 Spatial Autocorrelation (Moran's I Violation)
- Weather systems and drought events are spatially correlated across multi-county synoptic scales. A drought impacting McLean County, IL simultaneously impacts neighboring Livingston, Ford, and Champaign counties.
- Spatial error autocorrelation violates the Gauss-Markov assumption of independent residuals.

### 1.4 Non-Stationary Technological Trend Confounding
- Over the 2002–2025 study window, technological advancements (hybrid genetic yield potential, seed treatments, variable-rate fertilizer) increased average Corn Belt yields by approximately **$+1.9\text{ bu/acre/year}$**.
- Any feature that exhibits a secular trend over time (e.g., rising atmospheric temperatures, increasing planting progress pace) will exhibit a **spurious correlation** with corn yield, even in the complete absence of a causal agronomic relationship.

---

## 2. STATISTICALLY RIGOROUS INFERENCE RECOMMENDATIONS

To produce peer-review-defensible statistical relationships, the project must implement the following inferential standards:

### 2.1 Two-Way Cluster-Robust Standard Errors (County + Year)
Standard errors must be clustered simultaneously across counties (to account for serial correlation within a county) and years (to account for spatial correlation across all counties in a given year), following Cameron, Gelbach, and Miller (2011):
$$\mathbf{V}_{\text{two-way}} = \mathbf{V}_{\text{county}} + \mathbf{V}_{\text{year}} - \mathbf{V}_{\text{county} \cap \text{year}}$$
This ensures that neither repeated measures nor nationwide drought shocks artificially inflate statistical significance.

### 2.2 Spatial-Temporal Block Bootstrapping
When evaluating confidence intervals for model $R^2$ and feature importance rankings:
- Standard random row bootstrapping leaks correlated temporal and spatial neighbors into both the bootstrap sample and the out-of-bag sample.
- **Requirement:** Implement a **Block Bootstrap** that resamples entire states or entire calendar years as coherent contiguous units:
  $$\mathcal{B}_k \sim \text{Resample}(\{\text{State}_1, \dots, \text{State}_9\}) \quad \text{or} \quad \mathcal{B}_t \sim \text{Resample}(\{\text{Year}_{2002}, \dots, \text{Year}_{2025}\})$$

### 2.3 Detrended Within-County Fixed Effects Correlations
Before reporting bivariate correlations:
1. De-mean each variable by its county-specific historical mean (within-transformation):
   $$\tilde{X}_{i, t} = X_{i, t} - \bar{X}_i$$
   $$\tilde{Y}_{i, t} = Y_{i, t} - \bar{Y}_i$$
2. Remove the common linear/spline time trend:
   $$Y_{i, t}^* = \tilde{Y}_{i, t} - \hat{\beta}_{\text{trend}} \times t$$
3. Compute the correlation between detrended climate anomalies ($X_{i, t}^*$) and detrended yield shocks ($Y_{i, t}^*$).
This isolates the true causal weather response from static soil advantages and technological trends.
