# COMPLETE MACHINE LEARNING METHODOLOGY: MAIZE YIELD PREDICTION FRAMEWORK (2002–2026)

**Project Title:** Advanced County-Level Maize (*Zea mays* L.) Grain Yield Prediction Framework  
**Authors:** Ayan Khan and Mohit  
**Active Repository Commit:** `b00a8367c2173be6974004e99d2a3f43ec833d15`  
**Study Horizon:** 2002–2026 (Historical: 2002–2025; Prospective: 2026)  
**Total Historical Observations:** 16,390 certified county-years  

---

## 1. Train-Test Splitting Paradigms

To establish complete scientific transparency and resolve conflicting reporting conventions in agricultural ML literature, the framework implements two distinct, rigorously audited train-test splitting architectures:

### 1.1 Primary Pooled Stratified 80:20 Split (Authoritative Phase 19 Benchmark)
- **Design:** Stratified random sampling across all 24 historical calendar years (2002–2025) using fixed `random_state=42`.
- **Training Partition (80.0%):** **13,112 county-year observations**.
- **Testing Holdout (20.0%):** **3,278 county-year observations**.
- **Year Stratification Guarantee:** Exactly 80% of observations from each calendar year (2002 through 2025) are allocated to training, and exactly 20% to testing.
- **Partition Independence:** Exactly zero row overlap. Verified in `TRAIN_TEST_OBSERVATION_IDS_POOLED.csv`.
- **Scientific Purpose:** Evaluates spatial interpolation across observed regional climate regimes; serves as direct benchmark against conventional literature cross-validation studies.

### 1.2 Pure Chronological Out-of-Time 80:20 Split (Phase 18 Robustness Benchmark)
- **Design:** Strict historical block split based on calendar year boundaries.
- **Training Period (2002–2020, 19 Years, 79.17% of timeline):** **12,976 county-year observations**.
- **Testing Period (2021–2025, 5 Years, 20.83% of timeline):** **3,414 county-year observations**.
- **Scientific Purpose:** Evaluates genuine temporal forecasting into completely unobserved future macro-climate regimes, exposing model sensitivity to unseen weather distributions.

---

## 2. In-Sample Chronological Validation Framework

To prevent data snooping and ensure that hyperparameter tuning and model selection remain completely insulated from the test holdout, an expanding chronological validation framework was implemented **strictly inside the 80% training set** ($N = 13,112$):

```
Fold 1: [2002–2013 Train (N=6,561)] ──► [2014–2016 Val (N=1,617)]
Fold 2: [2002–2016 Train (N=8,178)] ──► [2017–2019 Val (N=1,471)]
Fold 3: [2002–2019 Train (N=9,649)] ──► [2020–2022 Val (N=1,626)]
Fold 4: [2002–2022 Train (N=11,275)] ──► [2023–2025 Val (N=1,359)]
```

### Validation Invariants:
1. **Expanding Historical Window:** Training sets expand forward in time, mirroring real-world operational forecasting where historical data accumulates over decades.
2. **Strict Future Target Isolation:** No validation year ever precedes or overlaps with its training block.
3. **Locked Test Partition:** The 3,278 test observations remain completely untouched during all validation cycles.

---

## 3. Comprehensive Machine Learning Algorithm Suite

The project benchmarked 17 distinct machine learning algorithms spanning five algorithmic families:

### 3.1 Linear & Regularized Parametric Models
1. **Ordinary Least Squares (OLS) Linear Regression:** Baseline unpenalized linear model; vulnerable to severe multicollinearity among collinear climate variables.
2. **Ridge Regression (L2 Regularization):** Adds penalty $\alpha \sum \beta_j^2$. Shrinks collinear coefficients, stabilizing predictions across weather anomalies.
3. **Lasso Regression (L1 Regularization):** Adds penalty $\lambda \sum |\beta_j|$. Enforces feature sparsity by zeroing out non-informative coefficients.
4. **ElasticNet:** Combines L1 and L2 penalties ($r \lambda \|\beta\|_1 + \frac{1-r}{2} \lambda \|\beta\|_2^2$). Excels in handling groups of correlated environmental predictors.
5. **Huber Regressor:** Robust linear regression with hybrid loss (squared loss for small residuals, absolute loss for large residuals), limiting sensitivity to extreme outliers.
6. **Support Vector Regression (LinearSVR):** Epsilon-insensitive loss function with linear kernel; constructs robust hyperplane within margin $\epsilon$.

### 3.2 Decision Trees & Bagging Ensembles
7. **Decision Tree Regressor:** Non-parametric recursive binary partitioning; establishes unconstrained non-linear baseline.
8. **Random Forest Regressor:** Bagged ensemble of 300 de-correlated trees with bootstrap aggregation and random feature subsets (`max_features='sqrt'`).
9. **Extra Trees Regressor (Extremely Randomized Trees):** Randomizes split thresholds as well as feature subsets, reducing variance and computational overhead.

### 3.3 Gradient Boosted Decision Trees (GBDT)
10. **AdaBoost Regressor:** Sequential boosting with adaptive sample weighting.
11. **Gradient Boosting Regressor (Scikit-Learn GBRT):** Classic gradient boosting optimizing Mean Squared Error.
12. **Histogram-based Gradient Boosting (HistGradientBoosting):** Bins continuous features into 256 integer bins, dramatically accelerating split finding.
13. **LightGBM Regressor (Microsoft):** Leaf-wise (best-first) tree growth with Gradient-based One-Side Sampling (GOSS) and Exclusive Feature Bundling (EFB). Fast, highly accurate.
14. **XGBoost Regressor (DMLC):** Depth-wise tree growth with exact second-order Taylor expansion gradients, column subsampling, and shrinkage. **Champion Model ($R^2 = 0.8986$)**.
15. **CatBoost Regressor (Yandex):** Symmetric (oblivious) decision trees with ordered boosting; highly resistant to overfitting on tabular features.

### 3.4 Blending & Stacking Meta-Learners
16. **Calibrated Quad-Tree Blend:** Weighted linear convex combination of the top 4 gradient boosted architectures (35% LightGBM + 35% XGBoost + 15% CatBoost + 15% HistGB).
17. **Ridge Stacking Meta-Learner:** Two-layer stacking ensemble. Level-1 base models generate out-of-fold predictions across chronological validation folds. Level-2 Ridge regression (alpha=5.0) learns optimal non-negative meta-weights, preventing meta-overfitting.

---

## 4. Feature Selection & Ablation Methodology

To evaluate feature redundancy and prevent over-parameterization across the 117 available predictors, a 5-tier ablation benchmark was conducted on identical training/testing folds:

| Configuration ID | Configuration Name | Predictor Count | Feature Composition | Test $R^2$ | Test RMSE (bu/ac) | Test MAE (bu/ac) |
| :---: | :--- | :---: | :--- | :---: | :---: | :---: |
| **Config A** | Climate & Soil Only | 42 | PRISM/NOAA weather + SoilGrids physical/chemical profiles | 0.8124 | 15.88 | 12.01 |
| **Config B** | Climate + Soil + Historical Lags | 54 | Config A + Autoregressive yield lags ($y_{t-1}, y_{t-2}$) | 0.8641 | 13.52 | 10.24 |
| **Config C** | + Advanced Agro-Stress Indices | 78 | Config B + Non-linear EDD, VPD, Compound Heat-Drought | 0.8812 | 12.63 | 9.48 |
| **Config D** | **Permutation-Selected Top 45** | **45** | Top 45 features ranked by out-of-fold permutation importance | **0.8885** | **12.24** | **9.18** |
| **Config E** | Master Multimodal Feature Set | 117 | Full integrated feature manifest (All 6 families) | **0.8904** | **12.13** | **9.08** |

### Key Architectural Finding:
**Config D (45 features)** captures **99.8%** of the explanatory power of the full 117-feature set while reducing feature dimensionality by **61.5%**. This drastically reduces inference latency, eliminates collinear noise, and minimizes data acquisition costs.

---

## 5. Hyperparameter Optimization Protocol

### 5.1 Optimization Engine & Search Space
- **Framework:** Optuna 5.0.0 utilizing the Tree-structured Parzen Estimator (TPE, `seed=42`).
- **Trial Budget:** **120 trials** across the primary candidate architectures.
- **Objective Function:** Minimize the unweighted Mean Validation RMSE across the 4 in-sample chronological folds:
  $$\mathcal{L}(\theta) = \frac{1}{4} \sum_{k=1}^4 \text{RMSE}_{\text{val}, k}(\theta)$$

### 5.2 Certified Optimal Hyperparameters (`BEST_HYPERPARAMETERS.json`)

#### Optimized XGBoost Regressor (Rank 1 Champion):
- `n_estimators`: 450
- `learning_rate`: 0.0284
- `max_depth`: 6
- `subsample`: 0.824
- `colsample_bytree`: 0.768
- `reg_alpha` (L1): 0.142
- `reg_lambda` (L2): 2.185
- `min_child_weight`: 5

#### Optimized LightGBM Regressor (Rank 2 Champion):
- `n_estimators`: 500
- `learning_rate`: 0.0245
- `num_leaves`: 38
- `max_depth`: 7
- `subsample`: 0.812
- `colsample_bytree`: 0.745
- `reg_alpha`: 0.085
- `reg_lambda`: 1.842
- `min_child_samples`: 25

#### Ridge Stacking Meta-Learner (Level-2):
- Base Learners: Tuned LightGBM, Tuned XGBoost, Tuned CatBoost, Tuned HistGB
- Meta-Model: Ridge Regression (`alpha=5.0`, `fit_intercept=True`)
- Learned Meta-Weights: LightGBM (0.384), XGBoost (0.412), CatBoost (0.118), HistGB (0.092)

---

## 6. Evaluation Metrics & Tail Stress Assessment

Model accuracy is evaluated across multiple statistical and agronomic dimensions:
1. **Coefficient of Determination ($R^2$):** Proportion of variance in observed yields explained by the model:
   $$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$
2. **Root Mean Squared Error (RMSE):** Standard deviation of residuals (penalizes large outlier errors):
   $$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2}$$
3. **Mean Absolute Error (MAE):** Average magnitude of absolute errors (robust linear penalty):
   $$\text{MAE} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$$
4. **Mean Absolute Percentage Error (MAPE):** Relative error percentage (filtered for $y_i > 0$):
   $$\text{MAPE} = \frac{100\%}{N} \sum_{i=1}^N \left| \frac{y_i - \hat{y}_i}{y_i} \right|$$
5. **Mean Prediction Bias:** Signed systematic over- or under-prediction:
   $$\text{Bias} = \frac{1}{N} \sum_{i=1}^N (\hat{y}_i - y_i)$$
6. **Severe Drought Tail RMSE & Bias ($y_i < 120\text{ bu/ac}$):** Targeted audit on acute yield collapse observations ($N=468$ in test holdout) to evaluate model performance under severe climate distress.

---

## 7. Data Leakage Prevention (8-Point Automated Audit)

Every data transformation in the modeling pipeline was subjected to automated programmatic verification in `audit_data_leakage.py` (all 8 checks passed):
1. **Train/Test ID Disjointness:** Verified zero overlapping `(county_fips, year)` tuples between training and testing sets.
2. **Validation Fold Temporal Disjointness:** Verified zero validation records appear in corresponding training folds.
3. **Imputation Parameter Containment:** Missing value medians fitted strictly on training subsets; test sets transformed using frozen training medians.
4. **Feature Scaling Containment:** Standard scalers fitted strictly on training subsets.
5. **Autoregressive Lag Directionality:** Enforced $t - k \ge 1$; verified zero contemporaneous yield $y_{c,t}$ appears in any predictor column.
6. **Seasonal Cutoff Enforcement:** Verified all weather variables terminate on or before September 30 (zero post-harvest weather leakage).
7. **Own-Row Target Isolation:** Verified target column `corn_yield_bu_acre` is excluded from feature matrices.
8. **2026 Prospective Quarantine:** Verified year 2026 records are 100% excluded from training loss functions and test scorecards.

---

## 8. Reproducibility Guarantee

- **Hardware & OS:** Windows Server / 11 AMD64, multi-core x86_64 architecture.
- **Software Versions:** Python 3.14.0, NumPy 2.2.6, Pandas 2.2.3, Scikit-Learn 1.6.1, LightGBM 4.7.0, XGBoost 3.4.1, CatBoost 1.2.10, Optuna 5.0.0.
- **Deterministic Seeding:** `random_state=42` and `PYTHONHASHSEED=42` enforced across all scripts.
- **Checksum Verification:** Master dataset SHA-256 (`Historical_Features_Phase12.csv`: `1ba70a5996...`) validated before every training cycle.
