# Machine Learning Readiness Assessment & Validation Strategy

**Project:** Maize/Corn Grain Yield Prediction Using Machine Learning (2002–2025)  
**Root Directory:** `E:/ml do not open/Maize_Yield_Research_2002_2026/`  
**Audit Date:** 2026-11-10  
**Auditor:** Senior Machine Learning Engineer & Research Reviewer  

---

## 1. Executive Verdict on ML Readiness

| Dimension | Readiness Status | Evidence & Metrics | Audit Notes |
|---|---|---|---|
| **Target Variable Quality** | **READY (High Quality)** | $N = 16,390$, Mean: 157.45 bu/ac, Range: 0.0–253.6 bu/ac. Zero missing target values. | Authentic USDA NASS observations; 0 synthetic labels. |
| **Feature Completeness** | **CONDITIONALLY READY** | 78 ML predictors spanning climate, soil, insurance, and condition. | Ready for baseline modeling; satellite NDVI and 2023 PRISM ppt require gap-fills. |
| **Data Leakage Quarantine** | **READY (Verified Clean)** | 0 post-harvest variables in `County_Year_Feature_Table.csv`. 2026 strictly unlabeled. | Complete separation of predictive features from post-harvest indemnities/progress. |
| **Sample Size & Power** | **READY (High Statistical Power)** | 16,390 historical county-years across 788 counties and 24 years. | Substantial statistical power for XGBoost, LightGBM, Random Forest, and TabNet. |
| **Geographic Coverage** | **READY (High Consistency)** | 9 core Corn Belt states (IA, IL, IN, MN, MO, NE, OH, SD, WI). | Standardized 5-digit county FIPS codes aligned across all sources. |
| **Temporal Consistency** | **READY (2002–2025 Continuous)** | 24 continuous calendar years of historical training/eval data. | Full temporal span encompassing major climate shock years (2012 drought, 2019 floods). |

**Overall Assessment:** **CONDITIONALLY READY FOR SUPERVISED REGRESSION.**  
The dataset is immediately viable for training gradient boosted trees (LightGBM, XGBoost, CatBoost), Random Forests, and Penalized Regressions (Ridge, Lasso, ElasticNet). High-end deep learning or remote-sensing multi-modal architectures should await satellite vegetation index acquisition.

---

## 2. Target Variable Characteristics & Suitability

- **Target Column:** `corn_yield_bu_acre`
- **Total Valid Observations:** 16,390
- **Summary Statistics:**
  - Mean: $157.45$ bu/acre
  - Standard Deviation: $34.18$ bu/acre
  - Median: $161.40$ bu/acre
  - Interquartile Range (IQR): $136.00$ to $182.20$ bu/acre
  - Minimum: $0.00$ bu/acre (Harding, Lawrence, Todd counties, SD)
  - Maximum: $253.60$ bu/acre (Macon County, IL, 2021)
  - Skewness: $-0.684$ (Moderate left skew reflecting acute drought failure years)
  - Kurtosis: $0.812$ (Platykurtic / near-normal)

### Target Distribution Considerations:
The target variable is well-behaved, continuous, and suitable for standard mean squared error (MSE), root mean squared error (RMSE), and mean absolute error (MAE) loss functions. Huber or Smooth L1 loss is recommended to provide robustness against severe local crop failure outliers without distorting regression gradients.

---

## 3. Recommended Validation & Partitioning Design

### Why Random K-Fold Shuffling Must NOT Be Used:
Randomly shuffling county-years into train and test folds introduces **severe temporal autocorrelation leakage**. Weather patterns are spatially widespread; if neighboring counties in the 2012 drought year are split randomly between training and testing, the model merely memorizes the 2012 weather shock rather than learning generalizable biophysical relationships.

### Recommended Evaluation Protocol: Time-Series Blocked Cross-Validation

```
+----------------------------------------------------------------------------------------------------+
| RECOMMENDED MACHINE LEARNING TIME-SERIES PARTITIONS                                               |
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|  [ TRAINING SET: 2002 – 2019 (18 Years) ]                                                          |
|  Records: 12,284 county-years (75.0% of labeled data)                                              |
|  Purpose: Model fitting, internal feature selection, and algorithm parameter training             |
|                                                                                                    |
|  [ VALIDATION SET: 2020 – 2022 (3 Years) ]                                                         |
|  Records: 2,174 county-years (13.3% of labeled data)                                               |
|  Purpose: Hyperparameter optimization, early stopping, and ensemble weight tuning                |
|                                                                                                    |
|  [ HELD-OUT BENCHMARK TEST SET: 2023 – 2025 (3 Years) ]                                            |
|  Records: 1,932 county-years (11.8% of labeled data)                                               |
|  Purpose: Final unbiased academic benchmark evaluation. NEVER touched during tuning!               |
|                                                                                                    |
|  [ UNLABELED PROSPECTIVE SET: 2026 (1 Future Year) ]                                                |
|  Records: 814 counties (100% of study area)                                                        |
|  Purpose: Prospective blind forecasting for the 2026 crop harvest                                  |
|                                                                                                    |
+----------------------------------------------------------------------------------------------------+
```

### Expanding-Window Time-Series Cross-Validation:
For hyperparameter optimization within the training partition (2002–2019), use **Expanding-Window Cross-Validation**:
- Fold 1: Train on 2002–2010 $\rightarrow$ Evaluate on 2011
- Fold 2: Train on 2002–2011 $\rightarrow$ Evaluate on 2012 (Drought benchmark)
- Fold 3: Train on 2002–2012 $\rightarrow$ Evaluate on 2013
- ...
- Fold 9: Train on 2002–2018 $\rightarrow$ Evaluate on 2019

---

## 4. Modeling Recommendations & Baseline Architectures

1. **Primary Recommended Algorithms:**
   - **LightGBM / XGBoost Regressor:** Best-in-class performance for tabular tabular datasets with heterogeneous feature scales (climate, soil, insurance). Native handling of missing values (e.g. 2023 PRISM gap).
   - **Random Forest Regressor:** Excellent non-linear benchmark; resistant to overfitting on high-dimensional collinear features.
   - **Ridge / ElasticNet with Splines:** Biophysical linear baseline. Splines capture non-linear temperature response functions ($T_{\max} > 86^\circ\text{F}$).
2. **Evaluation Metrics:**
   - Primary: Root Mean Squared Error ($\text{RMSE}$, in bu/acre).
   - Primary: Mean Absolute Error ($\text{MAE}$, in bu/acre).
   - Relative: Normalized RMSE ($\text{nRMSE} = \frac{\text{RMSE}}{\bar{y}} \times 100\%$).
   - Explained Variance: Coefficient of Determination ($R^2$).
   - Operational Benchmark: Percentage of counties predicted within $\pm 10\%$ of actual harvest yield.

