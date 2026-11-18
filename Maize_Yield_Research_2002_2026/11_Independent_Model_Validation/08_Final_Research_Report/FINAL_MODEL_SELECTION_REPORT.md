# FINAL MODEL SELECTION REPORT: MULTI-CRITERIA DECISION FRAMEWORK

**Project:** County-Level Maize Grain Yield Prediction Research Project (2002–2026)  
**Author:** Independent Statistical & Machine Learning Audit Team  
**Evaluation Scope:** 11 Benchmarked Algorithms Evaluated across Validation ($2020–2022$) and Out-of-Sample Test ($2023–2025$) Splits  

---

## 1. Executive Summary & Selection Decision

Selecting an operational model for large-scale agricultural yield forecasting and high-impact scientific publication requires balancing four distinct, often competing criteria:
1. **Out-of-Sample Generalization Accuracy (RMSE, MAE, $R^2$)**
2. **Extreme Error Boundedness (Max Error and Tail Risk)**
3. **Biophysical Interpretability and Causal Plausibility**
4. **Computational Efficiency and Operational Robustness (Native Missing Value Routing)**

### The Multi-Tier Selection Decision:
- **Primary Operational Reference Model:** **XGBoost Regressor (Set D: Integrated Soil, Growing-Season Climate & Yield Memory)**  
  *Justification:* While `HistGradientBoosting` and `LightGBM` slightly edged out XGBoost on raw test RMSE ($17.24$ and $17.33$ vs. $17.70$ bu/acre), XGBoost provides the best balance of exact baseline comparability, native NaN handling (crucial for Cook County, IL), complete compatibility with TreeSHAP exact tree path tracing, and robust spatial generalization across all 9 Corn Belt states.
- **High-Speed Operational Ensemble Partner:** **LightGBM Regressor**  
  *Justification:* Achieved the lowest Test MAE ($13.20$ bu/acre) with ultra-fast training ($0.35$ seconds), making it the premier candidate for rapid spatial bootstrap iterations and ensemble averaging.
- **Extreme Error Mitigation Partner:** **Multi-Layer Perceptron (`MLPRegressor`)**  
  *Justification:* Achieved the lowest Maximum Absolute Error on the test set ($71.99$ bu/acre vs. $\sim 100$ bu/acre for tree models), demonstrating that neural representations smooth out catastrophic tree leaf mispredictions during extreme outlier weather years.

---

## 2. Multi-Criteria Evaluation Matrix

All models evaluated on the standardized 40-feature Integrated Set D (`ADVANCED_ML_MODEL_COMPARISON.csv`):

| Model Architecture | Test RMSE (bu/ac) | Test MAE (bu/ac) | Test $R^2$ | Max Error (bu/ac) | Training Time (s) | Native NaN Handling | Interpretability (TreeSHAP) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **HistGradientBoosting** | **17.24** | 13.25 | **0.6567** | 95.36 | 2.92 | Yes | Partial |
| **MLPRegressor** | 17.26 | 13.57 | 0.6561 | **71.99** | 13.28 | No (Requires Imputation) | Kernel SHAP Only |
| **LightGBM** | 17.33 | **13.20** | 0.6533 | 101.42 | **0.35** | Yes | Yes (Fast) |
| **XGBoost (Certified)** | 17.70 | 13.69 | 0.6383 | 99.28 | 2.32 | **Yes (Robust)** | **Yes (Gold Standard)** |
| **CatBoost** | 17.85 | 13.85 | 0.6321 | 101.86 | 2.27 | Yes | Yes |
| **GradientBoosting** | 18.22 | 14.14 | 0.6167 | 100.77 | 19.72 | No | TreeSHAP |
| **SVR (RBF Kernel)** | 18.62 | 14.87 | 0.5996 | 78.30 | 6.28 | No | Kernel SHAP Only |
| **RandomForest** | 20.21 | 16.36 | 0.5285 | 103.99 | 1.02 | No | TreeSHAP |
| **ElasticNet** | 20.30 | 16.22 | 0.5243 | 94.39 | 0.93 | No | Linear Coefficients |
| **Ridge Regression** | 20.34 | 16.22 | 0.5223 | 93.43 | 0.01 | No | Linear Coefficients |
| **ExtraTrees** | 22.90 | 19.55 | 0.3946 | 95.07 | 0.34 | No | TreeSHAP |

---

## 3. Deep-Dive Tradeoff Analysis

### 3.1 Trees vs. Neural Networks vs. Linear Models
1. **The Linear Ceiling ($R^2 \approx 0.52$):** Linear models (Ridge, ElasticNet) cannot capture threshold-governed physiological crop reactions (such as pollen sterilisation above 86°F or soil water saturation). They simply approximate the 5-year county rolling average, resulting in high RMSE ($20.3$ bu/acre).
2. **The Power of Tree Ensembles:** Tree boosting algorithms capture complex split interactions:
   - Split 1: `yield_roll_mean_5yr > 175` (high productivity county)
   - Split 2: `noaa_tmax_jul_f > 86.5` (extreme pollination heat stress)
   - Split 3: `soil_bdod_0_5cm > 1.4` (compacted soil with poor water holding capacity)
   This compound partitioning drops test RMSE to **$17.2$–$17.7$ bu/acre**.
3. **The Neural Network Anomaly:** The MLPRegressor performed remarkably well ($17.26$ bu/acre RMSE, $R^2 = 0.6561$), outperforming standard Random Forest and matching LightGBM. Furthermore, its smooth continuous sigmoidal/ReLU transitions prevent the step-function extreme errors of tree models, yielding the lowest maximum error ($71.99$ bu/acre). However, MLPs require strict feature scaling, median imputation, and are substantially more difficult to explain to agronomic stakeholders.

### 3.2 Tree Boosting Showdown: XGBoost vs. LightGBM vs. HistGradientBoosting
- **LightGBM:** Fast (0.35s), lowest MAE (13.20 bu/acre), excellent candidate for massive spatial bootstrapping.
- **HistGradientBoosting:** Best test RMSE (17.24 bu/acre), built directly into scikit-learn with zero external C++ library dependencies.
- **XGBoost:** The industry benchmark. It offers mathematically exact second-order Taylor expansion tree splitting, explicit regularizers (`reg_alpha`, `reg_lambda`), and seamless, mature TreeSHAP integration.

---

## 4. Final Recommendation & Deployment Architecture

For scientific publication and operational forecasting in the 2026 crop season:
1. **Primary Model for Scientific Manuscript:** **XGBoost Regressor (Set D)**. Its exact correspondence to the Phase 1 certified baseline, established feature importance profile, and 100% reproducible metrics make it the primary anchor.
2. **Operational Ensemble (Recommended Production Enhancement):** A simple equal-weighted ensemble of **XGBoost + LightGBM + MLPRegressor**:
   $$\hat{y}_{\text{ensemble}} = \frac{1}{3}\hat{y}_{\text{XGBoost}} + \frac{1}{3}\hat{y}_{\text{LightGBM}} + \frac{1}{3}\hat{y}_{\text{MLP}}$$
   This tripartite ensemble combines the split-based biophysical logic of gradient boosted trees with the continuous bounding of neural networks, smoothing out local leaf outliers while preserving overall Corn Belt calibration.

