# CONSOLIDATED MODEL EVALUATION TABLE (80:20 CHRONOLOGICAL TRAIN-TEST SPLIT)

### Benchmark Architecture & Scope

- **Dataset:** 9 US Corn Belt States (814 Counties, 16,390 Historical Observations)
- **Training Period (80%):** 2002–2020 ($N = 13,397$ observations, 19 unique years, 79.17% of years)
- **Testing Period (20%):** 2021–2025 ($N = 2,993$ observations, 5 unique years, 20.83% of years)
- **Prospective Quarantine:** 2026 unlabelled forecast ($N = 814$ counties)
- **Number of Evaluated Models:** 17 Distinct ML Architectures
- **Target Variable:** `corn_yield_bu_acre`
- **Evaluation Subsets:** Full Test Cohort ($N = 2,993$) and Severe Stress Tail ($<120$ bu/ac, $N = 145$)

## Consolidated Model Performance Leaderboard

| Rank | Model_Name | Category | R2 | RMSE_bu_ac | MAE_bu_ac | MAPE_pct | Mean_Bias_bu_ac | Max_Error_bu_ac | Severe_Tail_RMSE | Severe_Tail_Bias | Train_Time_sec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | ElasticNet | Linear Baseline | 0.761 | 15.37 | 11.78 | 7.35 | -0.88 | 74.33 | 31.34 | 25.36 | 5.9 |
| 2 | Lasso Regression | Linear Baseline | 0.7593 | 15.43 | 11.85 | 7.38 | -0.63 | 74.07 | 31.23 | 25.07 | 4.08 |
| 3 | Support Vector Regression (SVR) | Kernel / Margin | 0.7537 | 15.6 | 12.02 | 7.49 | -0.38 | 72.03 | 31.32 | 25.36 | 0.48 |
| 4 | Ridge Regression | Linear Baseline | 0.7518 | 15.66 | 12.14 | 7.46 | -2.6 | 75.18 | 29.75 | 23.58 | 0.01 |
| 5 | Linear Regression (OLS) | Linear Baseline | 0.7303 | 16.33 | 12.81 | 7.78 | -3.89 | 76.31 | 29.01 | 22.59 | 0.07 |
| 6 | Ridge Stacking Meta-Learner | Stacked Ensemble | 0.7276 | 16.41 | 12.62 | 7.67 | -2.47 | 71.74 | 30.43 | 22.85 | 28.01 |
| 7 | LightGBM Regressor | Gradient Boosting | 0.7264 | 16.45 | 12.62 | 7.63 | -3.16 | 73.21 | 30.28 | 23.06 | 1.9 |
| 8 | Calibrated Quad-Tree Blend | Ensemble Blend | 0.7211 | 16.61 | 12.74 | 7.67 | -3.82 | 75.51 | 30.31 | 23.48 | 12.57 |
| 9 | XGBoost Regressor | Gradient Boosting | 0.7141 | 16.81 | 12.9 | 7.73 | -4.04 | 75.64 | 29.69 | 22.76 | 2.72 |
| 10 | HistGradientBoosting | Gradient Boosting | 0.7133 | 16.83 | 12.89 | 7.8 | -3.39 | 76.33 | 30.92 | 23.68 | 3.27 |
| 11 | CatBoost Regressor | Gradient Boosting | 0.7059 | 17.05 | 13.16 | 7.9 | -4.68 | 76.84 | 31.06 | 24.41 | 4.68 |
| 12 | Gradient Boosting (GBRT) | Boosting Ensemble | 0.6743 | 17.94 | 13.92 | 8.24 | -5.61 | 81.38 | 30.62 | 23.7 | 30.76 |
| 13 | Random Forest Regressor | Tree Ensemble | 0.6626 | 18.26 | 14.49 | 8.61 | -6.77 | 77.49 | 31.37 | 25.37 | 1.42 |
| 14 | Extra Trees Regressor | Tree Ensemble | 0.6249 | 19.26 | 15.71 | 9.19 | -9.19 | 74.99 | 31.22 | 25.9 | 0.52 |
| 15 | Decision Tree Regressor | Tree Baseline | 0.538 | 21.37 | 16.37 | 9.91 | -3.23 | 84.67 | 36.19 | 27.17 | 0.74 |
| 16 | Multilayer Perceptron (MLP) | Neural Network | 0.4815 | 22.64 | 16.74 | 10.19 | 0.35 | 106.65 | 40.41 | 28.02 | 7.74 |
| 17 | K-Nearest Neighbors (KNN) | Instance-Based | 0.3398 | 25.55 | 20.33 | 11.68 | -12.53 | 92.8 | 37.7 | 30.02 | 0.0 |

### Metric Definitions:
- **R²:** Coefficient of Determination ($1 - SS_{res} / SS_{tot}$)
- **RMSE:** Root Mean Squared Error (bushels / acre)
- **MAE:** Mean Absolute Error (bushels / acre)
- **MAPE:** Mean Absolute Percentage Error (%)
- **Mean Bias:** Mean Prediction Error (Predicted - Actual)
- **Max Error:** Maximum absolute residual in test set
- **Severe Tail RMSE / Bias:** Error metrics on extreme drought / stress observations (< 120 bu/ac)
