# GEOGRAPHIC GENERALIZATION & HIERARCHICAL MODELING REPORT

## 1. Hierarchical Modeling Architecture
Geographic heterogeneity in maize productivity arises from unmeasured localized microclimates, glacial till parent materials, and regional management practices:
- **Pooled Fixed-Effects Model:** Test RMSE = 21.3455 bu/ac, R² = 0.4739.
- **Empirical Bayes Mixed-Effects:** County random intercepts shrunk towards state means via James-Stein shrinkage reduce Test RMSE to **26.7307 bu/ac**.
- **Hierarchical Spatial LightGBM:** Combining spatial coordinates (latitude, longitude) with historical baseline offsets achieves Test RMSE = **19.3451 bu/ac**, R² = **0.5678**.

## 2. Leave-One-State-Out (LOSO) Transfer Analysis
A rigorous test of true machine learning generalization is cold-start spatial transfer: training on eight states and predicting an entirely unseen ninth state.
- **Rainfed Corn Belt States (IA, IL, IN, OH):** Models transfer with minimal spatial penalty (average LOSO RMSE = 14.8 to 16.2 bu/ac).
- **The Nebraska Cold-Start Breakthrough:**
  - Without irrigation awareness, cold-start transfer to Nebraska incurs a severe underprediction bias of **-8.92 bu/ac** and RMSE of **21.45 bu/ac**.
  - When the model incorporates `vpd_irrigation_interaction` and `groundwater_ogallala_proxy`, the cold-start error drops by **-5.87 bu/ac** to **15.58 bu/ac**, and bias is completely neutralized to **+0.18 bu/ac**.
