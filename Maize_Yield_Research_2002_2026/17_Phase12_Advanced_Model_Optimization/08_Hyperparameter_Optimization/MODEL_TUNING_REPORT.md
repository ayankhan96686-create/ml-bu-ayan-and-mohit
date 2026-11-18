# ADVANCED HYPERPARAMETER OPTIMIZATION REPORT

## Methodology & Validation Guardrails
To prevent data leakage into the prospective evaluation benchmarks:
1. All hyperparameter tuning was conducted strictly within the **2002–2022 historical training period**.
2. An **expanding-window temporal cross-validation protocol** (3 multi-year folds: 2017–2018, 2019–2020, 2021–2022) was utilized.
3. The 2023–2025 final benchmark was strictly withheld during hyperparameter selection.

## Best Hyperparameter Configurations & Validation Scores

| Algorithm | Best Configuration | Expanding-Window CV RMSE (bu/ac) |
| :--- | :--- | :---: |
| **LightGBM** | `learning_rate`: 0.03, `num_leaves`: 63, `n_estimators`: 400, `reg_alpha`: 1.0 | **16.5797** |
| **XGBoost** | `learning_rate`: 0.03, `max_depth`: 5, `n_estimators`: 300, `reg_lambda`: 3.0 | **16.8657** |
| **CatBoost** | `learning_rate`: 0.03, `depth`: 7, `iterations`: 400, `l2_leaf_reg`: 5.0 | **16.8557** |
| **HistGradientBoosting** | `learning_rate`: 0.02, `max_leaf_nodes`: 45, `max_iter`: 350 | **16.9084** |
| **MLP** | `hidden_layer_sizes`: (64, 32), `alpha`: 0.1, `learning_rate_init`: 0.002 | **19.3158** |
