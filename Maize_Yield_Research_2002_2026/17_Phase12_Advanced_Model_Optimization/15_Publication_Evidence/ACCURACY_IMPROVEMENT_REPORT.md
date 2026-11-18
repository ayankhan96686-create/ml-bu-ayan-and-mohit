# ACCURACY IMPROVEMENT & BENCHMARK PROGRESSION REPORT

| Research Phase | Model Architecture | Test RMSE (bu/ac) | Test R² | Severe Tail RMSE (<120 bu/ac) | Severe Tail Bias |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Phase 0 Baseline** | 5-Year Historical Mean Baseline | 22.45 | 0.428 | 48.50 bu/ac | +42.10 bu/ac |
| **Phase 5 Benchmark** | In-Season LightGBM Regressor | 15.63 | 0.718 | 29.38 bu/ac | +22.85 bu/ac |
| **Phase 10/11 Benchmark**| Ridge Stacking Meta-Learner | 14.92 | 0.743 | 30.12 bu/ac | +22.45 bu/ac |
| **Phase 12 Optimized** | **Mixture of Experts (MoE Gated)**| **14.58** | **0.755** | **27.85 bu/ac** | **+17.92 bu/ac** |
