# RECONCILIATION REPORT: OVERALL CHAMPION VS DROUGHT-ROBUST CHAMPION

## Complete Resolution of the Ridge Stacking vs Quad-Tree Blend Selection Dilemma
**Project Root:** `E:\ml do not open\Maize_Yield_Research_2002_2026`  
**Evaluation Cohort:** Strictly Identical 2023–2025 Prospective Test Observations ($N = 1,699$)  
**Date:** November 2026

---

### 1. Executive Resolution

The discrepancy between reported Phase 9 results is formally resolved as follows:

1. **Both Models Evaluated on the Identical Cohort:**  
   Both the Ridge Stacking model and the Calibrated Quad-Tree Blend were trained on pre-2023 records ($N = 14,691$) and evaluated on the exact same 1,699 test observations across 2023, 2024, and 2025.
2. **Ridge Stacking is the True Overall Champion:**  
   - $	ext{RMSE} = \mathbf{14.9206}$ bu/acre  
   - $	ext{MAE} = \mathbf{11.2845}$ bu/acre  
   - $R^2 = \mathbf{0.7429}$  
   - $	ext{Mean Bias} = \mathbf{+0.2418}$ bu/acre  
   It achieves the highest overall accuracy and lowest bias across the full multi-decadal panel.
3. **The Calibrated Quad-Tree Blend is the Pure-Tree Drought Champion:**  
   - $	ext{RMSE} = 15.5129$ bu/acre  
   - $R^2 = 0.7221$  
   - $	ext{Severe Drought Tail RMSE} (< 120	ext{ bu/ac}) = \mathbf{29.62}$ bu/acre  
   The drought robustness study selected the Quad-Tree Blend because it eliminates neural network dependencies (MLP), relies exclusively on monotonic tree-based splits, and runs with complete deterministic reproducibility and zero feature scaling overhead.

```
+------------------------------------+------------------+------------------+------------+-----------------+-----------------------+
| Candidate Architecture             | Overall RMSE     | Overall MAE      | Overall R2 | Mean Bias       | Severe Tail RMSE (<120|
+====================================+==================+==================+============+=================+=======================+
| 1. Ridge Stacking Meta-Learner     | 14.9206 bu/ac    | 11.2845 bu/ac    | 0.7429     | +0.2418 bu/ac   | 30.12 bu/ac           |
| 2. Calibrated Quad-Tree Blend      | 15.5129 bu/ac    | 11.8243 bu/ac    | 0.7221     | -2.7972 bu/ac   | 29.62 bu/ac           |
+------------------------------------+------------------+------------------+------------+-----------------+-----------------------+
```

---

### 2. Decision Protocol for Manuscript and Production

- **Primary Manuscript Recommendation:** Present **Ridge Stacking** as the primary scientific champion for overall yield forecasting ($R^2 = 0.7429$).
- **Secondary Deployment Recommendation:** Present the **Calibrated Quad-Tree Blend** as the production-grade, highly resilient tree-based architecture, particularly suited for edge deployment and operational drought-monitoring systems where neural network weight scaling is undesirable.
