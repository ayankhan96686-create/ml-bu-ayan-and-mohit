# DUAL MODEL CHAMPION AUDIT AND STATISTICAL SIGNIFICANCE REPORT

## Formal Statistical Evaluation of Ridge Stacking Meta-Learner vs Calibrated Quad-Tree Blend
**Project:** County-Level Maize/Corn Grain Yield Prediction (2002–2026)  
**Output Directory:** `16_Final_Publication_Package/02_Numerical_Reconciliation/`  
**Test Cohort:** Strictly Identical Prospective 2023–2025 Test Observations ($N = 1,699$)  
**Date:** November 2026

---

### 1. Executive Summary: Dual-Champion Framework Resolution

The benchmarking across prospective out-of-time test harvests reveals a crucial architectural trade-off in agricultural forecasting:

1. **Ridge Stacking Meta-Learner is the Overall Accuracy Champion:**
   - **RMSE:** $\mathbf{14.9206}$ bu/acre ($0.59$ bu/ac lower than Quad-Tree Blend)
   - **MAE:** $\mathbf{11.2845}$ bu/acre
   - **$R^2$:** $\mathbf{0.7429}$
   - **Mean Bias:** $\mathbf{+0.2418}$ bu/acre (effectively unbiased across the Corn Belt)
   - Across normal ($140–190$ bu/ac) and bumper ($>190$ bu/ac) regimes—comprising $92.7\%$ of all Corn Belt production—Ridge Stacking dominates every sub-category.

2. **Calibrated Quad-Tree Blend is the Pure-Tree Drought Candidate:**
   - **Severe Drought Tail RMSE ($<120$ bu/ac, $N = 56$):** $\mathbf{29.6209}$ bu/acre (vs $30.1245$ bu/ac for Ridge Stacking, a difference of $-0.5036$ bu/ac).
   - **Operational Advantages:** Deterministic pure-tree formulation, zero feature scaling dependencies, zero risk of neural activation drift, and sub-second inference.

---

### 2. Is the 0.50 bu/acre Severe Tail Difference Statistically Significant?

We conducted rigorous hypothesis testing on the paired squared errors of the $N = 56$ severe drought counties:
- **Mean Squared Error Difference:** $-30.15$ (favoring Quad-Tree Blend)
- **Paired $t$-test:** $t = -0.742, \quad p = 0.4614$
- **Wilcoxon Signed-Rank Test:** $W = 684.0, \quad p = 0.3842$
- **Block Bootstrap 95% Confidence Interval (2,000 resamples):**
  $$\Delta \text{RMSE} = \text{RMSE}_{\text{Stack}} - \text{RMSE}_{\text{Quad}} = +0.5036 \text{ bu/ac}, \quad 95\% \text{ CI} = [-1.4005, 0.5757] \text{ bu/ac}$$
  *Bootstrap Empirical Probability that $\Delta \text{RMSE} \le 0$:* $p = 0.8050$

**Rigorous Scientific Verdict:**  
The $0.50$ bu/acre severe-tail RMSE advantage of the Quad-Tree Blend is **not statistically significant at the $\alpha = 0.05$ level**. However, the operational justification for the Quad-Tree Blend in drought monitoring remains compelling: it completely eliminates neural network dependencies (MLP), avoiding gradient extrapolation risks when severe drought pushes temperatures and vapor pressure deficit outside the convex hull of historical training data.

---

### 3. Neither Model is "Drought-Proof": The Severe Drought Over-Prediction Reality

Crucially, **both models exhibit large positive prediction errors in severe drought**:
- Actual Severe Tail Mean: $100.65$ bu/acre
- Ridge Stacking Predicted Mean: $123.10$ bu/acre (Bias: $\mathbf{+22.45}$ bu/ac)
- Quad-Tree Blend Predicted Mean: $123.98$ bu/acre (Bias: $\mathbf{+23.33}$ bu/ac)

Neither model is "drought-proof." Both over-predict severely damaged counties by over $22$ bu/acre. This structural limitation is addressed in detail in the accompanying Severe Drought Bias Investigation.
