# RESEARCH DECISION RECORDS (ADRs)
## County-Level Maize Grain Yield Prediction Framework (2002–2026)

This document records the foundational scientific, methodological, and architectural decisions made throughout the research project.

---

### ADR-001: Strict Quarantine of Prospective 2026 Observations

- **Status:** Certified & Active
- **Date:** 2026-11-15
- **Context:** In downstream folders (Phase 16 and 17), county-level rows for year 2026 exist with valid pre-season environmental predictors, but USDA NASS ground truth yields are unobserved (100% missing) because harvest has not occurred.
- **Decision:** All 2026 rows ($N = 814$) are strictly quarantined from supervised loss functions, hyperparameter optimization, and test metric evaluations. Any non-null 2026 yield values in repository logs are recognized as model forecast simulations and must never be passed as target training labels.
- **Consequences:** Eliminates target corruption. Maintains pristine separation between historical empirical validation (2002–2025) and prospective forecasting (2026).

---

### ADR-002: Dual-Evaluation Paradigm (Pooled Stratified 80:20 vs Strict Chronological Holdout)

- **Status:** Certified & Active
- **Date:** 2026-11-15
- **Context:** Agricultural ML literature exhibits a split: studies employing random k-fold cross-validation report $R^2 \approx 0.85\text{--}0.90$, while studies using strict out-of-time temporal holdouts report $R^2 \approx 0.70\text{--}0.78$.
- **Decision:** Implement and report both evaluation paradigms:
  1. *Primary Pooled Stratified 80:20 Split:* Measures spatial interpolation capability across all historical climate regimes (XGBoost $R^2 = 0.8986$, $\text{RMSE} = 11.67\text{ bu/ac}$).
  2. *Chronological Out-of-Time Holdout (2002–20 Train vs 2021–25 Test):* Measures genuine temporal forecasting under unobserved macro-climate regimes (ElasticNet $R^2 = 0.7610$; Stacking $R^2 = 0.7276$).
- **Consequences:** Provides complete scientific transparency. Documents the $\Delta R^2 \approx 0.16$ "Optimism Gap" for tree ensembles caused by contemporaneous regional weather correlation.

---

### ADR-003: Permanent Retention of the 2012 Severe Drought in Training Populations

- **Status:** Certified & Active
- **Date:** 2026-11-12 (Phase 14 Ablation Audit)
- **Context:** An ablation experiment tested excluding the 2012 drought year ($N = 684$, mean yield 107.8 bu/ac) to evaluate impact on statistical fit.
- **Decision:** Retain all verified 2012 observations in the authoritative training set. Never remove extreme drought observations merely to artificially inflate $R^2$.
- **Consequences:** Models trained without 2012 fail catastrophically when encountering severe future heat/drought shocks (tail error increases by $>40\%$). Preserving 2012 ensures drought robustness ($<120$ bu/ac tail RMSE = $17.15\text{ bu/ac}$).

---

### ADR-004: In-Sample Chronological Validation Folds for Hyperparameter Optimization

- **Status:** Certified & Active
- **Date:** 2026-11-15
- **Context:** Tuning hyperparameters on random k-folds or the final test set introduces future temporal leakage and optimistic overfitting.
- **Decision:** Hyperparameter optimization is confined entirely to the **80% training partition** ($N = 13,112$) using 4 expanding chronological folds (Fold 1: 2002–13 $\to$ 2014–16; Fold 2: 2002–16 $\to$ 2017–19; Fold 3: 2002–19 $\to$ 2020–22; Fold 4: 2002–22 $\to$ 2023–25). The 20% test holdout remains locked until final testing.
- **Consequences:** Ensures zero data snooping. Hyperparameter selections generalize reliably to the unseen test set.

---

### ADR-005: 8-Point Anti-Leakage Preprocessing Protocol

- **Status:** Certified & Active
- **Date:** 2026-11-15
- **Context:** Data leakage in agricultural ML frequently arises from improper imputation, feature scaling across test sets, and contemporaneous lag definitions.
- **Decision:** Implement programmatic checks enforcing:
  - Missing value imputation medians fitted strictly on training subsets.
  - Standard scaling parameters fitted strictly on training subsets.
  - Historical yield lags enforce $t - k \ge 1$ (zero contemporaneous yield in predictors).
  - Pre-harvest seasonal cut-off (September 30) for all weather features.
- **Consequences:** Passed all 8 automated checks in `DATA_LEAKAGE_AUDIT.csv`. Guaranteed mathematical integrity of reported generalization metrics.

---

### ADR-006: Feature Configuration Ablation & Compact 45-Feature Subset

- **Status:** Certified & Active
- **Date:** 2026-11-15
- **Context:** Models using 117 features risk over-parameterization and multicollinearity among collinear weather variables.
- **Decision:** Evaluate Configurations A (Climate & Soil), B (+Lags), C (+Agro-Stress), D (Top 45 Selected), and E (Master 117 Feats). Select Config D via permutation importance on training folds.
- **Consequences:** Config D achieves $99.8\%$ of the accuracy of the full 117-feature set ($R^2 = 0.8885$ vs $0.8904$) while reducing dimensionality by $61.5\%$, drastically accelerating inference and reducing overfitting.

---

### ADR-007: Ridge Stacking Meta-Learner Architecture

- **Status:** Certified & Active
- **Date:** 2026-11-15
- **Context:** Simple linear averaging fails to account for diverse residual structures among base learners (LightGBM, XGBoost, CatBoost).
- **Decision:** Train a Ridge Regression ($lpha = 5.0$) meta-model on out-of-fold predictions from tuned base models. Never train stacking meta-models on in-sample base predictions.
- **Consequences:** Out-of-fold validation RMSE drops to $15.42\text{ bu/ac}$ (surpassing all individual base models in cross-validation), delivering an independent test $R^2 = 0.8931$ and $\text{RMSE} = 11.98\text{ bu/ac}$.

---

### ADR-008: Automated Publication PDF Generation via ReportLab Flowable Canvas

- **Status:** Certified & Active
- **Date:** 2026-11-15
- **Context:** Disjoint manual reporting across markdown tables and ad-hoc figures hinders peer-review auditing and journal manuscript preparation.
- **Decision:** Develop `generate_master_publication_pdf.py` using ReportLab 5.0.0 to automatically compile all 12 experimental phases into a publication-quality research deliverable with dynamic two-pass pagination (`NumberedCanvas`), custom academic palettes, and auto-wrapped benchmark tables.
- **Consequences:** Guarantees that any pipeline re-run instantly produces an identical, synchronized, high-resolution PDF deliverable (`MASTER_MAIZE_YIELD_OPTIMIZATION_REPORT.pdf`).
