# COMPREHENSIVE SCIENTIFIC EVIDENCE AUDIT REPORT

## Forensic Verification of Data Lineage, Metrics, Model Architectures, and Research Claims (Phases 0–10)
**Project Title:** County-Level Maize (*Zea mays L.*) Grain Yield Prediction Using Machine Learning (2002–2026)  
**Study Scope:** 9 US Corn Belt States, 814 Total Counties (788 Eligible Agricultural Counties)  
**Output Directory:** `16_Final_Publication_Package/01_Evidence_Audit/`  
**Date:** November 2026

---

### 1. Executive Audit Summary

As part of Phase 11 journal publication preparation, an independent forensic audit was conducted across all experimental artifacts, scripts, and reports generated from Phase 0 through Phase 10.

**Core Findings:**
1. **The 13 vs 12 Architecture Count Discrepancy is Fully Reconciled:**  
   The project evaluated **13 unique model architectures**:
   - 12 continuous yield regressors (Ridge Stacking, Quad-Tree Blend, Voting Ensemble, LightGBM, XGBoost, CatBoost, HistGB, RF, ExtraTrees, MLP, Ridge, LASSO).
   - 1 binary ecological niche viability classifier (Random Forest classifier separating optimal $>160$ bu/ac from stressed $<120$ bu/ac regimes, ROC-AUC = 0.892).
   Tables presenting continuous yield regression display the 12 regression models; when including the niche classifier, the total architecture count is exactly 13.
2. **Historical Data Integrity:**  
   The authoritative historical feature table (`Historical_Features_Corrected_v2.csv`) contains exactly 16,390 verified county-year observations across 788 counties over 24 calendar years (2002–2025). All 711 county records from the 2012 severe drought are preserved.
3. **Reconciliation of Reporting Typographies:**  
   The 108.9 bu/acre 2012 figure was traced to an early annotation string error in `perform_eda.py` Line 124. The authoritative unweighted mean is 111.75 bu/acre, and the acreage-weighted mean is 116.71 bu/acre.
4. **Resolution of Unsupported Claims:**  
   All claims describing models as "drought-proof" or asserting uniform 90% conformal coverage under extreme drought are formally retracted and classified in the `UNSUPPORTED_CLAIMS_REGISTER.csv`.

---

### 2. Evidence Audit Ledger Summary Table

```
Claim_ID                           Topic      Audit_Classification
 CLM-001           Historical Panel Size                  VERIFIED
 CLM-002       Total Geographic Coverage                  VERIFIED
 CLM-003   2012 Severe Drought Retention                  VERIFIED
 CLM-004     2012 Mean Yield Discrepancy       REQUIRES_CORRECTION
 CLM-005          Overall Model Champion                  VERIFIED
 CLM-006     Drought Robustness Champion VERIFIED_WITH_LIMITATIONS
 CLM-007 Prediction Interval Calibration VERIFIED_WITH_LIMITATIONS
 CLM-008       Prospective 2026 Forecast VERIFIED_WITH_LIMITATIONS
 CLM-009          Loss Function Collapse       REQUIRES_CORRECTION
 CLM-010    Residual Correction Boosting               UNSUPPORTED
```

---

### 3. Model Architecture Count Breakdown

```
                                  Category_Level  Exact_Count                                                                                                                                                                                                                                                                                                                    Specific_Entities                                                                               Scientific_Definition
              Unique Machine Learning Algorithms            8                                                                                                                                                                                               LightGBM, XGBoost, CatBoost, HistGradientBoosting, Random Forest, Extra Trees, Multi-Layer Perceptron (MLP), Ridge/LASSO Linear Models                              Distinct statistical learning formulations or optimization algorithms.
                      Unique Model Architectures           13 1. Ridge Stacking Meta-Learner, 2. Calibrated Quad-Tree Blend, 3. Tree-Based Voting Metamodel, 4. LightGBM Regressor, 5. XGBoost Regressor, 6. CatBoost Regressor, 7. HistGB Regressor, 8. Random Forest, 9. Extra Trees, 10. MLP Neural Net, 11. Ridge Regressor, 12. LASSO Regressor, 13. Ecological Niche RF Viability Classifier        Unique end-to-end predictive topologies including multi-model meta-learners and classifiers.
                  Unique Ensemble Configurations            3                                                                                                                                                                1. Ridge Stacking (alpha=150.0 on 5-fold OOF), 2. Calibrated Quad-Tree Blend (0.35/0.25/0.20/0.20 weights), 3. Tree-Based Voting Metamodel (0.30/0.25/0.15/0.15/0.15)                                              Architectures combining multiple base learner outputs.
Unique Hyperparameter Configurations Benchmarked           24                                                                                                                                                       Multiple loss-function variations (L2, Huber alpha=0.7/0.8/0.9, Fair c=15/20/25, L1, Quantile 0.5, Pseudo-Huber), tree depths, and regularization strengths across Phases 9-10  Distinct hyperparameter parameterizations evaluated in controlled ablation and sensitivity suites.
      Total Benchmark Validation Experiments Run           42                                                                                                                                                                                     13 architecture benchmarks + 12 loss-function validations + 7 rolling-origin temporal folds + 9 spatial LOSO folds + 1 prospective 2026 forecast Total discrete training/testing execution cycles executed across historical and prospective panels.
```
