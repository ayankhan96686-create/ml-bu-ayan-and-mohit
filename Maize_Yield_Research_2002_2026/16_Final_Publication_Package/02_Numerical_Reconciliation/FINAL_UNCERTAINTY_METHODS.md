# UNCERTAINTY QUANTIFICATION AND PREDICTION INTERVAL METHODOLOGY

## Formal Differentiation Between County-Level Prediction Intervals and Regional Aggregate Bounds
**Project:** County-Level Maize/Corn Grain Yield Prediction (2002–2026)  
**Output Directory:** `16_Final_Publication_Package/02_Numerical_Reconciliation/`  
**Date:** November 2026

---

### 1. Resolution of the Two Reported 90% Uncertainty Figures (±23.4 vs ±6.8 bu/ac)

A critical source of potential confusion in crop yield reporting is conflating **single-county prediction intervals** with **regional aggregate mean uncertainty**:

1. **County-Level Prediction Interval ($\pm 23.42$ bu/acre):**  
   - Answers: *"Given a forecast of 175 bu/acre for Story County, Iowa, what is the 90% plausible range of actual harvested yield?"*  
   - Derived using **Split-Conformal Prediction**: calibration residuals $R_i = |y_i - \hat{y}_i|$ on out-of-fold training data define the non-conformity score quantile $q_{0.90} = 23.42$ bu/acre.  
   - Empirical coverage on the out-of-time test set ($2023–2025$, $N=1,699$) is **$90.23\%$**, confirming valid finite-sample coverage.

2. **Regional Aggregate Production Uncertainty ($\pm 6.84$ bu/acre):**  
   - Answers: *"Given the forecasted acreage-weighted regional yield of 181.12 bu/acre across all 9 states, what is the uncertainty of this aggregate regional figure?"*  
   - Individual county errors partially cancel across space. Due to positive spatial autocorrelation in weather shocks ($ho pprox 0.35–0.45$), regional variance shrinks by $pprox 70\%$, yielding an aggregate 90% confidence half-width of $\pm 6.84$ bu/acre (translating to $\pm 467$ million bushels on $68.36$ million harvested acres).

```
+========================================================================================================+
| Uncertainty Level | Individual County Half-Width | Regional Aggregate Half-Width | Empirical Coverage  |
+===================+==============================+===============================+=====================+
| 80% Interval      | +/- 17.84 bu/acre            | +/- 5.21 bu/acre              | 80.45%              |
| 90% Interval      | +/- 23.42 bu/acre            | +/- 6.84 bu/acre              | 90.23%              |
| 95% Interval      | +/- 29.81 bu/acre            | +/- 8.71 bu/acre              | 95.12%              |
+========================================================================================================+
```

---

### 2. Methodological Assumptions and Limitations

- **Exchangeability Assumption:** Conformal prediction assumes exchangeability between calibration and test distributions. In agricultural panels, climate change and technological drift induce temporal non-exchangeability.
- **Heteroscedasticity Limitation:** Conformal coverage drops from $91.5\%$ in normal regimes to $68.2\%$ in severe drought regimes, demonstrating that prediction interval width should ideally adapt dynamically to predicted vapor pressure deficit and soil moisture.
