# SEVERE DROUGHT PREDICTION BIAS INVESTIGATION

## Root Causes of Systematic Forecast Over-Prediction During Extreme Yield Collapses
**Project:** County-Level Maize/Corn Grain Yield Prediction (2002–2026)  
**Output Directory:** `16_Final_Publication_Package/02_Numerical_Reconciliation/`  
**Identified Phenomenon:** Both champion models over-predict severe drought yields (<120 bu/acre) by approximately +22 to +23 bu/acre.  
**Date:** November 2026

---

### 1. The Phenomenon

During the 2023–2025 prospective test period, 56 county-year harvests experienced severe yield collapse (<120 bu/acre), with an average harvested yield of $100.65$ bu/acre (down from the regional normal of ~178 bu/ac).
- The Ridge Stacking ensemble predicted an average yield of $123.10$ bu/acre (Mean Bias $= +22.45$ bu/ac).
- The Calibrated Quad-Tree Blend predicted an average yield of $123.98$ bu/acre (Mean Bias $= +23.33$ bu/ac).

This report investigates the mechanistic and statistical causes of this persistent positive bias.

---

### 2. Mechanistic Root Causes

1. **Historical Yield Autoregressive Memory Pull:**  
   The engineered historical features (`yield_roll_mean_5yr` and `yield_lag_1yr`) are the single most powerful predictors in normal and bumper years, anchoring county-level predictions to local technological and soil baselines (~160–190 bu/ac). During an acute drought shock, these features act as strong Bayesian priors pulling predictions upward, counteracting the downward environmental signal.

2. **Rarity of Severe Regional Drought Training Examples:**  
   In the 21-year historical training panel (2002–2022), only a single widespread catastrophic drought year occurred across the entire Corn Belt: **2012**. Tree-based partition algorithms (GBDTs) have relatively few leaf nodes dedicated to the joint extreme parameter space where July EDD $> 60$, July VPD $> 3.2$ kPa, and July precipitation is below the 5th percentile.

3. **Non-Linear Biological Thresholds vs Smooth Tree Averaging:**  
   Maize reproductive physiology exhibits catastrophic non-linear thresholds:
   - When daily maximum temperatures exceed 35°C (95°F) during the 10-day silking window, pollen viability collapses within hours, causing fertilization failure and incomplete kernel set (nubbins or barren stalks).
   - In GBDT models, leaf predictions represent the empirical mean of training observations falling in that leaf. Because training leaves in extreme regions still contain some irrigated or soil-buffered counties with moderate yields (e.g. 110–130 bu/ac), the leaf average cannot reach extreme lows (e.g. 40–80 bu/ac) observed in rainfed sandy counties.

4. **Soil Water Capacity (AWC) and Subsoil Interaction:**  
   In counties with low soil water holding capacity (<140 mm in top 100 cm), transpirational buffering is exhausted within 7–10 days of zero rainfall, precipitating rapid leaf rolling and canopy senescence that satellite vegetation indices do not detect until after irreversible yield loss has occurred.

---

### 3. Why Post-Hoc "Drought Penalties" Were Rejected

We tested training secondary residual models and heuristic penalty functions. In all cases:
- Heuristic penalties calibrated on 2012 over-penalized normal summer heat spells in 2024 and 2025, converting a slight drought improvement into catastrophic over-prediction of losses across the remaining 1,643 test counties.
- Scientific integrity dictates that machine learning models must not be manipulated with ad-hoc artificial penalties. The +22 bu/acre drought bias is reported transparently as a fundamental boundary condition of regional empirical models.
