# COMPREHENSIVE 2012 DATA PRESERVATION AND SENSITIVITY AUDIT

## Rigorous Evaluation of Scenario 1 (Model Training Exclusion) vs Scenario 2 (Spine Deletion)
**Project:** County-Level Maize/Corn Grain Yield Prediction (2002–2026)  
**Output Directory:** `15_Final_Confirmatory_Validation/03_Drought_Robustness/`  
**Panel Coverage:** 2002–2025 ($N = 16,390$ records), 788 Eligible Counties, 9 US Corn Belt States  
**Date:** November 2026

---

### 1. Executive Summary: The Definitive Scientific Preservation Verdict

The 2012 US Corn Belt drought was the most severe climatological and agricultural event in North America since 1988, reducing average Corn Belt yields to 111.75 bu/acre (and rainfed core states like Missouri to 68.23 bu/acre).

A proposal was raised in early audit phases questioning whether removing 2012 could reduce model RMSE. This audit provides conclusive statistical and methodology proofs establishing why **the year 2012 must be permanently preserved in the authoritative historical dataset**:

1. **2012 Observations are Genuine Physical Signals, Not Outliers:**  
   Every one of the 711 county records in 2012 reflects verified USDA NASS harvest measurements under extreme thermal and hydrological stress. Excluding extreme observations simply to inflate empirical test $R^2$ constitutes severe scientific bias and destroys model generalizability to future climate shocks.

2. **Scenario 1 vs Scenario 2 Dissection:**
   - **Scenario 1 (Training Row Exclusion Only):** If 2012 is excluded *only* from the model fitting phase after constructing all features on the complete calendar spine, historical yield lags (`yield_lag_1yr`, `yield_lag_2yr`, `yield_roll_mean_5yr`) for 2013–2017 remain valid. However, the model loses its only severe training drought anchor, causing a **$+1.98$ bu/acre penalty in the severe drought tail** ($< 120$ bu/acre) during the 2023–2025 prospective test period.
   - **Scenario 2 (Spine Deletion / Dataset Purge):** If 2012 is deleted prior to feature engineering, the calendar spine is broken. The year 2013 erroneously treats 2011 as its 1-year lag; 5-year rolling averages across 2013–2017 become completely distorted; and spatial panel autoregressive estimators suffer severe coefficient attenuation. This scenario represents fatal data pipeline corruption.

3. **Empirical Evidence of Drought Under-Prediction Without 2012:**  
   Models trained without 2012 fail to recognize the non-linear inflection point of crop collapse. When tested on the 2023 drought tail ($N=56$, mean yield $100.65$ bu/ac):
   - With 2012: RMSE is $29.62$ bu/ac, capturing an average yield loss of $56.02$ bu/ac from normal.
   - Without 2012: RMSE inflates to $31.13$ bu/ac (Delta $+1.51$ bu/ac), and captured loss drops to $54.47$ bu/ac.
   - For LightGBM alone, severe tail RMSE increases from $29.38$ to $31.36$ bu/ac (a tail penalty of $+1.9821$ bu/ac).

---

### 2. Scenario 1 vs Scenario 2 Formal Comparison

| Evaluation Dimension | Scenario 1: Training Row Exclusion Only | Scenario 2: Calendar Spine Purge |
| :--- | :--- | :--- |
| **Data Integrity** | Complete calendar spine preserved (2002–2025) | Fractured calendar spine (gap at 2012) |
| **Lag 1-Year (2013)** | Accurately references 2012 yield shock | Erroneously references 2011 yield (2-year lag disguised as 1-year) |
| **5-Year Rolling Window** | Accurately includes 2012 in 2013–2017 baselines | Distorted calculation across 2013–2017 |
| **Severe Tail RMSE (<120)** | Degrades by $+1.51$ to $+1.98$ bu/ac | Catastrophic structural bias in lags and weather |
| **Scientific Viability** | Acceptable as isolated sensitivity experiment | Scientifically invalid and prohibited |

---

### 3. Conclusion & Authoritative Standard

The authoritative dataset `Historical_Features_Corrected_v2.csv` retains all 711 records of 2012. Models trained for production deployment and manuscript presentation must include 2012 to ensure resilience against future climate extremes.
