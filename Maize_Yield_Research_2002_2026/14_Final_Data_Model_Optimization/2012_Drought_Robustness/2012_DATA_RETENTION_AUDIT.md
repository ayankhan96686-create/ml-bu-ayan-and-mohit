# 2012 HISTORICAL DATA RETENTION AUDIT

## Scientific Justification, Biophysical Integrity, and Non-Deletion Certification
**Project Root:** `E:\ml do not open\Maize_Yield_Research_2002_2026`  
**Study Scope:** 9 US Corn Belt States | 788 Eligible Agricultural Counties  
**Target Subject:** 2012 US Corn Belt Extreme Drought Year ($N = 711$ Historical Records)  
**Date:** November 2026

---

### 1. The Biophysical Reality of the 2012 Crop Year

The 2012 crop season across the central United States represents the most severe agricultural drought and high-temperature stress event of the satellite and modern statistical era (2002–2025).

```
+------------------------------------+------------------+------------------+-----------------------+
| Biophysical Metric                 | Year 2012 Mean   | Decadal Normal   | Anomaly / Difference  |
+====================================+==================+==================+=======================+
| USDA County Corn Yield             | 111.75 bu/acre   | 157.45 bu/acre   | -45.70 bu/acre (-29%) |
| Minimum County Corn Yield          | 19.00 bu/acre    | 85.00 bu/acre    | Extreme failure tail  |
| July Extreme Degree Days (EDD >86F)| 198.04 EDD       | 35.82 EDD        | +553% heat stress     |
| July Vapor Pressure Deficit (VPD)  | 3.48 kPa         | 2.49 kPa         | +40% evaporative pull |
| July NOAA Precipitation            | 1.97 inches      | 3.94 inches      | -50% rainfall deficit |
+------------------------------------+------------------+------------------+-----------------------+
```

1. **Not a Measurement Error:**  
   The low yields recorded in 2012 are verified official USDA NASS statistical survey and administrative RMA crop insurance figures. Every county in Illinois, Indiana, Iowa, and Missouri experienced genuine biophysical crop impairment caused by pollination failure, kernel abortion, and rapid soil moisture exhaustion.
2. **The Only Multi-State Catastrophic Negative Anchor:**  
   In the pre-2023 training window (2002–2022), 2012 is the **sole continental-scale negative anchor**. Without 2012, machine learning models trained on 2002–2022 observe almost exclusively normal or bumper harvest years (2004, 2009, 2014, 2016, 2018).

---

### 2. Training-Row Exclusion vs Calendar Spine Corruption

A critical distinction must be drawn between two completely different ways of "removing 2012":

```
                                  [COMPLETE HISTORICAL PANEL 2002-2025]
                                                    |
                   +--------------------------------+--------------------------------+
                   |                                                                 |
    [SCENARIO A: Training-Row Exclusion]                            [SCENARIO B: Calendar Spine Deletion]
    - Engineer all lags on complete 2002-2025 spine.                 - Delete 2012 before lag engineering.
    - Exclude year==2012 only during model.fit().                   - 2013 lag_1yr becomes missing / broken.
    - 2013-2017 lags remain 100% mathematically valid.              - 2014 lag_2yr becomes missing / broken.
    - Impact: Model loses drought training examples.                 - 2013-2017 rolling means artificially jump +9.5 bu/ac.
    - Result: CatBoost RMSE +0.23 bu/ac; Tail RMSE +1.51 bu/ac.     - Result: 3,381 county-year records corrupted!
```

- **Scenario A (Controlled Sensitivity Test):** Evaluated in `2012_WITH_WITHOUT_TRAINING_COMPARISON.csv`. By keeping the calendar spine intact, we isolate the pure algorithmic cost of missing drought training targets.
- **Scenario B (Unscientific Data Deletion):** Deleting 2012 from raw tables destroys the integrity of historical lags across 5 subsequent years ($3,381$ records), inducing severe feature leakage and artificial yield drift.

---

### 3. Non-Deletion Invariant & Scientific Policy

1. **2012 Must Remain in the Authoritative Dataset:**  
   The authoritative historical feature table (`Historical_Features_Corrected_v2.csv`) retains all 711 county records for 2012.
2. **Never Delete Data to Artificially Inflate Metric Scores:**  
   Deleting difficult, low-yielding, or anomalous agricultural years to artificially elevate reported $R^2$ represents academic malpractice. A genuine machine learning model must accurately capture both record-breaking harvests and catastrophic droughts.
3. **The Solution is Biophysical Covariates, Not Sample Censorship:**  
   The path to higher accuracy is explicit modeling of Extreme Degree Days (`edd_july_f`), Vapor Pressure Deficit (`vpd_july_kpa`), Compound Heat-Drought (`compound_heat_drought_july`), and Irrigation Buffers (`irrigation_drought_buffer`), which empower the algorithm to predict drought collapse accurately.
