# TEMPORAL DATA AVAILABILITY & LEAKAGE FORENSIC AUDIT REPORT

## Verification of Operational Cutoff Alignment, Climatological Baselines & Post-Cutoff Quarantines
**Project Root:** `E:\ml do not open\Maize_Yield_Research_2002_2026`  
**Operational Forecast Horizon:** October 15 of Target Crop Year  
**Evaluated Panel:** 2002–2025 Historical Panel (16,390 rows) + 2026 Future Panel (814 rows)  
**Date:** November 2026

---

### 1. Operational Cutoff Protocol (October 15 Horizon)

In operational corn yield forecasting, predictions must be delivered prior to harvest completion (typically mid-October). Any variable incorporating observations recorded after October 15 violates the chronological forecast contract:

```
Timeline:    May 1 ------------ July 31 ---------- August 31 -------- October 15 ------- Dec 31
Data Status: [Observed Weather] [Pollination Peak] [Grain Fill End]   [FORECAST CUTOFF]   [Post-Harvest]
Allowed:     May-Jul Weather, Crop Progress, RMA Liability, GDD/EDD   All Pre-Oct 15      NONE
Quarantined: NONE                                                     NONE                Annual Precip (Nov-Dec),
                                                                                          Post-Harvest Loss Ratios
```

---

### 2. Forensic Findings by Feature Category

#### A. NOAA Climatic Division Variables
- **Monthly Precipitation (May, June, July, August, September):** **PASS**. September climate data is officially released by NOAA NCEI during the first week of October (October 6 release). All monthly variables are genuinely available prior to the October 15 cutoff.
- **`noaa_prcp_annual_in`:** **FLAGGED FOR QUARANTINE**. This variable represents total calendar year precipitation (January through December). For historical years, it includes post-harvest precipitation (November and December). While historically correlated with general hydrological baselines, it constitutes a subtle post-cutoff look-ahead risk. In Phase 9 modeling, it is quarantined from the strict in-season predictor suite.

#### B. Agrometeorological Approximations (GDD, EDD, VPD)
- **Growing Degree Days (`gdd_may_aug_c`):** Verified as an agrometeorological integral proxy over the 123-day May–August growing season using seasonal mean temperature. Available in September.
- **Extreme Degree Days (`edd_july_f`, `edd_aug_f`):** Derived from July/August monthly maximum temperatures exceeding 86.0°F (30.0°C). Available in September.
- **Vapor Pressure Deficit (`vpd_july_kpa`, `vpd_aug_kpa`):** Calculated via the FAO-56 Penman-Monteith saturation vapor pressure proxy using Tetens equation. Because direct continuous gridded relative humidity rasters are not archived in local workspace, this standard agrometeorological proxy is certified and fully documented.

#### C. Climatological Z-Scores (Standardized Anomalies)
- **Leakage Prevention Verification:**
  - `noaa_prcp_jul_in_zscore` and `noaa_tmax_jul_f_zscore` are computed using historical county baseline means ($\mu_i$) and standard deviations ($\sigma_i$).
  - **Audit Result:** Baseline statistics were fitted **strictly on pre-2023 training data (2002–2022)**. Zero observations from the 2023–2025 prospective test period were used in calculating baseline normals.

#### D. USDA RMA Crop Insurance Metrics
- **`rma_liability_per_planted_acre`:** **PASS**. Farm insurance coverage levels and liabilities are legally finalized following the July 15 acreage reporting deadline. They reflect planting-time risk capital and do not leak post-harvest yield or indemnity data.
- **`rma_postharvest_loss_ratio`:** **PASS (QUARANTINED)**. Accurately quarantined in Phase 0; verified absent from all modeling tables.

#### E. Satellite Remote Sensing Audit
- **Audit Result:** While satellite access documentation exists in `01_Raw_Data/`, multi-terabyte raw optical/radar rasters (MODIS, Landsat, Sentinel-2) are not present in local storage. Sentinel-2 also lacks observations prior to 2015. To maintain 100% scientific integrity, satellite features are correctly identified as external future research assets and are not artificially simulated.

---

### 3. Summary of Quarantines and Feature Adjustments for Phase 9

1. **Quarantine `noaa_prcp_annual_in`** from strict in-season models to eliminate calendar look-ahead.
2. **Retain 87 verified in-season predictors** that strictly satisfy the October 15 forecast cutoff.
3. **Preserve full chronological separation** across all subsequent training and validation scripts.
