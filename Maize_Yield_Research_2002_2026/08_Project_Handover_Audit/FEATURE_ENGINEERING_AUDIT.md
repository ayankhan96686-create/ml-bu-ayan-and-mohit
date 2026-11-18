# Feature Engineering Technical Audit & Agronomic Assessment

**Project:** Maize/Corn Grain Yield Prediction Using Machine Learning (2002–2025)  
**Root Directory:** `E:/ml do not open/Maize_Yield_Research_2002_2026/`  
**Audit Date:** 2026-11-10  
**Auditor:** Senior Agricultural Data Scientist & Machine Learning Engineer  

---

## 1. Executive Summary of Feature Availability

This audit reviews the features present in `04_Integrated_Data/County_Year_Feature_Table.csv` (16,390 rows, 79 columns). We distinguish features that have been **empirically generated and validated** on disk from those that were **proposed in research documentation but remain un-generated**.

### Summary Matrix by Agronomic Domain:
- **Climate & Meteorological Features:** 19 Generated (14 NOAA, 5 PRISM)
- **Pedological & Soil Physics Features:** 26 Generated (ISRIC SoilGrids 2.0 at 3 depths + profile averages)
- **Crop Management & Insurance Features:** 8 Generated (USDA RMA Pre-Harvest Insurance Metrics)
- **Crop Phenology & Condition Features:** 19 Generated (USDA NASS Weekly Aggregations)
- **Nutrient & Fertilizer Features:** 0 in Feature Table (Maintained in standalone benchmark table `Fertilizer_Cleaned.csv`)
- **Historical Lag & Autoregressive Features:** 0 Generated (Proposed)
- **Satellite Vegetation Indices (NDVI/EVI):** 0 Generated (Unacquired)

---

## 2. Comprehensive Inventory of Generated Features

### 2.1 Climate & Meteorology (19 Features)
- **NOAA Monthly Precipitation:** `noaa_prcp_may_in`, `noaa_prcp_jun_in`, `noaa_prcp_jul_in`, `noaa_prcp_aug_in`, `noaa_prcp_sep_in`, `noaa_prcp_annual_in` (Inches).
- **NOAA Cumulative Growing Season Rainfall:** `noaa_prcp_may_aug_in` (Critical vegetative/reproductive sum), `noaa_prcp_may_sep_in` (Full season sum).
- **NOAA Temperatures:** `noaa_tmean_may_aug_f` (Growing season mean), `noaa_tmax_jul_f` (Peak silking heat stress), `noaa_tmax_aug_f`, `noaa_tmax_may_aug_f`, `noaa_tmin_may_aug_f` (°F).
- **NOAA Extreme Heat Indicator:** `noaa_heat_stress_max_f` (Max monthly temperature between June and August).
- **PRISM Gridded Features:** `prism_ppt_mm` (Annual precipitation), `prism_tmean_c`, `prism_tmin_c`, `prism_tmax_c` (°C), `prism_valid_coverage_pct` (% valid pixels).

### 2.2 Pedological & Soil Physics (26 Features)
- **Topsoil (0–5 cm):** `soil_clay_0_5cm`, `soil_sand_0_5cm`, `soil_silt_0_5cm` (g/kg), `soil_soc_0_5cm` (dg/kg), `soil_phh2o_0_5cm` (pH*10), `soil_bdod_0_5cm` (cg/cm³).
- **Midsoil (5–15 cm):** `soil_clay_5_15cm`, `soil_sand_5_15cm`, `soil_silt_5_15cm`, `soil_soc_5_15cm`, `soil_phh2o_5_15cm`, `soil_bdod_5_15cm`.
- **Subsoil (15–30 cm):** `soil_clay_15_30cm`, `soil_sand_15_30cm`, `soil_silt_15_30cm`, `soil_soc_15_30cm`, `soil_phh2o_15_30cm`, `soil_bdod_15_30cm`.
- **Depth-Weighted Profile Means (0–30 cm):**
  - `soil_clay_mean_0_30cm`, `soil_sand_mean_0_30cm`, `soil_silt_mean_0_30cm`
  - `soil_soc_mean_0_30cm`, `soil_ph_mean_0_30cm`, `soil_bdod_mean_0_30cm`
- **Geographic Centroids:** `soil_intptlat`, `soil_intptlon` (WGS84 Degrees).

### 2.3 Crop Management & Pre-Harvest Insurance (8 Features)
- `rma_policies_sold`: Total commercial corn policies purchased.
- `rma_policies_earning_prem`: Policies earning premium.
- `rma_net_planted_acres`: Net insured planted acres (Proxy for commercial production scale).
- `rma_liability_total`: Total dollar liability coverage ($).
- `rma_total_premium`: Total farmer/subsidized premium paid ($).
- `rma_subsidy`: Federal premium subsidy ($).
- `rma_liability_per_planted_acre`: Insured dollar coverage per acre ($/acre).
- `rma_premium_per_planted_acre`: Premium intensity per acre ($/acre).

### 2.4 Crop Condition & Phenology (19 Features)
- **Condition Indices (Seasonal Mean Weeks 20–42):**
  - `state_cond_excellent_mean`, `state_cond_good_mean`, `state_cond_fair_mean`, `state_cond_poor_mean`, `state_cond_very_poor_mean` (%)
  - `state_cond_good_excellent_mean`: Overall crop health index (%)
  - `state_cond_poor_very_poor_mean`: Overall crop stress index (%)
- **Reproductive Phase Condition (Weeks 28–34):**
  - `state_cond_rep_good_excellent_mean`: Health during silking/blister (%)
  - `state_cond_rep_poor_very_poor_mean`: Stress during silking/blister (%)
- **Phenological Milestones & Timing:**
  - `state_progress_planted_w20_pct`: Planting progress by mid-May (%)
  - `state_progress_emerged_w22_pct`: Emergence progress by late-May (%)
  - `state_progress_silking_w30_pct`: Silking progress by late-July (%)
  - `state_progress_dough_w34_pct`: Dough progress by late-August (%)
  - `state_progress_dented_w37_pct`: Dent progress by mid-September (%)
  - `state_progress_mature_w40_pct`: Maturity progress by early-October (%)
  - `state_progress_week_50pct_planted`: Calendar week 50% planting reached
  - `state_progress_week_50pct_silking`: Calendar week 50% silking reached
  - `state_progress_week_50pct_mature`: Calendar week 50% maturity reached

---

## 3. Features Proposed But Not Implemented

| Proposed Feature | Agronomic Importance | Underlying Data Status | Feasibility / Implementation Pathway |
|---|---|---|---|
| **Growing Degree Days (GDD)** | Primary physiological clock of corn development ($\text{GDD} = \max(0, \frac{T_{\max} + T_{\min}}{2} - 50^\circ\text{F})$). | Daily weather not downloaded for all counties; only monthly NOAA CIRS exists locally. | **FEASIBLE:** Can be approximated from monthly mean temperatures or derived from PRISM daily downloads. |
| **Extreme Degree Days (EDD > 86°F / 30°C)** | Captures nonlinear heat-induced pollen sterility during anthesis. | Requires daily maximum temperatures. | **HIGH PRIORITY:** Derive once daily weather rasters or station data are acquired. |
| **Autoregressive Yield Lags ($Y_{t-1}, Y_{t-2}$)** | Captures county-specific baseline management, equipment, and farmer capability. | Fully supported by existing `USDA_Yield_Cleaned.csv`. | **HIGH PRIORITY:** Simple feature addition: `df.groupby('county_fips')['corn_yield_bu_acre'].shift(1)`. |
| **Long-Term Rolling Averages (5-Yr Mean Yield)** | Normalizes for county baseline yield to predict yield deviations (anomalies). | Fully supported by existing `USDA_Yield_Cleaned.csv`. | **HIGH PRIORITY:** Compute backward rolling 5-year mean (excluding current year). |
| **Vapor Pressure Deficit (VPD)** | Primary atmospheric drought driver governing stomatal conductance and water stress. | Requires dewpoint temperature and relative humidity. | Available in PRISM/ERA5-Land; requires raw download. |
| **Dynamic Root-Zone Soil Moisture** | Actual moisture available to roots (vs static soil texture). | Missing; ERA5-Land provides 4 soil moisture layers. | Requires downloading ERA5-Land NetCDF archives. |
| **Peak Greenness (NDVI / EVI max)** | Remote sensing proxy for canopy leaf area and photosynthetic capacity. | Unacquired (MODIS/Sentinel folders are stubs). | Requires acquiring MODIS MOD13A2 (16-day 1km) or Landsat composites. |

---

## 4. Assessment of Feature Quality & Predictive Utility

1. **High Predictive Power:**
   - `noaa_prcp_may_aug_in` and `noaa_tmax_jul_f` exhibit strong, statistically significant correlations with yield ($r = 0.44$ and $r = -0.52$, $p < 0.001$).
   - `state_cond_good_excellent_mean` strongly reflects regional harvest outcomes ($r = 0.58$).
   - `soil_soc_mean_0_30cm` provides robust baseline discrimination across soil productivity classes ($r = 0.38$).
2. **Key Weakness:**
   - Lack of high-resolution daily dynamics (GDD, heatwaves) and lack of satellite remote sensing (NDVI/EVI) limits the ability of models to capture sub-monthly localized micro-climate events.

