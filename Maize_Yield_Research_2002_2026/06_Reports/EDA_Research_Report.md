# Exploratory Data Analysis & Agronomic Research Report

**Project Title:** Maize (Corn) Grain Yield Prediction Using Machine Learning (2002–2025)  
**Study Region:** 9 US Corn Belt States (814 Counties)  
**Temporal Domain:** 2002–2025 (16,390 Labeled County-Years)  
**Report Directory:** `E:/ml do not open/Maize_Yield_Research_2002_2026/06_Reports/`  

---

## 1. Executive Summary

This report synthesizes the empirical findings from the Exploratory Data Analysis (EDA) conducted across the integrated multi-domain dataset. A total of **15 publication-grade figures (300 DPI)** and **8 summary statistical tables** were generated in `05_Data_Analysis/`.

### Key Agronomic Findings:
1. **Target Yield Distribution:** US Corn Belt grain yield is continuous, well-conditioned, and moderately left-skewed ($\text{Skewness} = -0.684, \text{Kurtosis} = 0.812$). Mean: $157.45$ bu/acre; Median: $161.40$ bu/acre; Interquartile Range: $136.00$ to $182.20$ bu/acre.
2. **Technological Growth vs. Climate Shocks:** Regional average yield exhibits a steady upward technological drift of **$+1.95$ bu/acre/year** driven by breeding genetics and agronomic management. This trend is interrupted by severe climate shock events—most acutely the **2012 historic drought**, where regional mean yield plummeted to **$108.9$ bu/acre** (a $33\%$ drop below trend).
3. **Primary Biophysical Drivers:**
   - **Reproductive Heat Stress:** July maximum temperature ($T_{\max}$) is the single strongest linear climatic predictor of yield ($r = -0.52, p < 0.001$), reflecting pollen desiccation and reduced grain set during silking.
   - **Cumulative Moisture:** May–August precipitation correlates positively with yield ($r = 0.44, p < 0.001$).
   - **Soil Organic Carbon (SOC):** 0–30cm SOC correlates strongly with county historical baseline yield ($r = 0.38$), separating highly productive Mollisols from lower-yielding Alfisols.
   - **Crop Condition Index:** State-level seasonal % Good+Excellent correlates heavily with county harvest outcomes ($r = 0.58$).

---

## 2. Inventory of Scientific EDA Visualizations

All visualization artifacts are stored as publication-grade PNG images at 300 DPI in `05_Data_Analysis/Visualizations/`:

| Figure ID | File Name | Agronomic Subject | Primary Scientific Interpretation |
|---|---|---|---|
| **Fig 1** | `01_yield_distribution.png` | Yield Histogram + KDE | Continuous distribution centered around 160 bu/ac with left tail reflecting drought/flood loss years. |
| **Fig 2** | `02_yield_by_state_boxplot.png` | State Productivity Boxplots | Ranked hierarchy: Iowa (Median 181 bu/ac) and Illinois (178 bu/ac) lead; South Dakota (129 bu/ac) has highest variance. |
| **Fig 3** | `03_annual_yield_trend.png` | Longitudinal Yield Trend | Linear technological growth (+1.95 bu/ac/yr) punctuated by severe climate shocks (2012 drought, 2019 wet spring). |
| **Fig 4** | `04_precip_vs_yield.png` | Hydroclimate Regression | Positive linear response to May–August precipitation ($r = 0.44$), plateauing around 22–25 inches. |
| **Fig 5** | `05_heat_stress_vs_yield.png` | Thermal Stress Regression | Strong negative linear response to July $T_{\max}$ ($r = -0.52$), illustrating extreme vulnerability during flowering. |
| **Fig 6** | `06_noaa_vs_prism_agreement.png`| Sensor Cross-Validation | Strong 1:1 agreement ($R^2 = 0.94$) between NOAA CIRS and PRISM gridded annual precipitation. |
| **Fig 7** | `07_soc_vs_yield.png` | Pedology vs Productivity | Topsoil Soil Organic Carbon (0–30cm) acts as a baseline productivity anchor ($r = 0.38$). |
| **Fig 8** | `08_soil_texture_boxplot.png` | Soil Physical Fractions | Demonstrates dominance of silty clay loam and silt loam across the Corn Belt (Mean: 49.9% silt, 27.1% clay). |
| **Fig 9** | `09_top20_correlations.png` | Feature Screening | Ranks top 20 linear predictors of yield (July Tmax, Crop Condition, Growing Season Rainfall dominate). |
| **Fig 10** | `10_domain_correlation_heatmap.png`| Collinearity Structure | Captures intra-domain collinearity (e.g., July Tmax vs Season Tmean: $r = 0.81$). |
| **Fig 11** | `11_missingness_top25.png` | Data Quality Audit | Confirms that only CV% (77.2% by design) and PRISM 2023 ppt (3.86% data gap) exceed 3% nulls. |
| **Fig 12** | `12_yield_coverage_by_year.png` | Spatial Completeness | Shows consistent reporting across 632 to 788 counties per year out of 814 total. |
| **Fig 13** | `13_condition_vs_yield.png` | Crop Condition Index | Strong positive linear alignment ($r = 0.58$) between state seasonal % Good+Excellent and county yield. |
| **Fig 14** | `14_insurance_liability_vs_yield.png`| Insurance Intensity | Positive correlation ($r = 0.32$) between insured dollar liability per acre and harvest yield. |
| **Fig 15** | `15_spatial_yield_map.png` | Geographic Yield Map | Spatial map of 814 county centroids displaying high yield core (central IA/IL) vs semi-arid western margins. |

---

## 3. Summary Statistics Tables Generated

The following statistical summary CSVs are available in `05_Data_Analysis/`:
- `Yield_Analysis/yield_summary_statistics.csv`: Global target distribution metrics.
- `Yield_Analysis/yield_by_state.csv`: State-by-state mean, standard deviation, quartiles, and record counts.
- `Yield_Analysis/yield_by_year.csv`: Annual longitudinal mean yield, standard deviation, and reporting county counts.
- `Yield_Analysis/temporal_split_validation.csv`: Target distribution consistency across Train, Val, Test splits.
- `Correlation_Analysis/feature_correlations_with_yield.csv`: Complete Pearson correlation ranking for all numeric features.
- `Correlation_Analysis/high_collinearity_pairs.csv`: Screened collinear feature pairs with $|r| \ge 0.85$.
- `Missing_Data_Analysis/feature_table_missingness.csv`: Column-by-column missing value counts and percentages.

