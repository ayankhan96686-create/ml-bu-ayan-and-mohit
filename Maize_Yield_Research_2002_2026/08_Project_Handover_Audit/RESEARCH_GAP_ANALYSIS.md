# Independent Scientific Review & Research Gap Analysis

**Project:** Maize/Corn Grain Yield Prediction Using Machine Learning (2002–2025)  
**Root Directory:** `E:/ml do not open/Maize_Yield_Research_2002_2026/`  
**Audit Date:** 2026-11-10  
**Auditor Role:** Independent Senior Research Reviewer & Agricultural Data Scientist  

---

## 1. Executive Summary of Research Gaps

As an independent scientific reviewer examining this project for academic rigor and machine learning reproducibility, I have evaluated the data infrastructure, agronomic assumptions, feature engineering, and modeling readiness.

While the project has assembled an impressive, 100% authentic multi-domain observational dataset (16,390 labeled county-years with zero synthetic data fabrication), **several significant scientific, methodological, and data acquisition gaps exist** that must be addressed prior to model benchmarking and publication.

---

## 2. Prioritized Ranking of Research Gaps

| Rank | Gap Category | Specific Research Gap Description | Severity | Impact on Research / Modeling | Recommended Remediation |
|---|---|---|---|---|---|
| **1** | **Missing Data** | **PRISM 2023 Precipitation Raster Missing.** | **CRITICAL** | Year 2023 PRISM precipitation has 632 missing values (3.86% of dataset). Models utilizing PRISM features cannot train or evaluate cleanly on 2023 without imputation or fallback. | Download the single missing GeoTIFF (`prism_ppt_us_25m_2023.tif`) from Oregon State PRISM FTP and re-run `process_prism.py`. |
| **2** | **Missing Data** | **Satellite Remote Sensing (NDVI, EVI, LAI) Unacquired.** | **CRITICAL** | Subfolders in `01_Raw_Data` (06_MODIS, 07_MODIS_LAI, 08_Landsat, 09_Sentinel) are empty stubs containing only README files. Modern agricultural ML relies heavily on canopy vegetation indices. | Download MODIS MOD13A2 (16-day 1km NDVI/EVI) or Google Earth Engine county-aggregated time series for 2002–2025. |
| **3** | **Missing Data** | **USDA Cropland Data Layer (CDL) Incomplete.** | **HIGH** | Only 2020 Iowa CDL was downloaded (124 MB). The remaining 8 states and 24 years are missing. CDL is needed to mask non-crop land cover before computing climate/soil zonal stats. | Use Google Earth Engine or USDA CropScape API to extract pure corn-pixel masks per county, rather than downloading hundreds of gigabytes of raw GeoTIFFs. |
| **4** | **Methodology** | **County Centroid Sampling vs. Pure Cropland Masking.** | **HIGH** | Soil and PRISM climate stats are sampled around county geometric centroids rather than exclusively over actual corn-planted acreage pixels. | In counties with mixed terrain (forests, urban areas, water bodies), centroid sampling introduces land-cover noise. Weight pixels using CDL corn masks. |
| **5** | **Feature Engineering** | **Lack of Growing Degree Days (GDD) & Extreme Degree Days (EDD).** | **HIGH** | Crop development is driven by thermal time (GDD base 50°F), and pollen sterility is triggered by acute heat (EDD > 86°F). Current features rely on monthly mean/max temperatures. | Calculate cumulative GDD (May–Aug) and count of days exceeding 86°F (30°C) using daily PRISM or NOAA GHCN station records. |
| **6** | **Feature Engineering** | **Missing Autoregressive Yield Lags & Detrending.** | **HIGH** | Corn yields have risen +1.95 bu/acre/year due to genetics and technology. Predicting raw yield conflates technology trends with annual weather shocks. | Engineer: (1) 1-year yield lag ($Y_{t-1}$), (2) 5-year rolling mean baseline, and (3) Target yield anomaly ($\Delta Y_t = Y_t - \text{Trend}_t$). |
| **7** | **Spatial Resolution** | **State-Level Broadcast of Crop Progress & Condition.** | **MEDIUM** | Crop progress and condition are recorded at state resolution, assigning identical values to northern and southern counties within a state (e.g. northern vs southern Illinois). | Accept as an aggregate macro predictor, but avoid interpreting state phenology as localized micro-phenology. |
| **8** | **Validation Design** | **Need for Spatial Block Cross-Validation.** | **MEDIUM** | Contiguous counties experience identical regional weather systems. Random cross-validation or purely temporal validation without spatial blocking risks spatial leakage. | Implement "Leave-One-State-Out" or spatial cluster cross-validation alongside temporal blocking. |
| **9** | **Data Inconsistency** | **Cook County, IL Centroid Soil Masking.** | **LOW** | Cook County, IL (`17031`) has 11 years of historical corn yield, but its centroid is masked as urban/lake by SoilGrids, producing 11 `NaN` rows. | Impute Cook County soil properties using the average of adjacent rural collar counties (Will/Kane/DuPage) or drop Cook County from row-crop training. |
| **10** | **Soil Dynamics** | **Static Soil Profiles vs. Dynamic Root-Zone Moisture.** | **MEDIUM** | SoilGrids provides static texture and organic carbon, but does not capture dynamic interannual soil water depletion or waterlogging during the growing season. | Acquire ERA5-Land volumetric soil water layers (Layer 1: 0–7cm, Layer 2: 7–28cm) across the growing season. |

---

## 3. What Should Be Fixed First (Action Plan)

1. **Immediate Fix (1 Hour):** Fetch the missing 2023 PRISM precipitation GeoTIFF from the PRISM archive to eliminate the 632 missing values in `prism_ppt_mm`.
2. **High-Yield Engineering (2 Hours):** Add autoregressive yield features ($Y_{t-1}$, 5-year rolling mean, county historical mean yield) directly into `integrate_datasets.py` from existing clean yield data.
3. **Core Research Gap (Next Major Milestone):** Acquire Google Earth Engine county-level time series for MODIS NDVI/EVI (peak greenness, greenup date) and CDL pure corn crop masks to elevate the project to top-tier academic research quality.

