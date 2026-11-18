# County-Level Maize (Zea mays L.) Grain Yield Prediction Research (2002–2026)

## Authors
- Ayan Khan & Mohit

## Overview
This repository contains the complete research pipeline, multi-source agricultural datasets, engineered multimodal predictors, validation frameworks, machine learning models, and publication deliverables for county-level corn grain yield prediction across the 9 primary US Corn Belt states:
- **States:** Iowa (`IA`), Illinois (`IL`), Indiana (`IN`), Minnesota (`MN`), Missouri (`MO`), Nebraska (`NE`), Ohio (`OH`), South Dakota (`SD`), Wisconsin (`WI`)
- **Spatial Coverage:** 814 counties (788 primary agricultural counties + 26 monitored counties)
- **Temporal Scope:** 2002–2025 (certified historical ground truth from USDA NASS) and 2026 (prospective in-season scenario forecasts)
- **Observations:** 16,390 county-year records
- **Predictors:** 117 multi-domain features spanning remote sensing (MODIS NDVI/EVI), surface meteorology (PRISM, NOAA), soil hydrology (SoilGrids, SSURGO), management/irrigation, and temporal yield memory lags

## Project Structure
- `00_Project_Admin/`: Research charters, scopes, and administration records
- `01_Raw_Data/`: Primary geospatial and tabular data sources
- `02_Extracted_Data/`: Preprocessed climate, soil, satellite, and crop progress records
- `03_Cleaned_Data/` & `03_Processed_Data/`: Data cleaning pipelines and county harmonization
- `04_Integrated_Data/` & `04_Manifests_and_Checksums/`: Merged multi-source feature stores and SHA-256 integrity checksums
- `05_Data_Analysis/` & `05_Reports/` / `06_Reports/`: Exploratory spatial-temporal analyses and EDA
- `07_Code/`: Production python data engineering and model training pipelines
- `08_Project_Handover_Audit/` to `12_Final_Scientific_Certification/`: Multi-stage verification and audit milestones
- `13_Publication_Enhancement/` to `16_Final_Publication_Package/`: Academic manuscript drafts, visual figures, and submission packages
- `17_Phase12_Advanced_Model_Optimization/`: Advanced model architectures and multi-seed benchmark suites
- `18_Consistent_80_20_Train_Test_Split/`: Out-of-time chronological train/test split framework (2002–2020 vs 2021–2025)
- `19_Master_Implementation_80_20_Optimization/`: Master implementation containing the primary pooled stratified 80:20 benchmark, expanding chronological validation folds, Optuna hyperparameter optimization, multimodal feature ablation, leakage audit, and publication report PDF

## Master Publication Report
The complete publication report is compiled and available at:
`19_Master_Implementation_80_20_Optimization/MASTER_MAIZE_YIELD_OPTIMIZATION_REPORT.pdf`

## Key Results (Master Optimization Holdout Benchmark)
- **Champion Architecture (Pooled Holdout):** Optimized XGBoost Regressor ($R^2 = 0.8986$, $\text{RMSE} = 11.67\text{ bu/acre}$, $\text{MAE} = 8.74\text{ bu/acre}$, $\text{MAPE} = 6.71\%$)
- **Top Ensemble:** Ridge Stacking Meta-Learner ($R^2 = 0.8931$, $\text{RMSE} = 11.98\text{ bu/acre}$)
- **Drought Stress Tail ($<120$ bu/acre):** Tail RMSE = $17.15\text{ bu/acre}$
- **Chronological Out-of-Time Forecasting:** $R^2 = 0.7276\text{--}0.7610$, $\text{RMSE} = 15.37\text{--}16.41\text{ bu/acre}$
