# Dataset Collection Report: Maize Yield Research Project (2002–2026)

## 1. Project Background and Objective
This report documents the autonomous data acquisition and engineering workflow conducted to establish the foundation for a county-level Corn (Maize) Grain Yield Prediction research project. The objective is to assemble high-fidelity, publicly available datasets from authoritative agricultural, meteorological, pedological, and earth observation agencies covering the United States Corn Belt across 25 continuous calendar years (2002 through 2026).

---

## 2. Study Area & Administrative Definition
The research study area comprises nine contiguous states forming the core of the US Corn Belt:
- **Iowa (IA)** — State FIPS `19` (99 Counties)
- **Illinois (IL)** — State FIPS `17` (102 Counties)
- **Indiana (IN)** — State FIPS `18` (92 Counties)
- **Minnesota (MN)** — State FIPS `27` (87 Counties)
- **Missouri (MO)** — State FIPS `29` (115 Counties)
- **Nebraska (NE)** — State FIPS `31` (93 Counties)
- **Ohio (OH)** — State FIPS `39` (88 Counties)
- **South Dakota (SD)** — State FIPS `46` (66 Counties)
- **Wisconsin (WI)** — State FIPS `55` (72 Counties)

**Total Study Counties**: Exactly **814 counties**.
Official boundary geometry and FIPS crosswalks were downloaded from the US Census Bureau TIGER/Line 2024 repository and compiled into `00_Project_Admin/Study_Area_Counties_FIPS.csv`.

---

## 3. Dataset Collection Details

| Category | Source / Organization | Downloaded Files | Extracted Files | Total Size | Time Horizon | Resolution | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Corn Yield (Primary Target)** | USDA NASS Quick Stats | 10 CSVs | — | 6.18 MB | 2002–2025 | County / Annual | Final Historical (2026 Unpublished) |
| **CDL Crop Classification** | USDA NASS CropScape | 3 files | — | 124.09 MB | 2008–2025 | 30m / Annual | Valid Raster & Legends |
| **PRISM Gridded Climate** | PRISM Climate Group / OSU | 119 ZIPs | 732 rasters | 588.36 MB | 2002–2026 | 4km / Annual & Monthly | Final 2002–2025; Provisional 2026 |
| **NOAA County Climate** | NOAA NCEI ClimDiv | 4 TXT | 4 CSVs | 158.70 MB | 2002–2026 | County / Monthly | Complete (through Oct 2026) |
| **ISRIC SoilGrids 2.0** | ISRIC - World Soil Information | 20 files | — | 88.11 MB | Baseline | 250m / 6 standard depths | Static Pedogenic Reference |
| **Crop Management & Insurance** | USDA RMA Summary of Business | 25 ZIPs | 26 files | 719.68 MB | 2002–2026 | County / Annual | Complete 25-Year Series |
| **Crop Phenology & Condition** | USDA NASS Quick Stats | 2 CSVs | — | 10.81 MB | 2002–2026 | State / Weekly | Complete 25-Year Series |
| **Fertilizer & Chemicals** | USDA ERS & NASS Quick Stats | 2 files | — | 7.51 MB | Historical | State / Periodic | Survey Vintages |
| **Administrative Boundaries** | US Census Bureau TIGER/Line | 3 files | 14 files | 231.63 MB | 2024 Vintage | Vector Polygons | Full 814 Counties Validated |
| **Supplementary Climate** | NASA POWER Agroclimatology | 1 CSV | — | 0.39 MB | 2002–2026 | Point / Daily | Continuous 2002–2026 |
| **Project Admin & Manifests** | Internal Data Engineering | 20 files | — | 1.82 MB | Complete | Multi-tabular | SHA-256 Verified |
| **TOTAL** | **Multi-Agency Federation** | **209 Raw** | **776 Extracted** | **1,924.72 MB (~1.88 GB)** | **2002–2026** | **Multi-scale** | **Verified & Ready** |

---

## 4. Methodological Compliance & Safety Protocols
1. **Disk Space Management**: Prior to data downloading, storage capacity was audited. The non-system `E:\` drive possessed 97.5 GB free storage. Total project footprint was constrained to 1.88 GB (<2% of available capacity), preserving 95.6 GB of operational headroom.
2. **Provenance & Cryptographic Auditing**: Every raw file was hashed using SHA-256. Cryptographic manifests are recorded in `04_Manifests_and_Checksums/SHA256_Checksums.csv`.
3. **Archive Preservation**: All original `.zip`, `.xls`, `.tif`, and `.txt` files remain unmodified in `01_Raw_Data/`. Extractions and downstream subsets reside exclusively in `02_Extracted_Data/`.
4. **Clean Boundary Principle**: `03_Processed_Data/` has been left clean, accompanied by a governance notice dedicating it to Phase 2 modeling.

