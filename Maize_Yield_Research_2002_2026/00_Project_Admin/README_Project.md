# Fresh Corn (Maize) Yield Prediction Research Dataset (2002–2026)

## Academic Research Data Repository
This repository contains the curated, verified, and traceably organized public agricultural research datasets for county-level Corn Grain Yield Prediction across the United States Corn Belt over the 25-year study period from 2002 to 2026 inclusive.

### Repository Root
`E:\ml do not open\Maize_Yield_Research_2002_2026\`

---

## Directory Architecture
```
Maize_Yield_Research_2002_2026/
│
├── 00_Project_Admin/                  # Administrative records, governance, and registers
│   ├── README_Project.md              # Project overview and directory guide
│   ├── Research_Scope.md              # Spatial, temporal, and methodological definitions
│   ├── Study_Area_Counties_FIPS.csv   # Master list of all 814 study counties with FIPS
│   ├── Source_Registry.csv            # Detailed catalog of all investigated data providers
│   ├── Download_Log.csv               # Complete log of downloaded files, sizes, and URLs
│   ├── Data_Coverage_Matrix.csv       # Source-by-year availability matrix (2002-2026)
│   ├── Data_Dictionary.csv            # Field-level metadata, units, and data types
│   ├── Missing_Data_Report.csv        # Comprehensive documentation of gaps & anomalies
│   ├── Licence_and_Citation_Records.md# Formal scientific citations and data licensing
│   └── Collection_Summary.md          # High-level executive collection overview
│
├── 01_Raw_Data/                       # Original, unmodified source files and archives
│   ├── 01_USDA_NASS_Yield/            # County-level corn grain yield CSVs (2002-2025)
│   ├── 02_USDA_CDL_Crop_Classification/ # 30m CDL raster GeoTIFF, legend, and code tables
│   ├── 03_PRISM_Weather/              # 4km gridded weather zips (ppt, tmin, tmax, tmean)
│   ├── 04_ERA5_Land_Weather/          # ERA5-Land technical specifications & CDS API guide
│   ├── 05_NOAA_Weather/               # NCEI ClimDiv county climate files (1895-2026)
│   ├── 06_MODIS_NDVI_EVI/             # MOD13Q1 250m specifications & AppEEARS guide
│   ├── 07_MODIS_LAI_FPAR/             # MOD15A2H 500m specifications & extraction guide
│   ├── 08_Landsat_Imagery/            # Landsat 5/7/8/9 30m Level-2 technical guide
│   ├── 09_Sentinel2_Imagery/          # Sentinel-2 MSI L2A specifications & temporal scope
│   ├── 10_Soil_gSSURGO/               # USDA NRCS gSSURGO Box repository & SDA REST guide
│   ├── 11_Soil_SoilGrids/             # ISRIC SoilGrids 250m master VRT rasters (soc, clay...)
│   ├── 12_Planting_and_Crop_Management/ # USDA RMA insurance/acreage & NASS crop progress
│   ├── 13_Fertilizer_and_Nutrients/   # USDA ERS workbook & NASS chemical application CSV
│   ├── 14_County_and_State_Boundaries/ # US Census TIGER/Line state & county shapefiles
│   └── 15_Other_Useful_Agricultural_Data/ # NASA POWER daily weather time series & ASD maps
│
├── 02_Extracted_Data/                 # Source-specific extracted files and filtered subsets
│   ├── 03_PRISM_Weather/              # Extracted PRISM gridded rasters and headers
│   ├── 05_NOAA_Weather/               # Filtered 2002-2026 county climate CSVs
│   ├── 12_Planting_and_Crop_Management/ # Extracted RMA pipe-delimited files & corn subset
│   └── 14_County_and_State_Boundaries/ # Extracted Census TIGER county and state shapefiles
│
├── 03_Processed_Data/                 # Intentionally empty for subsequent modeling phase
│   └── README.md                      # Governance notice on Phase 2 processing
│
├── 04_Manifests_and_Checksums/        # Data integrity verification and file inventories
│   ├── SHA256_Checksums.csv           # SHA-256 cryptographic hashes for all project files
│   ├── File_Inventory.csv             # Complete metadata inventory of every downloaded file
│   └── Duplicate_File_Report.csv      # Audit verifying absence of unintended duplicate data
│
└── 05_Reports/                        # Formal research and data verification reports
    ├── Dataset_Collection_Report.md   # Comprehensive dataset acquisition report
    ├── Year_Wise_Coverage_Report.md   # Chronological analysis from 2002 to 2026
    ├── Data_Quality_Report.md         # Rigorous audit of missingness, units, & formats
    └── Unavailable_Data_Report.md     # Traceable documentation of blockers and 2026 status
```

---

## Data Governance & Integrity Principles
1. **Zero Fabrication**: No missing observation, unobserved year, or provisional statistic has been interpolated, synthetic, or imputed.
2. **Provenance Traceability**: Every downloaded file is logged with its exact source URL, retrieval date, official version, and SHA-256 checksum.
3. **Format Preservation**: Original ZIP archives, GeoTIFFs, VRT rasters, shapefiles, Excel workbooks, and CSV tables are preserved intact in `01_Raw_Data/`.
4. **Distinct 2026 Documentation**: 2026 provisional or unpublished observations are strictly segregated from finalized historical statistics.

