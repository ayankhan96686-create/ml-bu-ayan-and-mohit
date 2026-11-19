# USDA NRCS Gridded SSURGO (gSSURGO) Database Access Guide

## Overview
The Gridded Soil Survey Geographic (gSSURGO) database is produced by the National Cooperative Soil Survey (NCSS) and USDA Natural Resources Conservation Service (NRCS). It combines traditional SSURGO tabular attribute databases with a 10-meter raster of map unit delineations.

## Key Soil Physical & Chemical Properties
- **Available Water Capacity (AWC)**: `awc_r` (cm/cm)
- **Soil Organic Matter (SOM)**: `om_r` (%)
- **Soil Reaction (pH)**: `ph1to1h2o_r`
- **Particle Size Fractions**: `claytotal_r`, `sandtotal_r`, `silttotal_r` (%)
- **Bulk Density**: `dbthirdbar_r` (g/cm^3)
- **Depth to Restrictive Layer**: `resdept_r` (cm)
- **Hydrologic Soil Group**: `hydgrp` (A, B, C, D)
- **Corn Non-Irrigated Productivity Index / NCCPI (National Commodity Crop Productivity Index)**: `nccpi3corn`

## Access Repository & Important Transition
- **Geospatial Data Gateway Decommissioning**: The USDA NRCS Geospatial Data Gateway was permanently retired as of March 31, 2026.
- **Current Official Download Location**: USDA-NRCS official Box repository:
  - Portal: https://www.nrcs.usda.gov/resources/data-and-reports/gridded-soil-survey-geographic-database-gssurgo
  - Box Repository: https://nrcs.app.box.com/v/soils
  - Navigate to `State Databases - soils` to download ESRI File Geodatabases for:
    - `gSSURGO_IA.gdb.zip` (Iowa)
    - `gSSURGO_IL.gdb.zip` (Illinois)
    - `gSSURGO_IN.gdb.zip` (Indiana)
    - `gSSURGO_MN.gdb.zip` (Minnesota)
    - `gSSURGO_MO.gdb.zip` (Missouri)
    - `gSSURGO_NE.gdb.zip` (Nebraska)
    - `gSSURGO_OH.gdb.zip` (Ohio)
    - `gSSURGO_SD.gdb.zip` (South Dakota)
    - `gSSURGO_WI.gdb.zip` (Wisconsin)
- **Alternative Programmatic Access (USDA Soil Data Access - SDA)**:
  - Endpoint: `https://sdmdataaccess.nrcs.usda.gov/`
  - SQL REST service allows tabular querying by county FIPS without downloading multi-gigabyte geodatabases.

## Temporal Nature Note
Soil properties from gSSURGO represent static/pedogenic baseline soil characteristics (reflecting centuries of pedogenesis) rather than dynamic annual weather. In accordance with research rules, static soil maps must not be duplicated into 25 artificial annual observations.

