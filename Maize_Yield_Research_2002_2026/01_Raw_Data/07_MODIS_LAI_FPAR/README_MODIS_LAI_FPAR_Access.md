# MODIS MOD15A2H (LAI & FPAR) Access Guide (NASA Earthdata / LP DAAC)

## Overview
The MOD15A2H Version 6.1 Moderate Resolution Imaging Spectroradiometer (MODIS) combined Leaf Area Index (LAI) and Fraction of Photosynthetically Active Radiation (FPAR) product:
- **Spatial Resolution**: 500 meters
- **Temporal Resolution**: 8-day composite
- **Temporal Coverage**: July 2002 to present (Terra)
- **Key Scientific Layers**:
  - `Lai_500m`: Leaf Area Index (scale factor 0.1, valid range 0–100 -> 0.0–10.0 m^2/m^2)
  - `Fpar_500m`: Fraction of absorbed photosynthetically active radiation (scale factor 0.01, valid range 0–100 -> 0.0–1.0)
  - `FparLai_QC`: Quality control bitmask flags detailing main algorithm vs. backup algorithm (empirical NDVI) and saturation/cloud cover
  - `FparExtra_QC`: Internal snow, cloud, aerosol, and cirrus flags

## Access Requirements & Blocker Documentation
1. **NASA Earthdata Account**: Requires free login credentials from https://urs.earthdata.nasa.gov/
2. **AppEEARS Automated Area Extraction**:
   - Access URL: https://appeears.earthdatacloud.nasa.gov/
   - Allows polygon-based extraction of growing season observations (May 1 to October 31) for 2002–2026 for the 814 Corn Belt study counties.
3. **Data Volume Considerations**:
   - Downloading raw MODIS sinusoidal tiles across 25 years represents ~500 GB of HDF files.
   - Recommended approach: Use AppEEARS API or Google Earth Engine catalog (`MODIS/061/MOD15A2H`) to extract county-level zonal mean LAI and FPAR profiles.

