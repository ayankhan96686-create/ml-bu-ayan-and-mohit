# Landsat Collection 2 Level-2 Surface Reflectance Access Guide (USGS)

## Overview & Sensor Coverage (2002–2026)
Landsat provides 30-meter multispectral surface reflectance observations spanning the entire study window:
1. **Landsat 5 TM (Thematic Mapper)**: Operating from 1984 to November 2011 (covers 2002–2011).
2. **Landsat 7 ETM+ (Enhanced Thematic Mapper Plus)**: Operating from 1999 to present (Note: May 31, 2003 Scan Line Corrector failure [SLC-off] creates data striping artifacts that require gap-filling or masking).
3. **Landsat 8 OLI/TIRS**: Operating from February 2013 to present (covers 2013–2026).
4. **Landsat 9 OLI-2/TIRS-2**: Operating from September 2021 to present (covers 2022–2026).

## Key Spectral Bands for Corn Yield Research
- Band 2 (Blue, 0.45–0.51 µm)
- Band 3 (Green, 0.53–0.59 µm)
- Band 4 (Red, 0.64–0.67 µm: Chlorophyll absorption)
- Band 5 (Near Infrared NIR, 0.85–0.88 µm: Canopy cell structure)
- Band 6 (Shortwave Infrared SWIR-1, 1.57–1.65 µm: Canopy moisture)
- Band 7 (Shortwave Infrared SWIR-2, 2.11–2.29 µm: Lignin/cellulose)
- QA_PIXEL: Bitmask flags for cloud, cirrus, cloud shadow, snow, and water.

## Access Requirements & Blocker Documentation
1. **USGS EROS Account**: Free registration at https://ers.cr.usgs.gov/
2. **USGS Machine-to-Machine (M2M) API**:
   - Documentation: https://m2m.cr.usgs.gov/
   - Endpoint: `https://m2m.cr.usgs.gov/api/api/json/stable/`
   - Requires API token generated via `login-token` endpoint.
3. **Storage Considerations**:
   - The 9 Corn Belt states span ~50 WRS-2 Worldwide Reference System Path/Row tiles.
   - Downloading full Level-2 scene bundles across 25 years represents >15 Terabytes of imagery.
   - Recommended strategy: Query cloud-optimized GeoTIFFs (COGs) on AWS Open Data (`s3://usgs-landsat/collection02/level-2/`) or compute spectral indices (NDVI, EVI, NDWI, NDRE) directly using Planetary Computer or Earth Engine to extract county-level zonal means.

