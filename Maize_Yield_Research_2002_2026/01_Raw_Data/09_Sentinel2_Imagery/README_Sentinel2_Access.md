# Sentinel-2 MSI Surface Reflectance (L2A) Access Guide (Copernicus)

## Overview & Sensor Coverage Limitation (Important Research Note)
Sentinel-2 is a European Space Agency (ESA) twin-satellite constellation:
- **Sentinel-2A Launch**: June 23, 2015
- **Sentinel-2B Launch**: March 7, 2017
- **Temporal Resolution**: 5-day revisit at equator with 2 satellites
- **Spatial Resolution**: 10m (B2, B3, B4, B8), 20m (B5, B6, B7, B8A, B11, B12), 60m (B1, B9, B10)
- **CRITICAL TEMPORAL SCOPE FINDING**: Sentinel-2 data is only available from late 2015 to 2026. **It cannot cover the 2002–mid-2015 historical period.** For the complete 2002–2026 historical horizon, MODIS and Landsat 5/7/8 are required.

## Key Bands for Maize Phenology & Nitrogen Status
- Band 4 (Red, 665 nm)
- Band 5, 6, 7 (Vegetation Red Edge, 705 nm, 740 nm, 783 nm: Critical for chlorophyll and nitrogen status)
- Band 8 (Broad NIR, 842 nm)
- Band 8A (Narrow NIR, 865 nm)
- Band 11, 12 (SWIR, 1610 nm, 2190 nm: Canopy water content)

## Access Requirements & Blocker Documentation
1. **Copernicus Data Space Ecosystem (CDSE) Account**: Free registration at https://dataspace.copernicus.eu/
2. **OData and STAC REST APIs**:
   - Authentication Endpoint: `https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token`
   - STAC Catalog: `https://catalogue.dataspace.copernicus.eu/stac/`
   - OData Catalog: `https://catalogue.dataspace.copernicus.eu/odata/v1/Products`
3. **Data Volume**: A single Sentinel-2 L2A tile is ~600 MB–1 GB. Downloading raw imagery for the Corn Belt for a single growing season requires >5 TB.

