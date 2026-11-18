# Unavailable Data & Access Requirements Report

## 1. Overview
In academic data acquisition, transparent documentation of access restrictions, unavailable vintages, and physical temporal boundaries is as critical as recording successfully downloaded files. This report catalogues every dataset requiring user credentials, physical temporal limits, and unpublished 2026 releases.

---

## 2. Portals Requiring User Accounts, Authentication, or Approval

### A. ECMWF / Copernicus Climate Data Store (ERA5-Land)
- **Status**: Automated direct download blocked without user API key.
- **Requirement**: Free personal account on Copernicus CDS (https://cds.climate.copernicus.eu/).
- **Action Required by User**:
  1. Register on CDS.
  2. Accept the ERA5-Land license terms.
  3. Create `~/.cdsapirc` containing your Personal Access Token.
- **Ready-to-Run Script**: Complete Python script configured for the 9-state Corn Belt bounding box provided in `01_Raw_Data/04_ERA5_Land_Weather/README_ERA5_Land_Access.md`.

### B. NASA Earthdata / LP DAAC (MODIS MOD13Q1 & MOD15A2H)
- **Status**: Direct file retrieval blocked by NASA Earthdata Login gate.
- **Requirement**: Free user registration at NASA Earthdata (https://urs.earthdata.nasa.gov/).
- **Action Required by User**:
  1. Register and authorize AppEEARS and LP DAAC in your Earthdata profile.
  2. Configure `%USERPROFILE%\_netrc` with login credentials.
- **Ready-to-Run Script**: Provided in `01_Raw_Data/06_MODIS_NDVI_EVI/` and `07_MODIS_LAI_FPAR/`.

### C. USGS EROS / EarthExplorer (Landsat Collection 2)
- **Status**: Automated API queries blocked by USGS EROS M2M token requirement.
- **Requirement**: Free user registration at USGS EROS (https://ers.cr.usgs.gov/).
- **Action Required by User**:
  1. Generate M2M API token.
  2. For large-scale batch processing, query USGS Landsat AWS Open Data bucket (`s3://usgs-landsat/collection02/level-2/`) without egress fees.
- **Documentation**: Provided in `01_Raw_Data/08_Landsat_Imagery/README_Landsat_Access.md`.

### D. Copernicus Data Space Ecosystem (Sentinel-2 MSI)
- **Status**: Direct API download blocked by CDSE OAuth2 credentials.
- **Requirement**: Free account on https://dataspace.copernicus.eu/.
- **Action Required by User**: Generate OAuth2 Client ID / Secret in user settings.
- **Documentation**: Provided in `01_Raw_Data/09_Sentinel2_Imagery/README_Sentinel2_Access.md`.

### E. USDA NRCS gSSURGO Box Cloud Transition
- **Status**: Historical Geospatial Data Gateway decommissioned on March 31, 2026.
- **Requirement**: Direct downloads are hosted on Box (https://nrcs.app.box.com/v/soils).
- **Documentation**: Navigation guide and Soil Data Access (SDA) REST query alternatives provided in `01_Raw_Data/10_Soil_gSSURGO/README_gSSURGO_Access.md`.

---

## 3. Physical Sensor & Historical Availability Boundaries

### A. Sentinel-2 Constellation Temporal Bound
- **Physical Boundary**: Sentinel-2A was launched in June 2015. Sentinel-2 data **does not physically exist** between 2002 and mid-2015.
- **Methodological Solution**: Long-term optical continuity (2002–2026) must rely on Terra/Aqua MODIS and Landsat 5/7/8/9.

### B. Landsat 7 Scan Line Corrector Failure (SLC-off)
- **Physical Boundary**: On May 31, 2003, the Scan Line Corrector aboard Landsat 7 failed, causing recurring wedge-shaped data gaps.
- **Methodological Solution**: Utilize Landsat 5 TM (2002–2011) and Landsat 8/9 (2013–2026), or apply official USGS Level-2 gap-filling algorithms.

### C. USDA Cropland Data Layer (CDL) Pre-2008 State Coverage
- **Historical Boundary**: Full 48-state contiguous US coverage began in 2008. Prior to 2008, CDL covered only select states (e.g. IA, IL, IN, NE).
- **Methodological Solution**: Use USGS NLCD cultivated land masks and MODIS Land Cover Type (MCD12Q1) as verified alternatives for pre-2008 non-CDL state years.

---

## 4. Current 2026 Publication Delays & Provisional Status
- **USDA NASS Corn Grain Yield 2026**: Unpublished as of November 2026; final post-harvest county estimates scheduled for release in February/March 2027.
- **USDA Cropland Data Layer 2026**: Unpublished; released Q1 2027.
- **PRISM Annual Gridded Weather 2026**: Calendar year incomplete; provisional growing season months (May–Oct 2026) collected.
- **NOAA NCEI Climate 2026**: Months 1–10 collected; months 11–12 pending calendar progression.
- **USDA RMA Insurance 2026**: Active current-year business file collected; subject to final indemnity audits in spring 2027.

