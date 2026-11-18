# Research Scope Specification: Corn Grain Yield Prediction (2002–2026)

## 1. Research Topic
County-level annual Corn Grain Yield Prediction across the United States Corn Belt using multi-source agricultural, meteorological, pedological, remote sensing, and management data.

## 2. Geographical Scope: The US Corn Belt
The provisional study area encompasses the nine primary agricultural states defining the core of the US Corn Belt:
1. **Iowa (IA)** — State FIPS `19` (99 Counties)
2. **Illinois (IL)** — State FIPS `17` (102 Counties)
3. **Indiana (IN)** — State FIPS `18` (92 Counties)
4. **Minnesota (MN)** — State FIPS `27` (87 Counties)
5. **Missouri (MO)** — State FIPS `29` (115 Counties)
6. **Nebraska (NE)** — State FIPS `31` (93 Counties)
7. **Ohio (OH)** — State FIPS `39` (88 Counties)
8. **South Dakota (SD)** — State FIPS `46` (66 Counties)
9. **Wisconsin (WI)** — State FIPS `55` (72 Counties)

**Total Included Study Counties**: Exactly **814 counties**.
All county boundaries, names, and FIPS codes are verified against the official US Census Bureau TIGER/Line 2024 boundary registry and documented in `Study_Area_Counties_FIPS.csv`.

### Geographical Assumptions & Corrections
- **Independent Cities / Consolidated Municipalities**: Missouri includes one independent city (St. Louis City, FIPS `29510`), which is treated as a county-equivalent by the US Census Bureau. Agricultural statistics for St. Louis City are typically suppressed or aggregated into 'OTHER COUNTIES'.
- **Shannon County (SD) Renaming**: Shannon County, South Dakota (former FIPS `46113`) was officially renamed Oglala Lakota County (new FIPS `46102`) in 2015. Both FIPS crosswalks are preserved to maintain longitudinal integrity.
- **Study Boundary Integrity**: No state or county has been excluded or substituted.

## 3. Temporal Scope: 2002 to 2026 (25 Calendar Years Inclusive)
- **Start Year**: 2002 (Corresponds to widespread deployment of Terra MODIS satellite observations, PRISM gridded climate coverage, and continuous digital USDA records).
- **End Year**: 2026 (Current calendar year).
- **Temporal Horizon**: 25 continuous calendar years.

### Treatment of the 2026 Observation Year
- As of November 2026, the 2026 crop calendar year is concluding.
- **Yield Labels**: Final county-level Corn Grain Yield estimates from USDA NASS for the 2026 crop year are scheduled for official release in February–March 2027. Consequently, 2026 county yield labels are officially unpublished.
- **Weather Observations**: NOAA NCEI monthly county weather and PRISM 4km gridded rasters for 2026 are published and collected as **provisional** data through October 2026 (covering the complete 2026 maize growing season).
- **Crop Progress**: USDA NASS weekly crop progress and condition reports are collected and finalized through November 2026.
- **Crop Insurance**: USDA RMA 2026 Summary of Business flat files reflect active current-year crop policies.
- **Rule**: Under no circumstances are 2026 yield labels fabricated, interpolated, or projected as actual ground-truth observations.

