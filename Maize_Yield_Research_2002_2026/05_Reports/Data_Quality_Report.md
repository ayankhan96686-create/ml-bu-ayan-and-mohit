# Data Quality Report: Maize Yield Research Dataset (2002–2026)

## 1. Executive Summary
This quality audit evaluates the integrity, format compliance, spatial/temporal consistency, and statistical fidelity of the collected datasets.

---

## 2. Yield Ground Truth Quality Audit
- **File Name**: `USDA_NASS_Corn_Grain_Yield_9States_2002_2026_723B6328-B866-3F54-AB6D-07F62EE064E9.csv`
- **Total Records**: 16,956 county-year observations across 2002–2025.
- **Grain vs. Silage Distinctions**:
  - Exact queried Data Item: `"CORN, GRAIN - YIELD, MEASURED IN BU / ACRE"`
  - Corn for Silage (`"CORN, SILAGE - YIELD, MEASURED IN TONS / ACRE"`), sweet corn, and popcorn were strictly segregated and excluded from the yield target file.
- **Value Preservation**: Source values and units (`BU / ACRE`) remain unaltered.
- **Suppressed Observations**: Observations suppressed by USDA NASS to protect individual producer confidentiality (e.g. `(D)` in older tables) or statistical minimum survey responses are flagged with their official NASS CV (Coefficient of Variation) percentage.
- **Residual Categories**: State-level residual categories (`"OTHER COUNTIES"`) are retained in the raw file with ANSI code `99` and preserved without artificial spatial allocation.
- **Missing Value Fabrication**: Zero values were imputed or interpolated. The year 2026 is properly logged as unobserved.

---

## 3. Climate Data Consistency & Format Audit
- **NOAA NCEI County Climate**:
  - Record layout: 11-character fixed identifier + 12 monthly values.
  - Value parsing: Precipitation in hundredths of inches; Temperature in tenths of degrees Fahrenheit.
  - Missing value sentinel: `-9.99` or `-99.99` strictly identified. 2026 records contain valid observations for months 1–10 and `-99.99` sentinels for future months 11–12.
- **PRISM Gridded Weather**:
  - File format: ESRI Band Interleaved by Line (`.bil`), ASCII grid (`.asc`), and projection (`.prj`).
  - Spatial alignment: 4km grid cells on WGS84 geographic coordinate system (`EPSG:4326`).
  - Checksum audit: All downloaded ZIP archives passed ZIP CRC32 and SHA-256 verification before extraction.

---

## 4. Soil Data Integrity
- **Pedogenic Baseline vs. Artificial Annual Duplication**:
  - ISRIC SoilGrids 2.0 provides static soil properties (clay, sand, silt, soc, phh2o, bdod) representing millennial soil development.
  - In strict compliance with research instructions, static soil grids are stored as single master baselines across six standard depths (0–5cm, 5–15cm, 15–30cm, 30–60cm, 60–100cm, 100–200cm).
  - They have **not** been artificially duplicated into 25 redundant annual copies.

---

## 5. Duplicate File & Integrity Verification
- **Total Files Scanned**: 964 raw and extracted files.
- **Duplicate Set Findings**: Exactly 3 duplicate sets detected by cryptographic SHA-256 matching:
  1. `prism_*_us_25m_*.prj`: Identical standard WGS-84 projection parameter files bundled within PRISM ZIPs.
  2. `tl_2024_us_county.cpg` & `tl_2024_us_state.cpg`: Standard UTF-8 character encoding files from Census shapefiles.
  3. `tl_2024_us_county.prj` & `tl_2024_us_state.prj`: Standard NAD83 coordinate reference projection files from Census shapefiles.
- **Conclusion**: Zero unintended duplicate data records or overlapping datasets exist.

