# FINAL MODEL LIMITATIONS & ACADEMIC CAVEATS

## Transparent Boundary Conditions for Peer-Reviewed Submission
**Project Root:** `E:\ml do not open\Maize_Yield_Research_2002_2026`  
**Model Version:** `v2.0_Optimized_Production_Ensemble`  
**Date:** November 2026

---

### 1. Spatial Resolution Limitations
- **County-Level Aggregation:** The primary target variable is USDA NASS county-level average yield. This spatial unit necessarily averages out within-county microclimates, individual farm soil variations, and management differences. The model cannot predict field-scale yield variability without sub-field satellite reflectance and precision yield monitor data.

### 2. Phenological Granularity
- **State-Level Crop Progress Curves:** USDA NASS weekly crop progress and condition reports are published at the state level rather than the county level. Consequently, all counties within a given state share the same week of 50% silking or planting progress, which introduces spatial smoothing along state borders.

### 3. Irrigation Data Boundaries
- **Proxy Nature of Irrigation:** While Nebraska's statewide high-irrigation regime is explicitly modeled alongside RMA insurance liability guarantees, annual metered groundwater pumping data per county does not exist in standard USDA datasets. Extreme changes in local aquifer restrictions cannot be sensed in real time.

### 4. 2026 Prospective Verification Status
- **Unlabeled Future Target:** The prospective 2026 forecast (181.31 bu/acre) remains an out-of-sample forward prediction. It cannot be scientifically verified as accurate until USDA NASS releases official county-level crop production figures in early 2027.
