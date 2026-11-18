# NEXT-GENERATION FEATURE ENGINEERING PROPOSALS

## Agronomic, Biophysical & Methodological Justification
**Project:** County-Level Maize/Corn Grain Yield Prediction Research Project (2002–2026)  
**Phase:** Phase 8 — Advanced Feature Engineering  
**Date:** November 2026

---

### 1. Biophysical and Agronomic Foundations

Standard linear crop models and naive machine learning pipelines frequently collapse during extreme climate years because they rely on broad monthly averages (e.g. mean summer temperature and precipitation). Corn physiology, however, exhibits acute non-linear sensitivity during critical developmental stages:

```
               [ VEGETATIVE PHASE ]           [ REPRODUCTIVE PHASE ]        [ MATURATION ]
Timeline:     May 1 -------------- June 30 ---- July 1 ------------ Aug 15 -- Sept 1 ----- Oct 15
Key Stages:   Emergence -> Rapid Leaf Growth   V12 -> Silking -> Blister    Dough -> Dent -> Black Layer
Key Stress:   Wet soil, cold planting delays   Extreme Heat (>86F), VPD     Nighttime Warmth, Frost
Drivers:      GDD Base 10C, Delayed Planting   EDD, Pollen Desiccation      Kernel Filling, Senescence
```

---

### 2. Feature Family Formulations & Agronomic Logic

#### A. Thermal Regimes: GDD vs Extreme Degree Days (EDD)
- **Growing Degree Days (GDD):**
  $$	ext{GDD} = \sum_{t=1}^{N} \max\left(0, \min(30^\circ	ext{C}, T_{	ext{mean}}) - 10^\circ	ext{C}ight)$$
  Accumulation of thermal time drives vegetative node development and initiation of reproductive organs. Up to ~1400 GDD, thermal accumulation linearly benefits corn development.
- **Extreme Degree Days (EDD):**
  $$	ext{EDD} = \sum_{t=1}^{N} \max\left(0, T_{\max} - 86^\circ	ext{F} / 30^\circ	ext{C}ight)$$
  Above 86°F (30°C), maize photosynthetic machinery is inhibited by Rubisco activase deactivation, and pollen viability falls precipitously within 2 to 4 hours of exposure. This non-linear threshold is explicitly captured by `edd_july_f` and `edd_aug_f`.

#### B. Atmospheric Moisture Deficit: Vapor Pressure Deficit (VPD)
- **Vapor Pressure Deficit (VPD):**
  $$	ext{VPD} = e_s(T_{\max}) - e_a(T_{	ext{dew}})$$
  Calculated using the Tetens saturation vapor pressure equation:
  $$e_s(T) = 0.61078 \exp\left(rac{17.27 T}{T + 237.3}ight) 	ext{ [kPa]}$$
  Lobell et al. (2014) showed that modern high-density US hybrid corn has increased sensitivity to atmospheric drought. High VPD forces stomata to close even when soil moisture is adequate, truncating daily carbon assimilation and leading to kernel abortion.

#### C. Compound Heat-Drought Interaction
- Extreme heat alone reduces yield by ~1% per day; severe drought alone reduces yield by ~1.5% per day. However, when high EDD co-occurs with low July rainfall ($P < 2.0$ inches), yield losses non-linearly compound to >4% per day due to runaway sensible heat flux and rapid root-zone depletion. Formulated as `compound_heat_drought_july`.

#### D. Irrigation Heterogeneity: The Nebraska Discontinuity
- In Nebraska, >65% of corn acreage is irrigated from the Ogallala Aquifer. In Iowa and Illinois, <2% of corn acreage is irrigated. Rainfed models transferred to Nebraska fail ($R^2 = 0.1764$) because high temperature and low precipitation in Nebraska do not induce moisture starvation.
- We formulate `irrigation_regime_proxy` and `irrigation_drought_buffer` to decouple artificial groundwater applications from meteorological drought stress.

#### E. Multi-Depth Physical Soil Profile Horizons
- Rather than a single static surface soil metric, we deploy four distinct depth horizons (0–5cm, 5–15cm, 15–30cm, and 0–30cm mean) for Bulk Density, Clay, Sand, Silt, pH, and Soil Organic Carbon.
- `subsoil_clay_contrast` ($	ext{Clay}_{15-30} / 	ext{Clay}_{0-5}$) identifies dense argillic subsoil layers that store water in the deep profile during July drought.

---

### 3. Leakage Prevention and Temporal Invariants

To guarantee publication-grade scientific validity:
1. All standardized climate anomaly z-scores ($Z_{	ext{prcp}}, Z_{	ext{tmax}}$) are computed using historical county baseline means and standard deviations fitted **strictly on pre-2023 training observations**.
2. No future information from the 2023–2025 test window is leaked into feature calculations.
3. All advanced features are computable under the October 15 forecast cutoff.
