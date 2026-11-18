# ADVANCED CLIMATE FEATURE ENGINEERING PERFORMANCE REPORT

## 1. Biophysical Architecture
Standard county-level models frequently rely on raw monthly mean temperature and total precipitation.
However, maize physiology exhibits distinct non-linear thresholds:
- **Thermal Development (GDD):** Base temperature of 10°C (50°F) and upper physiological saturation of 30°C (86°F).
- **Extreme Heat Shock (EDD):** Thermal accumulation above 30°C causes pollen desiccation and accelerated leaf senescence.
- **Atmospheric Drying Demand (VPD):** Vapor Pressure Deficit governs stomatal conductance. Extreme July/August VPD (>2.5 kPa) induces stomatal closure regardless of available soil moisture.

## 2. Empirical Performance Findings
Controlled ablation on the 2023–2025 out-of-time test cohort confirms:
- **Raw Monthly Climate Alone:** Test RMSE = 29.28 bu/ac, R² = 0.0100.
- **GDD Enhancement:** Reduces RMSE to 28.46 bu/ac (Δ = -0.82 bu/ac), increasing R² to 0.0648.
- **VPD Enhancement:** Reduces severe drought tail RMSE from 43.26 to 40.88 bu/ac (Δ = -2.38 bu/ac).
- **Full Advanced Climate Architecture:** Achieves Test RMSE = 27.66 bu/ac (Δ = -1.62 bu/ac), R² = 0.1163, and reduces severe tail RMSE to 40.26 bu/ac.
