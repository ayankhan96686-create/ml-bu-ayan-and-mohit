# CROP PHENOLOGY & GROWTH-STAGE MODELING REPORT

## Stage-Specific Stress Timing
Maize yield sensitivity is highly non-linear with respect to ontogenetic stage:
- **Vegetative Stage (V1–V12):** Mild water stress reduces plant height and canopy volume but causes minimal irreversible kernel loss.
- **Flowering / Silking (R1):** Extreme heat (>35°C / 95°F) desiccates pollen grains and causes silk drying, leading to irreversible kernel abortion.
- **Grain Filling (R2–R5):** Elevated August VPD accelerates leaf senescence, reducing individual kernel test weight.

By interacting state-level crop progress timings with county meteorological observations (`silking_thermal_interaction`, `grain_filling_stress`), models achieved a severe drought tail RMSE reduction to **30.15 bu/acre** with reduced positive tail bias.
