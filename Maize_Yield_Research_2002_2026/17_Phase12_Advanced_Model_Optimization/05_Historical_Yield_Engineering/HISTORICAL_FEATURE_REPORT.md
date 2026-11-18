# ADVANCED HISTORICAL YIELD ENGINEERING REPORT

## Decoupling Trend, Mean, and Volatility Memory
The baseline historical pipeline utilized only 3 features: `yield_lag_1yr`, `yield_lag_2yr`, and `yield_roll_mean_5yr`.
In Phase 12, we engineered a 13-feature comprehensive historical memory suite:
- **Higher-Order Lags:** `yield_lag_3yr`, `yield_lag_4yr` capture multi-year rotation effects (corn-corn vs. corn-soybean).
- **Multi-Window Rolling Averages:** `yield_roll_mean_3yr`, `yield_roll_median_5yr` provide robust central tendency estimates resilient to single-year drought shocks.
- **County Technological Trend Slope:** `yield_trend_3yr`, `yield_trend_5yr` capture local genetic gain and agronomic management improvements (~1.8 to 2.2 bu/ac/year).
- **Yield Volatility & Deviation:** `yield_roll_std_5yr`, `yield_coefficient_of_variation` provide intrinsic measures of county soil vulnerability to drought.
- **Empirical Validation:** Expanding from 3 to 13 memory features reduced out-of-time test RMSE from 21.81 to **20.53 bu/acre** (Δ = -1.28 bu/ac) and improved R² from 0.4506 to **0.5131**.
