# ⏱️ HISTORICAL YIELD LAG METHODOLOGY & LEAKAGE PREVENTION AUDIT

**Project:** Regional Maize/Corn Grain Yield Prediction (2002–2026)  
**Location:** `E:/ml do not open/Maize_Yield_Research_2002_2026/09_Independent_Validation/`  
**Topic:** Critical Audit of Lag Generation, Unbalanced Panels, and Trend Leakage  

---

## 1. THE FATAL FLAW OF `groupby('county_fips').shift(1)`

In standard panel data machine learning pipelines, authors frequently generate temporal lag features using:
```python
# PROPOSED NAIVE LAG IMPLEMENTATION (INCORRECT & DANGEROUS)
df['yield_lag_1'] = df.groupby('county_fips')['corn_yield_bu_acre'].shift(1)
```

### 1.1 Empirical Demonstration of the Flaw
`County_Year_Master_Dataset.csv` has **3,146 missing county-years** in the target variable `corn_yield_bu_acre`. Furthermore, in `County_Year_Feature_Table.csv`, rows with missing targets were filtered out:
```python
ml_df = merged[(merged["year"] <= 2025) & (merged["corn_yield_bu_acre"].notna())].copy()
```
When `ml_df` contains only labeled rows, the index of years within a county is **non-consecutive**.

#### Concrete Numerical Example:
Consider Reynolds County, Missouri (FIPS 29179), or an intermittent county such as Shannon County, MO:
- Suppose a county reported yields only in: **2004 (110 bu/ac)**, **2010 (125 bu/ac)**, and **2015 (140 bu/ac)**.
- If one runs `.groupby('county_fips')['corn_yield_bu_acre'].shift(1)` on this table:
  - For year 2010: `shift(1)` selects the row immediately above it &rarr; **Yield in 2004 (110 bu/ac)**!
  - The model is told that the previous year ($t-1$, 2009) yield was 110 bu/ac, when in reality that observation is **6 years stale**!
  - For year 2015: `shift(1)` selects **2010 (125 bu/ac)**, presenting a **5-year-old yield** as a 1-year lag!

### 1.2 Scientific Consequences:
1. **False Contemporaneous Signals:** Decadal agricultural technology improvements (hybrid seed genetics, higher planting density) make a 6-year-old yield systematically lower than the true expected baseline, misleading gradient boosters and linear models into predicting artificial yield crashes.
2. **Asymmetric Data Availability in 2026:** In 2026, if a county did not report in 2025 (or 2024), naive shifting will grab whatever arbitrary older year exists, causing massive prediction distortions.

---

## 2. THE MATHEMATICALLY RIGOROUS CALENDAR-YEAR JOIN

To ensure that a lag strictly reflects the true $t-1$ calendar year (and is correctly marked as `NaN` if $t-1$ was not reported), the lag must be constructed via an **explicit relational self-join on `(county_fips, year - 1)`**:

```python
def create_robust_calendar_lags(df_master, max_lags=3):
    \"\"\"
    Creates true calendar year lags by joining on county_fips and (year - lag).
    Guarantees that missing calendar years result in NaN, preventing stale observations.
    \"\"\"
    base_df = df_master.copy()
    
    # Isolate targets
    target_lookup = df_master[['county_fips', 'year', 'corn_yield_bu_acre']].dropna(subset=['corn_yield_bu_acre']).copy()
    
    for lag in range(1, max_lags + 1):
        lag_sub = target_lookup.copy()
        # Shift the key year forward by 'lag'
        lag_sub['join_year'] = lag_sub['year'] + lag
        lag_sub = lag_sub.rename(columns={'corn_yield_bu_acre': f'yield_lag_{lag}yr'})
        
        base_df = base_df.merge(
            lag_sub[['county_fips', 'join_year', f'yield_lag_{lag}yr']],
            left_on=['county_fips', 'year'],
            right_on=['county_fips', 'join_year'],
            how='left'
        ).drop(columns=['join_year'])
        
    return base_df
```

### Verification Table:
| Year | Actual Yield ($y_t$) | True $t-1$ Lag (`join`) | Naive `shift(1)` (Flawed) | Error Delta |
| :---: | :---: | :---: | :---: | :---: |
| **2010** | 165 bu/ac | 158 bu/ac (2009) | 158 bu/ac | 0 bu/ac (Correct) |
| **2011** | 168 bu/ac | 165 bu/ac (2010) | 165 bu/ac | 0 bu/ac (Correct) |
| **2012 (Drought)** | 115 bu/ac | 168 bu/ac (2011) | 168 bu/ac | 0 bu/ac (Correct) |
| **2013 (Missing)** | *Unreported* | 115 bu/ac (2012) | *Row dropped* | N/A |
| **2014** | 175 bu/ac | **NaN (2013 was missing)** | **115 bu/ac (2012!)** | **-60 bu/ac CORRUPTION!** |

---

## 3. AVOIDING DATA LEAKAGE IN ROLLING MEANS & DETRENDING

### 3.1 Strict Prohibition of Future Yields in Moving Averages
When computing rolling 3-year or 5-year county historical yield averages:
- **Rule:** The window must strictly use historical lags:
  $$\bar{Y}_{i, t}^{\text{hist}} = \frac{1}{3} \sum_{k=1}^3 Y_{i, t-k}$$
- **Common Violation:** Computing centered moving averages $\frac{1}{3}(Y_{i, t-1} + Y_{i, t} + Y_{i, t+1})$ leaks both current-year target $Y_{i, t}$ and future harvest $Y_{i, t+1}$ into the features.

### 3.2 Fold-Level Detrending Without Data Leakage
US corn yields exhibit an upward trend of approximately $+1.9\text{ bu/acre/year}$ due to technological advancements. If detrending is performed across the entire dataset prior to cross-validation:
- Test-set yields leak into the global slope estimation $\beta_{\text{trend}}$.
- **Mandatory Solution:** Linear or spline trend estimation must be **fit solely on the training fold**, and then applied to the validation/test fold:
  $$\hat{y}_{i, t}^{\text{detrended}} = y_{i, t} - \hat{\beta}_{\text{train}} \times (\text{year}_t - 2002)$$
  $$\hat{\beta}_{\text{train}} = \arg\min_{\beta} \sum_{i, t \in \text{Train}} (y_{i, t} - \beta \times \text{year}_t - \alpha)^2$$
