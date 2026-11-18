# CHRONOLOGICAL EXPANDING-WINDOW VALIDATION AUDIT

## 1. Validation Design Architecture

To ensure that hyperparameter optimization, model selection, and feature selection reflect genuine agricultural forecasting conditions, validation folds are constructed using an **expanding-window chronological design** strictly inside the primary 80% training partition.

### Strict Validation Invariants Enforced:
1. **Partition Isolation:** Validation data is drawn exclusively from the 80% training set ($N = 13,112$). The final 20% test partition ($N = 3,278$) is strictly quarantined and untouched during all tuning procedures.
2. **Grouped Calendar Years:** All observations from each validation year remain grouped together. No random splitting of observations within the same crop season.
3. **Strict Chronological Directionality:** In every fold, training observations strictly precede validation observations ($\max(\text{Train}) < \min(\text{Val})$). No future information is exposed to the model.
4. **Representative Expanding Windows:** As the historical period grows, each successive fold trains on a deeper historical archive, simulating operational annual deployment.

## 2. Chronological Validation Folds Summary

| Fold | Training Period | Validation Period | Training Observations | Validation Observations | Val Mean Yield (bu/ac) | Val Yield Std | Val Severe Tail (<120 bu) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 2002–2013 (12 yrs) | 2014–2016 (3 yrs) | 7,039 | 1,617 | 170.19 | 27.53 | 85 |
| 2 | 2002–2016 (15 yrs) | 2017–2019 (3 yrs) | 8,656 | 1,471 | 175.68 | 28.50 | 53 |
| 3 | 2002–2019 (18 yrs) | 2020–2022 (3 yrs) | 10,127 | 1,626 | 174.80 | 31.51 | 105 |
| 4 | 2002–2022 (21 yrs) | 2023–2025 (3 yrs) | 11,753 | 1,359 | 183.41 | 29.96 | 52 |


## 3. Validation Leakage Prevention Verification

- **Zero Target Contamination:** Validation targets are never included in the respective fold's training feature matrices.
- **Dynamic Pipeline Fitting:** Imputation and feature scaling are fit exclusively on the fold's training data and applied to the fold's validation observations.
- **Fold-to-Fold Metric Aggregation:** Model tuning minimizes the average chronological validation RMSE across all 4 folds, ensuring robustness across diverse weather regimes.
