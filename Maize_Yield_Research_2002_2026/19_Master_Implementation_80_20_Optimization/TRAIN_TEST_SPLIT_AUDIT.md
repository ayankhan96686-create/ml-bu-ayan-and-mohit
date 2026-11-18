# PRIMARY 80:20 POOLED STRATIFIED TRAIN-TEST SPLIT AUDIT

## 1. Split Design & Population Scope

In accordance with Step 2 directives, we created one pooled 80:20 random split across the complete eligible historical dataset (2002–2025). Rather than restricting the evaluation to a late-period temporal block, this primary benchmark samples observations from across all available crop years using stratified sampling by calendar year with a fixed random seed of 42.

- **Total Eligible Historical Observations:** 16,390 records
- **Primary Training Set (80%):** 13,112 records (80.0%)
- **Primary Testing Set (20%):** 3,278 records (20.0%)
- **Random Seed:** 42 (Reproducible)
- **Stratification Strategy:** Proportional calendar year stratification across all 24 historical years (2002–2025)
- **Total Counties Represented:** 788 (Train: 785, Test: 762)
- **Prospective Year 2026:** Quarantined from supervised split (ground truth yield unobserved)

## 2. Year-Wise Stratification Breakdown

| Year | Total Observations | Training Records (80%) | Testing Records (20%) | Train % | Test % |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 2002 | 762.0 | 610.0 | 152.0 | 80.0% | 19.9% |
| 2003 | 764.0 | 611.0 | 153.0 | 80.0% | 20.0% |
| 2004 | 751.0 | 601.0 | 150.0 | 80.0% | 20.0% |
| 2005 | 756.0 | 605.0 | 151.0 | 80.0% | 20.0% |
| 2006 | 733.0 | 586.0 | 147.0 | 80.0% | 20.1% |
| 2007 | 739.0 | 591.0 | 148.0 | 80.0% | 20.0% |
| 2008 | 726.0 | 581.0 | 145.0 | 80.0% | 20.0% |
| 2009 | 721.0 | 577.0 | 144.0 | 80.0% | 20.0% |
| 2010 | 742.0 | 593.0 | 149.0 | 79.9% | 20.1% |
| 2011 | 711.0 | 569.0 | 142.0 | 80.0% | 20.0% |
| 2012 | 711.0 | 569.0 | 142.0 | 80.0% | 20.0% |
| 2013 | 683.0 | 546.0 | 137.0 | 79.9% | 20.1% |
| 2014 | 695.0 | 556.0 | 139.0 | 80.0% | 20.0% |
| 2015 | 653.0 | 522.0 | 131.0 | 79.9% | 20.1% |
| 2016 | 674.0 | 539.0 | 135.0 | 80.0% | 20.0% |
| 2017 | 676.0 | 541.0 | 135.0 | 80.0% | 20.0% |
| 2018 | 607.0 | 485.0 | 122.0 | 79.9% | 20.1% |
| 2019 | 556.0 | 445.0 | 111.0 | 80.0% | 20.0% |
| 2020 | 737.0 | 590.0 | 147.0 | 80.0% | 19.9% |
| 2021 | 617.0 | 494.0 | 123.0 | 80.1% | 19.9% |
| 2022 | 677.0 | 542.0 | 135.0 | 80.1% | 19.9% |
| 2023 | 632.0 | 505.0 | 127.0 | 79.9% | 20.1% |
| 2024 | 546.0 | 437.0 | 109.0 | 80.0% | 20.0% |
| 2025 | 521.0 | 417.0 | 104.0 | 80.0% | 20.0% |


## 3. State-Wise Geographic Representation

| State | Total Records | Training Records | Testing Records | Train % | Test % |
| :---: | :---: | :---: | :---: | :---: | :---: |
| IA | 2,272 | 1,800 | 472 | 79.2% | 20.8% |
| IL | 2,213 | 1,808 | 405 | 81.7% | 18.3% |
| IN | 1,968 | 1,598 | 370 | 81.2% | 18.8% |
| MN | 1,709 | 1,366 | 343 | 79.9% | 20.1% |
| MO | 1,779 | 1,432 | 347 | 80.5% | 19.5% |
| NE | 1,888 | 1,484 | 404 | 78.6% | 21.4% |
| OH | 1,879 | 1,475 | 404 | 78.5% | 21.5% |
| SD | 1,231 | 981 | 250 | 79.7% | 20.3% |
| WI | 1,451 | 1,168 | 283 | 80.5% | 19.5% |


## 4. Split Integrity Verification

1. **Exact Conservation:** $13,112 + 3,278 = 16,390$ records. Zero observations dropped or unaccounted for.
2. **Mutual Exclusivity:** The intersection between train index set and test index set is empty ($\emptyset$). Zero observation overlap.
3. **Year Representation:** Every single year from 2002 to 2025 contains exactly 80.0% ($\pm 0.1%$) training samples and 20.0% testing samples.
4. **Target Integrity:** Target variable (`corn_yield_bu_acre`) definition is identical across both splits with zero missing values.
5. **Quarantine Compliance:** 2026 unlabelled records ($N = 814$) are excluded from test evaluation to prevent pseudo-label contamination.
