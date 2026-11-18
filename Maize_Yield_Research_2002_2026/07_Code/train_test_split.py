"""
07_Code/train_test_split.py
Centralized Chronological 80:20 Train-Test Splitting Module
Maize/Corn Grain Yield Prediction Research Project (2002-2025)

Enforces strict, uniform chronological train-test splitting across all
machine learning experiments, baseline benchmarks, and ensemble models.
"""

import os
import json
import numpy as np
import pandas as pd

# Canonical Chronological 80:20 Definition for 2002-2025 (24 years)
TRAIN_START_YEAR = 2002
TRAIN_END_YEAR = 2020    # 19 years (79.17% ~ 80%)
TEST_START_YEAR = 2021   # 5 years (20.83% ~ 20%)
TEST_END_YEAR = 2025

CHRONOLOGICAL_SPLIT_CONFIG = {
    "split_strategy": "Chronological Block (No random row shuffling)",
    "target_ratio": "80:20",
    "total_years": 24,
    "train_year_range": [TRAIN_START_YEAR, TRAIN_END_YEAR],
    "train_years_count": 19,
    "train_years_ratio_pct": 79.17,
    "test_year_range": [TEST_START_YEAR, TEST_END_YEAR],
    "test_years_count": 5,
    "test_years_ratio_pct": 20.83,
    "prospective_year_quarantine": 2026,
    "county_assignment_policy": "Identical year boundaries applied to all counties"
}


def get_chronological_80_20_split(df, year_col="year"):
    """
    Computes and returns the canonical 80:20 chronological train and test masks.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset containing a year column.
    year_col : str
        Name of the year column (default: 'year').
        
    Returns:
    --------
    train_mask : pd.Series (bool)
        Boolean mask where True indicates training observations (2002-2020).
    test_mask : pd.Series (bool)
        Boolean mask where True indicates testing observations (2021-2025).
    split_meta : dict
        Detailed metadata describing observation counts, percentages, and verification.
    """
    if year_col not in df.columns:
        raise KeyError(f"Year column '{year_col}' not found in dataframe.")

    unique_years = sorted(df[year_col].unique())
    
    # Filter to historical range (2002-2025), preserving 2026 quarantine
    hist_years = [y for y in unique_years if y <= TEST_END_YEAR]
    
    n_years = len(hist_years)
    n_train_years = int(round(0.80 * n_years))
    
    train_years = hist_years[:n_train_years]
    test_years = hist_years[n_train_years:]
    
    train_mask = df[year_col].isin(train_years)
    test_mask = df[year_col].isin(test_years)
    
    # Integrity checks
    assert len(set(train_years).intersection(set(test_years))) == 0, "Temporal overlap detected!"
    assert train_mask.sum() + test_mask.sum() == len(df[df[year_col] <= TEST_END_YEAR]), "Record conservation violated!"
    assert df.loc[train_mask, year_col].max() < df.loc[test_mask, year_col].min(), "Temporal ordering violated!"
    
    train_count = int(train_mask.sum())
    test_count = int(test_mask.sum())
    total_count = train_count + test_count
    
    split_meta = {
        "train_years": [int(y) for y in train_years],
        "test_years": [int(y) for y in test_years],
        "train_observation_count": train_count,
        "test_observation_count": test_count,
        "total_historical_observations": total_count,
        "train_observation_pct": round(100.0 * train_count / total_count, 2),
        "test_observation_pct": round(100.0 * test_count / total_count, 2),
        "unique_counties_in_train": int(df.loc[train_mask, "county_fips"].nunique()) if "county_fips" in df.columns else None,
        "unique_counties_in_test": int(df.loc[test_mask, "county_fips"].nunique()) if "county_fips" in df.columns else None,
        "temporal_overlap": False,
        "data_leakage_risk": "Zero (Strict chronological split point at year 2020/2021 boundary)"
    }
    
    return train_mask, test_mask, split_meta


def audit_data_leakage(df, train_mask, test_mask, feature_cols, target_col="corn_yield_bu_acre", year_col="year"):
    """
    Performs rigorous data leakage checks across training and test splits.
    
    Returns:
    --------
    pd.DataFrame containing audit check items, expected criteria, observed status, and pass/fail verdict.
    """
    audit_results = []
    
    # Check 1: Temporal overlap
    train_yrs = set(df.loc[train_mask, year_col].unique())
    test_yrs = set(df.loc[test_mask, year_col].unique())
    overlap = train_yrs.intersection(test_yrs)
    audit_results.append({
        "Audit_Item": "Temporal Non-Overlap",
        "Criterion": "Train years and test years must be mutually disjoint sets",
        "Observed_Detail": f"Train: {sorted(train_yrs)}, Test: {sorted(test_yrs)}, Overlap: {list(overlap)}",
        "Status": "PASS" if len(overlap) == 0 else "FAIL"
    })
    
    # Check 2: Chronological Ordering
    max_train_yr = df.loc[train_mask, year_col].max()
    min_test_yr = df.loc[test_mask, year_col].min()
    audit_results.append({
        "Audit_Item": "Chronological Directionality",
        "Criterion": "max(Train Years) < min(Test Years)",
        "Observed_Detail": f"Max Train Year: {max_train_yr}, Min Test Year: {min_test_yr}",
        "Status": "PASS" if max_train_yr < min_test_yr else "FAIL"
    })
    
    # Check 3: Missing Target Values
    train_target_nulls = df.loc[train_mask, target_col].isnull().sum()
    test_target_nulls = df.loc[test_mask, target_col].isnull().sum()
    audit_results.append({
        "Audit_Item": "Target Completeness",
        "Criterion": "Target variable must have zero missing/null values in both splits",
        "Observed_Detail": f"Train Nulls: {train_target_nulls}, Test Nulls: {test_target_nulls}",
        "Status": "PASS" if (train_target_nulls == 0 and test_target_nulls == 0) else "FAIL"
    })
    
    # Check 4: Observation Conservation
    total_hist = (train_mask | test_mask).sum()
    audit_results.append({
        "Audit_Item": "Observation Conservation",
        "Criterion": "Total Split Observations must equal verified historical records (16,390)",
        "Observed_Detail": f"Total Split Observations: {total_hist} (Train: {train_mask.sum()}, Test: {test_mask.sum()})",
        "Status": "PASS" if total_hist == 16390 else "FAIL"
    })
    
    # Check 5: Quarantined Leakage Features
    quarantine_cols = [
        "rma_postharvest_policies_indemnified", "rma_postharvest_indemnity_amount",
        "rma_postharvest_loss_ratio", "state_progress_postharvest_harvested_w44_pct",
        "noaa_prcp_annual_in"
    ]
    present_leaks = [c for c in quarantine_cols if c in feature_cols]
    audit_results.append({
        "Audit_Item": "Post-Harvest Feature Quarantine",
        "Criterion": "No post-harvest indemnity, late-harvest progress, or full-year weather in feature set",
        "Observed_Detail": f"Quarantined features present: {present_leaks}",
        "Status": "PASS" if len(present_leaks) == 0 else "FAIL"
    })
    
    # Check 6: 2012 Severe Drought Retention
    n_2012_train = len(df.loc[train_mask & (df[year_col] == 2012)])
    audit_results.append({
        "Audit_Item": "2012 Drought Retention in Train",
        "Criterion": "All 711 certified 2012 extreme drought records must be present in training fold",
        "Observed_Detail": f"Observed 2012 records in train: {n_2012_train}",
        "Status": "PASS" if n_2012_train == 711 else "FAIL"
    })
    
    return pd.DataFrame(audit_results)


def export_split_artifacts(output_dir, df, train_mask, test_mask, split_meta):
    """
    Exports split configuration and observation assignment IDs.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Config JSON
    config_path = os.path.join(output_dir, "TRAIN_TEST_SPLIT_CONFIG.json")
    with open(config_path, "w") as f:
        json.dump({**CHRONOLOGICAL_SPLIT_CONFIG, **split_meta}, f, indent=2)
        
    # 2. Observation Assignment CSV
    obs_df = df[["county_fips", "state", "year"]].copy() if "county_fips" in df.columns else df[["year"]].copy()
    obs_df["split_assignment"] = np.where(train_mask, "TRAIN_2002_2020", np.where(test_mask, "TEST_2021_2025", "EXCLUDED_PROSPECTIVE_2026"))
    obs_path = os.path.join(output_dir, "TRAIN_TEST_OBSERVATION_IDS.csv")
    obs_df.to_csv(obs_path, index=False)
    
    print(f"Exported split configuration to {config_path}")
    print(f"Exported observation IDs to {obs_path}")

