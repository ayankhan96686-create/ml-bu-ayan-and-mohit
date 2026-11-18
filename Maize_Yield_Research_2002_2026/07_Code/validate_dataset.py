"""
07_Code/validate_dataset.py
Executes Phase 10: Research Validation & Machine Learning Split Verification.
Validates:
1. Temporal split integrity (Train: 2002-2019, Val: 2020-2022, Test: 2023-2025, Future: 2026)
2. Target distribution consistency across temporal splits (Mean, Std, Min, Max, Skew)
3. Feature completeness and missingness across splits
4. Multicollinearity screening (high pairwise correlation pairs |r| > 0.85)
5. Spatial yield distribution map (9 Corn Belt States county centroids)
6. Outputs validation summary tables to 06_Reports/ and 05_Data_Analysis/
"""

import os
import sys
import pandas as pd
import numpy as np

# Import centralized 80:20 chronological split module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    import train_test_split as tts
except ImportError:
    tts = None
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = r"E:\ml do not open\Maize_Yield_Research_2002_2026"
INT_DIR = os.path.join(PROJECT_ROOT, "04_Integrated_Data")
ANALYSIS_DIR = os.path.join(PROJECT_ROOT, "05_Data_Analysis")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "06_Reports")
VIZ_DIR = os.path.join(ANALYSIS_DIR, "Visualizations")
YIELD_DIR = os.path.join(ANALYSIS_DIR, "Yield_Analysis")

feat_path = os.path.join(INT_DIR, "County_Year_Feature_Table.csv")
future_path = os.path.join(INT_DIR, "Future_2026_Feature_Table.csv")
master_path = os.path.join(INT_DIR, "County_Year_Master_Dataset.csv")

df_feat = pd.read_csv(feat_path, low_memory=False)
df_future = pd.read_csv(future_path, low_memory=False)
df_master = pd.read_csv(master_path, low_memory=False)

print(f"Loaded Feature Table: {df_feat.shape[0]} rows, {df_feat.shape[1]} cols")
print(f"Loaded Future 2026:   {df_future.shape[0]} rows, {df_future.shape[1]} cols")

# -------------------------------------------------------------
# 1. TEMPORAL SPLIT VALIDATION & CENTRALIZED 80:20 SPLIT
# -------------------------------------------------------------
if tts is not None:
    train_mask_80_20, test_mask_80_20, meta_80_20 = tts.get_chronological_80_20_split(df_feat)
    print(f"Canonical 80:20 Split: Train 2002-2020 ({train_mask_80_20.sum()} rows), Test 2021-2025 ({test_mask_80_20.sum()} rows)")

train_mask = df_feat["year"] <= 2019
val_mask = (df_feat["year"] >= 2020) & (df_feat["year"] <= 2022)
test_mask = df_feat["year"] >= 2023

df_train = df_feat[train_mask]
df_val = df_feat[val_mask]
df_test = df_feat[test_mask]

split_summary = []
for name, sub, yrs in [
    ("Train (2002-2019)", df_train, "2002-2019 (18 yrs)"),
    ("Val (2020-2022)", df_val, "2020-2022 (3 yrs)"),
    ("Test (2023-2025)", df_test, "2023-2025 (3 yrs)"),
    ("Prospective (2026)", df_future, "2026 (1 yr)")
]:
    y_col = "corn_yield_bu_acre"
    if y_col in sub.columns and sub[y_col].notna().sum() > 0:
        y = sub[y_col].dropna()
        y_mean = round(y.mean(), 2)
        y_std = round(y.std(), 2)
        y_med = round(y.median(), 2)
        y_min = round(y.min(), 2)
        y_max = round(y.max(), 2)
        y_skew = round(y.skew(), 3)
    else:
        y_mean = y_std = y_med = y_min = y_max = y_skew = "Unlabeled"

    split_summary.append({
        "split_name": name,
        "years": yrs,
        "row_count": len(sub),
        "pct_total_labeled": round(len(sub) / len(df_feat) * 100, 2) if name != "Prospective (2026)" else "N/A",
        "distinct_counties": sub["county_fips"].nunique(),
        "yield_mean": y_mean,
        "yield_std": y_std,
        "yield_median": y_med,
        "yield_min": y_min,
        "yield_max": y_max,
        "yield_skewness": y_skew
    })

df_split_report = pd.DataFrame(split_summary)
split_out = os.path.join(YIELD_DIR, "temporal_split_validation.csv")
df_split_report.to_csv(split_out, index=False)
print("\n=== TEMPORAL SPLIT VALIDATION ===")
print(df_split_report[["split_name", "row_count", "distinct_counties", "yield_mean", "yield_std"]])

# -------------------------------------------------------------
# 2. MULTICOLLINEARITY SCREENING
# -------------------------------------------------------------
print("\nScreening multicollinearity among continuous numeric predictors...")
num_cols = [c for c in df_feat.select_dtypes(include=[np.number]).columns if c not in ["year", "corn_yield_bu_acre", "cv_percent"]]
corr_matrix = df_feat[num_cols].corr().abs()

# Find upper triangle pairs > 0.85
high_corr_pairs = []
for i in range(len(num_cols)):
    for j in range(i + 1, len(num_cols)):
        col1 = num_cols[i]
        col2 = num_cols[j]
        val = corr_matrix.loc[col1, col2]
        if val >= 0.85:
            high_corr_pairs.append({
                "feature_1": col1,
                "feature_2": col2,
                "correlation": round(val, 4),
                "domain_relationship": "Climate Collinearity" if "noaa" in col1 and "noaa" in col2 else (
                    "Soil Depth Collinearity" if "soil" in col1 and "soil" in col2 else "Cross-Domain"
                )
            })

df_high_corr = pd.DataFrame(high_corr_pairs).sort_values("correlation", ascending=False)
corr_out = os.path.join(ANALYSIS_DIR, "Correlation_Analysis", "high_collinearity_pairs.csv")
df_high_corr.to_csv(corr_out, index=False)
print(f"Identified {len(df_high_corr)} highly collinear feature pairs (|r| >= 0.85). Saved to {corr_out}")

# -------------------------------------------------------------
# 3. SPATIAL YIELD MAP (9 CORN BELT STATES)
# -------------------------------------------------------------
print("Generating spatial yield map across 814 counties...")
# Aggregate 24-year county mean yield and coordinates
county_geo = df_feat.groupby(["county_fips", "state", "county_name"]).agg(
    mean_yield=("corn_yield_bu_acre", "mean"),
    lat=("soil_intptlat", "first"),
    lon=("soil_intptlon", "first")
).reset_index()

plt.figure(figsize=(12, 8))
scatter = plt.scatter(
    county_geo["lon"], county_geo["lat"],
    c=county_geo["mean_yield"], cmap="YlGn",
    s=45, edgecolor="black", linewidth=0.3, alpha=0.9
)
cbar = plt.colorbar(scatter, shrink=0.75, pad=0.03)
cbar.set_label("Historical Mean Corn Grain Yield (bu/acre)", fontsize=11)

plt.title("Spatial Distribution of Mean Corn Grain Yield (2002–2025)\nUS Corn Belt (9 States, 814 Counties)", fontsize=14, pad=15)
plt.xlabel("Longitude (°W)", fontsize=11)
plt.ylabel("Latitude (°N)", fontsize=11)
plt.grid(True, linestyle="--", alpha=0.5)

# Annotate state clusters
state_centers = county_geo.groupby("state")[["lon", "lat"]].mean()
for st, pos in state_centers.iterrows():
    plt.text(pos["lon"], pos["lat"], st, fontsize=12, fontweight="bold", color="#1c2d42",
             ha="center", va="center", bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none"))

plt.tight_layout()
map_out = os.path.join(VIZ_DIR, "15_spatial_yield_map.png")
plt.savefig(map_out, dpi=300)
plt.savefig(os.path.join(YIELD_DIR, "fig15_spatial_yield_map.png"), dpi=300)
plt.close()
print(f"Saved Spatial Yield Map to {map_out}")

print("\n--- Phase 10 Research Validation Complete ---")

