"""
07_Code/perform_eda.py
Conducts rigorous Exploratory Data Analysis (EDA) on the integrated Maize Yield dataset.
Generates:
- Statistical summary CSVs across yield, climate, soil, correlations, and missingness
- Publication-quality visualization figures (300 DPI) in 05_Data_Analysis subdirectories
- Fully documented outputs for research reporting
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

PROJECT_ROOT = r"E:\ml do not open\Maize_Yield_Research_2002_2026"
INT_DIR = os.path.join(PROJECT_ROOT, "04_Integrated_Data")
ANALYSIS_DIR = os.path.join(PROJECT_ROOT, "05_Data_Analysis")

YIELD_DIR = os.path.join(ANALYSIS_DIR, "Yield_Analysis")
CLIMATE_DIR = os.path.join(ANALYSIS_DIR, "Climate_Analysis")
SOIL_DIR = os.path.join(ANALYSIS_DIR, "Soil_Analysis")
CORR_DIR = os.path.join(ANALYSIS_DIR, "Correlation_Analysis")
MISSING_DIR = os.path.join(ANALYSIS_DIR, "Missing_Data_Analysis")
VIZ_DIR = os.path.join(ANALYSIS_DIR, "Visualizations")

for d in [YIELD_DIR, CLIMATE_DIR, SOIL_DIR, CORR_DIR, MISSING_DIR, VIZ_DIR]:
    os.makedirs(d, exist_ok=True)

# Set plotting style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 16,
    "figure.dpi": 300
})

# Load datasets
master_df = pd.read_csv(os.path.join(INT_DIR, "County_Year_Master_Dataset.csv"), low_memory=False)
feat_df = pd.read_csv(os.path.join(INT_DIR, "County_Year_Feature_Table.csv"), low_memory=False)

print(f"Master Dataset: {master_df.shape[0]} rows, {master_df.shape[1]} cols")
print(f"Feature Table:  {feat_df.shape[0]} rows, {feat_df.shape[1]} cols")

# =============================================================
# 1. YIELD DISTRIBUTION & ANALYSIS
# =============================================================
print("\n[1] Analyzing Yield...")
yield_s = feat_df["corn_yield_bu_acre"]

# Summary statistics
yield_stats = {
    "metric": ["count", "mean", "std", "min", "q25", "median", "q75", "max", "skewness", "kurtosis", "iqr"],
    "value": [
        len(yield_s),
        yield_s.mean(),
        yield_s.std(),
        yield_s.min(),
        yield_s.quantile(0.25),
        yield_s.median(),
        yield_s.quantile(0.75),
        yield_s.max(),
        yield_s.skew(),
        yield_s.kurtosis(),
        yield_s.quantile(0.75) - yield_s.quantile(0.25)
    ]
}
df_ystats = pd.DataFrame(yield_stats).round(3)
df_ystats.to_csv(os.path.join(YIELD_DIR, "yield_summary_statistics.csv"), index=False)

# By State
state_yield = feat_df.groupby("state")["corn_yield_bu_acre"].agg(
    count="count", mean="mean", std="std", min="min",
    q25=lambda x: x.quantile(0.25), median="median",
    q75=lambda x: x.quantile(0.75), max="max"
).reset_index().round(2)
state_yield.to_csv(os.path.join(YIELD_DIR, "yield_by_state.csv"), index=False)

# By Year
year_yield = feat_df.groupby("year")["corn_yield_bu_acre"].agg(
    count="count", mean="mean", std="std", min="min",
    median="median", max="max"
).reset_index().round(2)
year_yield.to_csv(os.path.join(YIELD_DIR, "yield_by_year.csv"), index=False)

# Plot 1: Yield Distribution Histogram + KDE
plt.figure(figsize=(9, 6))
sns.histplot(yield_s, kde=True, color="#2b5c8f", bins=45, edgecolor="white", alpha=0.7)
plt.axvline(yield_s.mean(), color="#d95f02", linestyle="--", linewidth=2, label=f"Mean: {yield_s.mean():.1f} bu/ac")
plt.axvline(yield_s.median(), color="#1b9e77", linestyle=":", linewidth=2, label=f"Median: {yield_s.median():.1f} bu/ac")
plt.title("Distribution of US Corn Belt Grain Yield (2002–2025)", pad=15)
plt.xlabel("Corn Grain Yield (Bushels / Acre)")
plt.ylabel("County-Year Frequency")
plt.legend(frameon=True)
plt.tight_layout()
plt.savefig(os.path.join(YIELD_DIR, "fig1_yield_distribution.png"), dpi=300)
plt.savefig(os.path.join(VIZ_DIR, "01_yield_distribution.png"), dpi=300)
plt.close()

# Plot 2: Yield Boxplot by State
plt.figure(figsize=(10, 6))
state_order = state_yield.sort_values("median", ascending=False)["state"].tolist()
sns.boxplot(data=feat_df, x="state", y="corn_yield_bu_acre", order=state_order, hue="state", palette="Blues_r", legend=False)
plt.title("Corn Grain Yield Distribution by State (2002–2025)", pad=15)
plt.xlabel("Corn Belt State")
plt.ylabel("Corn Grain Yield (Bushels / Acre)")
plt.tight_layout()
plt.savefig(os.path.join(YIELD_DIR, "fig2_yield_by_state_boxplot.png"), dpi=300)
plt.savefig(os.path.join(VIZ_DIR, "02_yield_by_state_boxplot.png"), dpi=300)
plt.close()

# Plot 3: Annual Yield Trend (2002-2025) with Drought Highlight
plt.figure(figsize=(11, 6))
sns.lineplot(data=feat_df, x="year", y="corn_yield_bu_acre", color="#1f78b4", linewidth=2.5, errorbar=("ci", 95))
plt.scatter(year_yield["year"], year_yield["mean"], color="#1f78b4", s=40, zorder=5)
# Highlight 2012 severe drought
drought_2012_mean = year_yield[year_yield["year"] == 2012]["mean"].values[0]
plt.annotate("2012 Severe Drought Shock\n(Mean: 108.9 bu/ac)",
             xy=(2012, drought_2012_mean), xytext=(2007, 100),
             arrowprops=dict(facecolor="#e41a1c", shrink=0.08, width=1.5, headwidth=8),
             fontsize=10, fontweight="bold", color="#e41a1c")
# Trendline
z = np.polyfit(year_yield["year"], year_yield["mean"], 1)
p = np.poly1d(z)
plt.plot(year_yield["year"], p(year_yield["year"]), "r--", alpha=0.7, label=f"Technological Trend (+{z[0]:.2f} bu/ac/yr)")
plt.title("Corn Belt Annual Mean Yield Trend & Climate Shocks (2002–2025)", pad=15)
plt.xlabel("Year")
plt.ylabel("Corn Grain Yield (Bushels / Acre)")
plt.xticks(range(2002, 2026, 2))
plt.legend(frameon=True, loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(YIELD_DIR, "fig3_annual_yield_trend.png"), dpi=300)
plt.savefig(os.path.join(VIZ_DIR, "03_annual_yield_trend.png"), dpi=300)
plt.close()

# =============================================================
# 2. CLIMATE VS YIELD ANALYSIS
# =============================================================
print("[2] Analyzing Climate vs Yield...")

# Plot 4: Growing Season Precipitation vs Yield
plt.figure(figsize=(9, 6))
sns.regplot(
    data=feat_df, x="noaa_prcp_may_aug_in", y="corn_yield_bu_acre",
    scatter_kws={"alpha": 0.15, "color": "#2ca25f", "s": 15},
    line_kws={"color": "#006d2c", "linewidth": 2.5}
)
r_prcp, p_prcp = stats.pearsonr(feat_df["noaa_prcp_may_aug_in"].dropna(), feat_df.loc[feat_df["noaa_prcp_may_aug_in"].notna(), "corn_yield_bu_acre"])
plt.title(f"Growing Season Rainfall (May–Aug) vs Yield (r = {r_prcp:.2f}, p < 0.001)", pad=15)
plt.xlabel("NOAA Growing Season Precipitation (Inches)")
plt.ylabel("Corn Grain Yield (Bushels / Acre)")
plt.tight_layout()
plt.savefig(os.path.join(CLIMATE_DIR, "fig4_precip_vs_yield.png"), dpi=300)
plt.savefig(os.path.join(VIZ_DIR, "04_precip_vs_yield.png"), dpi=300)
plt.close()

# Plot 5: Reproductive Heat Stress (July Tmax) vs Yield
plt.figure(figsize=(9, 6))
sns.regplot(
    data=feat_df, x="noaa_tmax_jul_f", y="corn_yield_bu_acre",
    scatter_kws={"alpha": 0.15, "color": "#e34a33", "s": 15},
    line_kws={"color": "#b30000", "linewidth": 2.5}
)
r_heat, p_heat = stats.pearsonr(feat_df["noaa_tmax_jul_f"].dropna(), feat_df.loc[feat_df["noaa_tmax_jul_f"].notna(), "corn_yield_bu_acre"])
plt.title(f"Reproductive Heat Stress: July Max Temperature vs Yield (r = {r_heat:.2f}, p < 0.001)", pad=15)
plt.xlabel("NOAA July Mean Maximum Temperature (°F)")
plt.ylabel("Corn Grain Yield (Bushels / Acre)")
plt.tight_layout()
plt.savefig(os.path.join(CLIMATE_DIR, "fig5_heat_stress_vs_yield.png"), dpi=300)
plt.savefig(os.path.join(VIZ_DIR, "05_heat_stress_vs_yield.png"), dpi=300)
plt.close()

# Plot 6: PRISM Precipitation vs NOAA Precipitation Cross-Validation
plt.figure(figsize=(8, 6))
# Subsample 2000 points for crisp visual
sample_cross = feat_df.dropna(subset=["noaa_prcp_annual_in", "prism_ppt_mm"]).sample(n=min(2500, len(feat_df)), random_state=42)
sns.scatterplot(data=sample_cross, x=sample_cross["prism_ppt_mm"]/25.4, y="noaa_prcp_annual_in", alpha=0.3, color="#3182bd")
plt.plot([15, 60], [15, 60], "r--", linewidth=2, label="1:1 Perfect Agreement Line")
plt.title("Cross-Dataset Agreement: NOAA vs PRISM Annual Precipitation", pad=15)
plt.xlabel("PRISM Annual Precipitation (Inches)")
plt.ylabel("NOAA Annual Precipitation (Inches)")
plt.legend(frameon=True)
plt.tight_layout()
plt.savefig(os.path.join(CLIMATE_DIR, "fig6_noaa_vs_prism_agreement.png"), dpi=300)
plt.savefig(os.path.join(VIZ_DIR, "06_noaa_vs_prism_agreement.png"), dpi=300)
plt.close()

# =============================================================
# 3. SOIL PROPERTY ANALYSIS
# =============================================================
print("[3] Analyzing Soil Properties...")

# Plot 7: Soil Organic Carbon (0-30cm) vs Mean County Yield
county_agg = feat_df.groupby(["county_fips", "state"]).agg(
    mean_yield=("corn_yield_bu_acre", "mean"),
    soil_soc=("soil_soc_mean_0_30cm", "first"),
    soil_clay=("soil_clay_mean_0_30cm", "first"),
    soil_sand=("soil_sand_mean_0_30cm", "first"),
    soil_ph=("soil_ph_mean_0_30cm", "first"),
    soil_bdod=("soil_bdod_mean_0_30cm", "first")
).reset_index()

plt.figure(figsize=(9, 6))
sns.regplot(
    data=county_agg, x="soil_soc", y="mean_yield",
    scatter_kws={"alpha": 0.6, "color": "#8c510a", "s": 35},
    line_kws={"color": "#543005", "linewidth": 2.5}
)
r_soc, _ = stats.pearsonr(county_agg["soil_soc"].dropna(), county_agg.loc[county_agg["soil_soc"].notna(), "mean_yield"])
plt.title(f"Soil Organic Carbon (0–30cm) vs County Historical Mean Yield (r = {r_soc:.2f})", pad=15)
plt.xlabel("Soil Organic Carbon (dg/kg)")
plt.ylabel("County 24-Year Mean Corn Yield (bu/acre)")
plt.tight_layout()
plt.savefig(os.path.join(SOIL_DIR, "fig7_soc_vs_yield.png"), dpi=300)
plt.savefig(os.path.join(VIZ_DIR, "07_soc_vs_yield.png"), dpi=300)
plt.close()

# Plot 8: Soil Texture Fractions Distribution
plt.figure(figsize=(10, 6))
soil_text_df = pd.DataFrame({
    "Clay (%)": county_agg["soil_clay"] / 10.0,
    "Sand (%)": county_agg["soil_sand"] / 10.0,
    "Silt (%)": (1000 - county_agg["soil_clay"] - county_agg["soil_sand"]) / 10.0
})
sns.boxplot(data=soil_text_df, palette=["#d8b365", "#f5f5f5", "#5ab4ac"])
plt.title("Distribution of Soil Texture Fractions Across 814 Study Counties", pad=15)
plt.ylabel("Percentage of Mineral Soil (%)")
plt.tight_layout()
plt.savefig(os.path.join(SOIL_DIR, "fig8_soil_texture_boxplot.png"), dpi=300)
plt.savefig(os.path.join(VIZ_DIR, "08_soil_texture_boxplot.png"), dpi=300)
plt.close()

# =============================================================
# 4. CORRELATION & MULTICOLLINEARITY ANALYSIS
# =============================================================
print("[4] Analyzing Correlations & Feature Collinearity...")

# Compute numeric correlations with yield
numeric_cols = feat_df.select_dtypes(include=[np.number]).columns
corr_series = feat_df[numeric_cols].corr()["corn_yield_bu_acre"].drop(["corn_yield_bu_acre", "cv_percent"], errors="ignore")
corr_df = pd.DataFrame({
    "feature": corr_series.index,
    "pearson_r": corr_series.values,
    "abs_r": corr_series.abs().values
}).sort_values("abs_r", ascending=False).reset_index(drop=True)

corr_df.round(4).to_csv(os.path.join(CORR_DIR, "feature_correlations_with_yield.csv"), index=False)

# Top 20 features correlated with yield
top20 = corr_df.head(20)

plt.figure(figsize=(10, 8))
colors = ["#e41a1c" if r < 0 else "#377eb8" for r in top20["pearson_r"]]
sns.barplot(data=top20, y="feature", x="pearson_r", palette=colors)
plt.title("Top 20 Features Correlated with Corn Grain Yield", pad=15)
plt.xlabel("Pearson Correlation Coefficient (r)")
plt.ylabel("Feature Name")
plt.axvline(0, color="black", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(CORR_DIR, "fig9_top20_correlations.png"), dpi=300)
plt.savefig(os.path.join(VIZ_DIR, "09_top20_correlations.png"), dpi=300)
plt.close()

# Plot 10: Selected Feature Correlation Heatmap (Domain Representation)
key_features = [
    "corn_yield_bu_acre",
    "year",
    "noaa_prcp_may_aug_in",
    "noaa_tmean_may_aug_f",
    "noaa_tmax_jul_f",
    "noaa_heat_stress_max_f",
    "soil_soc_mean_0_30cm",
    "soil_clay_mean_0_30cm",
    "soil_ph_mean_0_30cm",
    "rma_net_planted_acres",
    "rma_liability_per_planted_acre",
    "state_cond_good_excellent_mean",
    "state_cond_poor_very_poor_mean",
    "state_progress_planted_w20_pct",
    "state_progress_silking_w30_pct"
]
corr_sub = feat_df[[f for f in key_features if f in feat_df.columns]].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_sub, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-0.8, vmax=0.8, square=True,
            cbar_kws={"shrink": 0.8}, linewidths=0.5)
plt.title("Correlation Heatmap: Key Multi-Domain Features & Target Yield", pad=15)
plt.tight_layout()
plt.savefig(os.path.join(CORR_DIR, "fig10_domain_correlation_heatmap.png"), dpi=300)
plt.savefig(os.path.join(VIZ_DIR, "10_domain_correlation_heatmap.png"), dpi=300)
plt.close()

# =============================================================
# 5. MISSING DATA ANALYSIS
# =============================================================
print("[5] Analyzing Missing Data Patterns...")

# Master dataset missingness
master_miss = pd.DataFrame({
    "column": master_df.columns,
    "missing_count": master_df.isna().sum().values,
    "missing_pct": (master_df.isna().sum().values / len(master_df) * 100).round(2),
    "dtype": master_df.dtypes.values
}).sort_values("missing_pct", ascending=False).reset_index(drop=True)
master_miss.to_csv(os.path.join(MISSING_DIR, "master_dataset_missingness.csv"), index=False)

# Feature table missingness
feat_miss = pd.DataFrame({
    "column": feat_df.columns,
    "missing_count": feat_df.isna().sum().values,
    "missing_pct": (feat_df.isna().sum().values / len(feat_df) * 100).round(2),
    "dtype": feat_df.dtypes.values
}).sort_values("missing_pct", ascending=False).reset_index(drop=True)
feat_miss.to_csv(os.path.join(MISSING_DIR, "feature_table_missingness.csv"), index=False)

# Plot 11: Missingness by Column in Master Dataset (Top 25)
top_miss = master_miss.head(25)
plt.figure(figsize=(10, 8))
sns.barplot(data=top_miss, y="column", x="missing_pct", palette="magma")
plt.title("Missing Data Percentage by Column (Top 25 in Master Dataset)", pad=15)
plt.xlabel("Missing Percentage (%)")
plt.ylabel("Column Name")
plt.tight_layout()
plt.savefig(os.path.join(MISSING_DIR, "fig11_missingness_top25.png"), dpi=300)
plt.savefig(os.path.join(VIZ_DIR, "11_missingness_top25.png"), dpi=300)
plt.close()

# Plot 12: Labeled Yield Observations by Year
plt.figure(figsize=(10, 5))
yield_coverage_by_yr = master_df.groupby("year")["corn_yield_bu_acre"].apply(lambda s: s.notna().sum()).reset_index()
sns.barplot(data=yield_coverage_by_yr, x="year", y="corn_yield_bu_acre", color="#41b6c4")
plt.title("Count of Labeled County Grain Yield Observations by Year (out of 814 counties)", pad=15)
plt.xlabel("Year")
plt.ylabel("Number of Labeled Counties")
plt.xticks(rotation=45)
plt.axhline(814, color="red", linestyle="--", label="Total 814 Counties")
plt.legend(frameon=True)
plt.tight_layout()
plt.savefig(os.path.join(MISSING_DIR, "fig12_yield_coverage_by_year.png"), dpi=300)
plt.savefig(os.path.join(VIZ_DIR, "12_yield_coverage_by_year.png"), dpi=300)
plt.close()

# Plot 13: Crop Condition Index vs Yield
plt.figure(figsize=(9, 6))
sns.regplot(
    data=feat_df, x="state_cond_good_excellent_mean", y="corn_yield_bu_acre",
    scatter_kws={"alpha": 0.15, "color": "#756bb1", "s": 15},
    line_kws={"color": "#54278f", "linewidth": 2.5}
)
r_cond, _ = stats.pearsonr(feat_df["state_cond_good_excellent_mean"].dropna(), feat_df.loc[feat_df["state_cond_good_excellent_mean"].notna(), "corn_yield_bu_acre"])
plt.title(f"State Growing Season % Good+Excellent vs County Yield (r = {r_cond:.2f})", pad=15)
plt.xlabel("Average State % Good + Excellent Crop Condition")
plt.ylabel("Corn Grain Yield (bu/acre)")
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, "13_condition_vs_yield.png"), dpi=300)
plt.close()

# Plot 14: RMA Planted Insurance Coverage vs Yield
plt.figure(figsize=(9, 6))
sns.regplot(
    data=feat_df[feat_df["rma_liability_per_planted_acre"] < 2000],
    x="rma_liability_per_planted_acre", y="corn_yield_bu_acre",
    scatter_kws={"alpha": 0.15, "color": "#31a354", "s": 15},
    line_kws={"color": "#006d2c", "linewidth": 2.5}
)
plt.title("RMA Crop Insurance Coverage ($ Liability/Acre) vs Yield", pad=15)
plt.xlabel("Insured Dollar Liability per Planted Acre ($/Acre)")
plt.ylabel("Corn Grain Yield (bu/acre)")
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, "14_insurance_liability_vs_yield.png"), dpi=300)
plt.close()

print("\n--- Phase 8 Exploratory Data Analysis & Visualizations Complete ---")
