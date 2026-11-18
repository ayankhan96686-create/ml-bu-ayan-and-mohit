"""
07_Code/clean_agriculture.py
Processes authentic agricultural management, condition, progress, insurance, and nutrient datasets.
Generates:
1. 03_Cleaned_Data/Crop_Condition_Cleaned.csv (State-Year annual crop condition indices)
2. 03_Cleaned_Data/Crop_Progress_Cleaned.csv (State-Year phenological progress & milestone timing)
3. 03_Cleaned_Data/RMA_Cleaned.csv (County-Year RMA insurance coverage with explicit post-harvest leakage quarantine)
4. 03_Cleaned_Data/Fertilizer_Cleaned.csv (State-Year nutrient application survey rates)
"""

import os
import glob
import pandas as pd
import numpy as np

PROJECT_ROOT = r"E:\ml do not open\Maize_Yield_Research_2002_2026"
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "03_Cleaned_Data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

STATE_FIPS_MAP = {
    "ILLINOIS": "17",
    "INDIANA": "18",
    "IOWA": "19",
    "MINNESOTA": "27",
    "MISSOURI": "29",
    "NEBRASKA": "31",
    "OHIO": "39",
    "SOUTH DAKOTA": "46",
    "WISCONSIN": "55"
}

STATE_ABBR_MAP = {
    "17": "IL", "18": "IN", "19": "IA", "27": "MN", "29": "MO",
    "31": "NE", "39": "OH", "46": "SD", "55": "WI"
}

# -------------------------------------------------------------
# 1. PROCESS USDA NASS CROP CONDITION
# -------------------------------------------------------------
print("Processing USDA NASS Crop Condition...")
cond_files = glob.glob(os.path.join(PROJECT_ROOT, "01_Raw_Data", "12_Planting_and_Crop_Management", "*Crop_Condition*.csv"))
if not cond_files:
    raise FileNotFoundError("Crop Condition CSV not found.")

df_cond_raw = pd.read_csv(cond_files[0], low_memory=False)
df_cond = df_cond_raw[
    (df_cond_raw["Year"] >= 2002) & 
    (df_cond_raw["Year"] <= 2026) & 
    (df_cond_raw["State"].str.upper().isin(STATE_FIPS_MAP.keys()))
].copy()

# Clean numeric values
df_cond["value_clean"] = pd.to_numeric(
    df_cond["Value"].astype(str).str.replace(",", "").str.strip(), errors="coerce"
)
df_cond["week_num"] = df_cond["Period"].str.extract(r"(\d+)").astype(float)
df_cond["state_upper"] = df_cond["State"].str.upper()
df_cond["state_fips"] = df_cond["state_upper"].map(STATE_FIPS_MAP)

# Pivot to wide format: Year, State, Week
piv_cond = df_cond.pivot_table(
    index=["Year", "state_upper", "state_fips", "week_num"],
    columns="Data Item",
    values="value_clean"
).reset_index()

item_map = {
    "CORN - CONDITION, MEASURED IN PCT EXCELLENT": "pct_excellent",
    "CORN - CONDITION, MEASURED IN PCT GOOD": "pct_good",
    "CORN - CONDITION, MEASURED IN PCT FAIR": "pct_fair",
    "CORN - CONDITION, MEASURED IN PCT POOR": "pct_poor",
    "CORN - CONDITION, MEASURED IN PCT VERY POOR": "pct_very_poor"
}
piv_cond.rename(columns=item_map, inplace=True)

# Calculate composite condition indices per week
piv_cond["pct_good_excellent"] = piv_cond["pct_good"].fillna(0) + piv_cond["pct_excellent"].fillna(0)
piv_cond["pct_poor_very_poor"] = piv_cond["pct_poor"].fillna(0) + piv_cond["pct_very_poor"].fillna(0)

# Annual aggregation per state-year
# Seasonal mean (Weeks 20-42)
cond_season = piv_cond[(piv_cond["week_num"] >= 20) & (piv_cond["week_num"] <= 42)].groupby(
    ["Year", "state_upper", "state_fips"]
).agg(
    state_cond_excellent_mean=("pct_excellent", "mean"),
    state_cond_good_mean=("pct_good", "mean"),
    state_cond_fair_mean=("pct_fair", "mean"),
    state_cond_poor_mean=("pct_poor", "mean"),
    state_cond_very_poor_mean=("pct_very_poor", "mean"),
    state_cond_good_excellent_mean=("pct_good_excellent", "mean"),
    state_cond_poor_very_poor_mean=("pct_poor_very_poor", "mean")
).reset_index()

# Critical reproductive window mean (Weeks 28-34, ~July to mid-August)
cond_rep = piv_cond[(piv_cond["week_num"] >= 28) & (piv_cond["week_num"] <= 34)].groupby(
    ["Year", "state_upper", "state_fips"]
).agg(
    state_cond_rep_good_excellent_mean=("pct_good_excellent", "mean"),
    state_cond_rep_poor_very_poor_mean=("pct_poor_very_poor", "mean")
).reset_index()

df_cond_clean = pd.merge(cond_season, cond_rep, on=["Year", "state_upper", "state_fips"], how="left")
df_cond_clean.rename(columns={"Year": "year", "state_upper": "state_name"}, inplace=True)
df_cond_clean["state"] = df_cond_clean["state_fips"].map(STATE_ABBR_MAP)

# Reorder columns
cond_cols = [
    "year", "state", "state_name", "state_fips",
    "state_cond_excellent_mean", "state_cond_good_mean", "state_cond_fair_mean",
    "state_cond_poor_mean", "state_cond_very_poor_mean",
    "state_cond_good_excellent_mean", "state_cond_poor_very_poor_mean",
    "state_cond_rep_good_excellent_mean", "state_cond_rep_poor_very_poor_mean"
]
df_cond_clean = df_cond_clean[cond_cols].sort_values(["state", "year"]).round(2)
cond_out_path = os.path.join(OUTPUT_DIR, "Crop_Condition_Cleaned.csv")
df_cond_clean.to_csv(cond_out_path, index=False)
print(f"Saved Crop Condition to {cond_out_path}: {df_cond_clean.shape[0]} rows, {df_cond_clean.shape[1]} cols")

# -------------------------------------------------------------
# 2. PROCESS USDA NASS CROP PROGRESS
# -------------------------------------------------------------
print("Processing USDA NASS Crop Progress...")
prog_files = glob.glob(os.path.join(PROJECT_ROOT, "01_Raw_Data", "12_Planting_and_Crop_Management", "*Crop_Progress*.csv"))
if not prog_files:
    raise FileNotFoundError("Crop Progress CSV not found.")

df_prog_raw = pd.read_csv(prog_files[0], low_memory=False)
df_prog = df_prog_raw[
    (df_prog_raw["Year"] >= 2002) & 
    (df_prog_raw["Year"] <= 2026) & 
    (df_prog_raw["State"].str.upper().isin(STATE_FIPS_MAP.keys()))
].copy()

df_prog["value_clean"] = pd.to_numeric(
    df_prog["Value"].astype(str).str.replace(",", "").str.strip(), errors="coerce"
)
df_prog["week_num"] = df_prog["Period"].str.extract(r"(\d+)").astype(float)
df_prog["state_upper"] = df_prog["State"].str.upper()
df_prog["state_fips"] = df_prog["state_upper"].map(STATE_FIPS_MAP)

# Target items:
prog_items = {
    "CORN - PROGRESS, MEASURED IN PCT PLANTED": "pct_planted",
    "CORN - PROGRESS, MEASURED IN PCT EMERGED": "pct_emerged",
    "CORN - PROGRESS, MEASURED IN PCT SILKING": "pct_silking",
    "CORN - PROGRESS, MEASURED IN PCT DOUGH": "pct_dough",
    "CORN - PROGRESS, MEASURED IN PCT DENTED": "pct_dented",
    "CORN - PROGRESS, MEASURED IN PCT MATURE": "pct_mature",
    "CORN, GRAIN - PROGRESS, MEASURED IN PCT HARVESTED": "pct_harvested"
}
df_prog_sub = df_prog[df_prog["Data Item"].isin(prog_items.keys())].copy()
df_prog_sub["item_short"] = df_prog_sub["Data Item"].map(prog_items)

piv_prog = df_prog_sub.pivot_table(
    index=["Year", "state_upper", "state_fips", "week_num"],
    columns="item_short",
    values="value_clean"
).reset_index()

# For each state-year, calculate milestone progress & timing
records_prog = []
for (yr, st_name, st_fips), grp in piv_prog.groupby(["Year", "state_upper", "state_fips"]):
    grp = grp.sort_values("week_num")
    
    # Milestone % by calendar week
    w20_planted = grp[grp["week_num"] == 20]["pct_planted"].values
    val_w20_planted = w20_planted[0] if len(w20_planted) > 0 and not np.isnan(w20_planted[0]) else np.nan
    if np.isnan(val_w20_planted):
        # Nearest available week <= 21
        cand = grp[(grp["week_num"] >= 19) & (grp["week_num"] <= 21)]["pct_planted"].dropna()
        val_w20_planted = cand.iloc[-1] if len(cand) > 0 else np.nan

    w22_emerged = grp[grp["week_num"] == 22]["pct_emerged"].values
    val_w22_emerged = w22_emerged[0] if len(w22_emerged) > 0 and not np.isnan(w22_emerged[0]) else np.nan
    if np.isnan(val_w22_emerged):
        cand = grp[(grp["week_num"] >= 21) & (grp["week_num"] <= 23)]["pct_emerged"].dropna()
        val_w22_emerged = cand.iloc[-1] if len(cand) > 0 else np.nan

    w30_silking = grp[grp["week_num"] == 30]["pct_silking"].values
    val_w30_silking = w30_silking[0] if len(w30_silking) > 0 and not np.isnan(w30_silking[0]) else np.nan
    if np.isnan(val_w30_silking):
        cand = grp[(grp["week_num"] >= 29) & (grp["week_num"] <= 31)]["pct_silking"].dropna()
        val_w30_silking = cand.iloc[-1] if len(cand) > 0 else np.nan

    w34_dough = grp[grp["week_num"] == 34]["pct_dough"].values
    val_w34_dough = w34_dough[0] if len(w34_dough) > 0 and not np.isnan(w34_dough[0]) else np.nan

    w37_dented = grp[grp["week_num"] == 37]["pct_dented"].values
    val_w37_dented = w37_dented[0] if len(w37_dented) > 0 and not np.isnan(w37_dented[0]) else np.nan

    w40_mature = grp[grp["week_num"] == 40]["pct_mature"].values
    val_w40_mature = w40_mature[0] if len(w40_mature) > 0 and not np.isnan(w40_mature[0]) else np.nan

    # Post-harvest progress (leakage variable for pre-harvest models)
    w44_harvested = grp[grp["week_num"] == 44]["pct_harvested"].values
    val_w44_harvested = w44_harvested[0] if len(w44_harvested) > 0 and not np.isnan(w44_harvested[0]) else np.nan

    # Week 50% milestone reached
    def get_week_50(col_name):
        sub = grp[grp[col_name] >= 50]
        if len(sub) > 0:
            return sub["week_num"].min()
        return np.nan

    rec = {
        "year": yr,
        "state_name": st_name,
        "state_fips": st_fips,
        "state": STATE_ABBR_MAP.get(st_fips),
        "state_progress_planted_w20_pct": val_w20_planted,
        "state_progress_emerged_w22_pct": val_w22_emerged,
        "state_progress_silking_w30_pct": val_w30_silking,
        "state_progress_dough_w34_pct": val_w34_dough,
        "state_progress_dented_w37_pct": val_w37_dented,
        "state_progress_mature_w40_pct": val_w40_mature,
        "state_progress_week_50pct_planted": get_week_50("pct_planted"),
        "state_progress_week_50pct_silking": get_week_50("pct_silking"),
        "state_progress_week_50pct_mature": get_week_50("pct_mature"),
        "state_progress_postharvest_harvested_w44_pct": val_w44_harvested
    }
    records_prog.append(rec)

df_prog_clean = pd.DataFrame(records_prog).sort_values(["state", "year"]).round(2)
prog_out_path = os.path.join(OUTPUT_DIR, "Crop_Progress_Cleaned.csv")
df_prog_clean.to_csv(prog_out_path, index=False)
print(f"Saved Crop Progress to {prog_out_path}: {df_prog_clean.shape[0]} rows, {df_prog_clean.shape[1]} cols")

# -------------------------------------------------------------
# 3. PROCESS USDA RMA SUMMARY OF BUSINESS (SOBCOV)
# -------------------------------------------------------------
print("Processing USDA RMA Summary of Business...")
rma_cols_28 = [
    "year", "state_fips", "state_abbr", "county_fips_3d", "county_name",
    "commodity_code", "commodity_name", "insurance_plan_code", "insurance_plan_abbr",
    "coverage_type", "delivery_type", "coverage_level", "policies_sold",
    "policies_earning_prem", "policies_indemnified", "units_earning_prem",
    "units_indemnified", "quantity_type", "net_planted_acres", "endorsed_acres",
    "liability", "total_premium", "subsidy", "state_subsidy", "additional_subsidy",
    "eobf_reserve", "indemnity_amount", "loss_ratio"
]

target_state_fips = set(STATE_FIPS_MAP.values())
rma_records = []

for yr in range(2002, 2027):
    # Locate text file in sobcov_YYYY
    sobcov_folder = os.path.join(PROJECT_ROOT, "02_Extracted_Data", "12_Planting_and_Crop_Management", f"sobcov_{yr}")
    txt_files = glob.glob(os.path.join(sobcov_folder, "*.txt"))
    if not txt_files:
        continue
    txt_file = txt_files[0]
    
    with open(txt_file, "r", encoding="latin1") as f:
        for line in f:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 28:
                continue
            st_fips = parts[1].zfill(2)
            if st_fips not in target_state_fips:
                continue
            comm_code = parts[5].lstrip("0")
            if comm_code != "41": # Corn code is 0041
                continue
            ct_3d = parts[3].zfill(3)
            if ct_3d in ("998", "999"): # Exclude unassigned/residual balance
                continue
            
            fips_5d = f"{st_fips}{ct_3d}"
            
            # Numeric fields
            try:
                pol_sold = float(parts[12]) if parts[12] else 0.0
                pol_earn = float(parts[13]) if parts[13] else 0.0
                pol_indem = float(parts[14]) if parts[14] else 0.0
                planted_acres = float(parts[18]) if parts[18] else 0.0
                liability = float(parts[20]) if parts[20] else 0.0
                tot_prem = float(parts[21]) if parts[21] else 0.0
                subsidy = float(parts[22]) if parts[22] else 0.0
                indemnity = float(parts[26]) if parts[26] else 0.0
            except ValueError:
                continue
            
            rma_records.append({
                "county_fips": fips_5d,
                "state_fips": st_fips,
                "state": parts[2],
                "year": yr,
                "pol_sold": pol_sold,
                "pol_earn": pol_earn,
                "pol_indem": pol_indem,
                "planted_acres": planted_acres,
                "liability": liability,
                "tot_prem": tot_prem,
                "subsidy": subsidy,
                "indemnity": indemnity
            })

df_rma_all = pd.DataFrame(rma_records)
print(f"Loaded {len(df_rma_all)} policy rows for Corn across 2002-2026.")

# County-year aggregation
df_rma_agg = df_rma_all.groupby(["county_fips", "state_fips", "state", "year"]).agg(
    rma_policies_sold=("pol_sold", "sum"),
    rma_policies_earning_prem=("pol_earn", "sum"),
    rma_net_planted_acres=("planted_acres", "sum"),
    rma_liability_total=("liability", "sum"),
    rma_total_premium=("tot_prem", "sum"),
    rma_subsidy=("subsidy", "sum"),
    # Quarantined post-harvest leakage variables:
    rma_postharvest_policies_indemnified=("pol_indem", "sum"),
    rma_postharvest_indemnity_amount=("indemnity", "sum")
).reset_index()

# Pre-harvest derived financial metrics
df_rma_agg["rma_liability_per_planted_acre"] = np.where(
    df_rma_agg["rma_net_planted_acres"] > 0,
    df_rma_agg["rma_liability_total"] / df_rma_agg["rma_net_planted_acres"],
    np.nan
)
df_rma_agg["rma_premium_per_planted_acre"] = np.where(
    df_rma_agg["rma_net_planted_acres"] > 0,
    df_rma_agg["rma_total_premium"] / df_rma_agg["rma_net_planted_acres"],
    np.nan
)
# Post-harvest loss ratio (indemnity / premium)
df_rma_agg["rma_postharvest_loss_ratio"] = np.where(
    df_rma_agg["rma_total_premium"] > 0,
    df_rma_agg["rma_postharvest_indemnity_amount"] / df_rma_agg["rma_total_premium"],
    np.nan
)

rma_out_path = os.path.join(OUTPUT_DIR, "RMA_Cleaned.csv")
df_rma_agg.sort_values(["county_fips", "year"]).round(2).to_csv(rma_out_path, index=False)
print(f"Saved RMA Cleaned to {rma_out_path}: {df_rma_agg.shape[0]} rows, {df_rma_agg.shape[1]} cols")

# -------------------------------------------------------------
# 4. PROCESS USDA NASS CHEMICAL USE / FERTILIZER
# -------------------------------------------------------------
print("Processing USDA NASS Chemical Use / Fertilizer...")
fert_files = glob.glob(os.path.join(PROJECT_ROOT, "01_Raw_Data", "13_Fertilizer_and_Nutrients", "*Fertilizer_Chemical*.csv"))
if not fert_files:
    raise FileNotFoundError("Fertilizer Chemical CSV not found.")

df_fert_raw = pd.read_csv(fert_files[0], low_memory=False)
df_fert = df_fert_raw[
    (df_fert_raw["Domain"] == "FERTILIZER") &
    (df_fert_raw["State"].str.upper().isin(STATE_FIPS_MAP.keys()))
].copy()

df_fert["value_clean"] = pd.to_numeric(
    df_fert["Value"].astype(str).str.replace(",", "").str.strip(), errors="coerce"
)
df_fert["state_upper"] = df_fert["State"].str.upper()
df_fert["state_fips"] = df_fert["state_upper"].map(STATE_FIPS_MAP)
df_fert["state"] = df_fert["state_fips"].map(STATE_ABBR_MAP)

# Map nutrient category
def categorize_nutrient(cat_str):
    cat_str = str(cat_str).upper()
    if "NITROGEN" in cat_str:
        return "nitrogen"
    elif "PHOSPHATE" in cat_str:
        return "phosphate"
    elif "POTASH" in cat_str:
        return "potash"
    elif "SULFUR" in cat_str:
        return "sulfur"
    return "other"

df_fert["nutrient"] = df_fert["Domain Category"].apply(categorize_nutrient)

# We care about application rate (LB / ACRE / YEAR) and % treated
rate_item = "CORN - APPLICATIONS, MEASURED IN LB / ACRE / YEAR, AVG"
pct_item = "CORN - TREATED, MEASURED IN PCT OF AREA PLANTED, AVG"

df_rates = df_fert[df_fert["Data Item"] == rate_item].pivot_table(
    index=["Year", "state", "state_fips"],
    columns="nutrient",
    values="value_clean"
).reset_index()
df_rates.rename(columns={
    "nitrogen": "fert_nitrogen_rate_lb_acre",
    "phosphate": "fert_phosphate_rate_lb_acre",
    "potash": "fert_potash_rate_lb_acre",
    "sulfur": "fert_sulfur_rate_lb_acre"
}, inplace=True)

df_pct = df_fert[df_fert["Data Item"] == pct_item].pivot_table(
    index=["Year", "state", "state_fips"],
    columns="nutrient",
    values="value_clean"
).reset_index()
df_pct.rename(columns={
    "nitrogen": "fert_nitrogen_pct_treated",
    "phosphate": "fert_phosphate_pct_treated",
    "potash": "fert_potash_pct_treated",
    "sulfur": "fert_sulfur_pct_treated"
}, inplace=True)

df_fert_clean = pd.merge(df_rates, df_pct, on=["Year", "state", "state_fips"], how="outer")
df_fert_clean.rename(columns={"Year": "year"}, inplace=True)
fert_out_path = os.path.join(OUTPUT_DIR, "Fertilizer_Cleaned.csv")
df_fert_clean.sort_values(["state", "year"]).round(2).to_csv(fert_out_path, index=False)
print(f"Saved Fertilizer Cleaned to {fert_out_path}: {df_fert_clean.shape[0]} rows, {df_fert_clean.shape[1]} cols")

print("\n--- Phase 5 Agricultural Cleaning Complete ---")

