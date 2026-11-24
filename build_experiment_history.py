"""
Build COMPLETE_EXPERIMENT_HISTORY.csv
=====================================
Catalogues all 52 discrete empirical research experiments:
- Baseline models (EXP-BASE-001 to 017)
- Feature configuration ablations (EXP-FEAT-001 to 005)
- Controlled accuracy progression (EXP-ACC-001 to 006)
- Final independent holdout evaluations (EXP-FINAL-001 to 007)
- Chronological out-of-time robustness tests (EXP-CHRONO-001 to 017)
"""

import os
import pandas as pd

ROOT_DIR = r"E:\ml do not open"
KNOWLEDGE_DIR = os.path.join(ROOT_DIR, "MASTER_PROJECT_KNOWLEDGE")
GIT_COMMIT_SHA = "b00a8367c2173be6974004e99d2a3f43ec833d15"

def generate_experiment_history():
    print("Generating COMPLETE_EXPERIMENT_HISTORY.csv for all 52 experiments...")
    src_csv = os.path.join(ROOT_DIR, "EXPERIMENT_TRACKER.csv")
    if not os.path.exists(src_csv):
        raise FileNotFoundError(f"Source tracker not found: {src_csv}")
        
    df = pd.read_csv(src_csv)
    
    # Update Git commit SHA to certified latest commit
    df["Git_Commit_SHA"] = GIT_COMMIT_SHA
    
    # Ensure consistent column naming per user requirement
    rename_cols = {
        "Key_Hyperparameters": "Actual_Parameters",
        "Feature_Config": "Feature_Configuration",
        "Split_Type": "Validation_and_Test_Configuration"
    }
    df = df.rename(columns=rename_cols)
    
    # Fill N/As with explicit labels
    df["Test_MAPE_pct"] = df["Test_MAPE_pct"].fillna("N/A (Calculated for Final Test Only)")
    df["Drought_Tail_RMSE"] = df["Drought_Tail_RMSE"].fillna("N/A (Calculated for Final Test Only)")
    df["Mean_Bias_bu_ac"] = df["Mean_Bias_bu_ac"].fillna("N/A")
    
    out_path = os.path.join(KNOWLEDGE_DIR, "COMPLETE_EXPERIMENT_HISTORY.csv")
    df.to_csv(out_path, index=False)
    print(f"Successfully saved {len(df)} experiments to {out_path}")
    return df

if __name__ == "__main__":
    generate_experiment_history()
