# REPRODUCIBILITY AND EXECUTION GUIDE

## Comprehensive Instructions for Independent Replication of Experimental Results
**Project:** County-Level Maize Grain Yield Prediction (2002–2026)  
**Output Directory:** `16_Final_Publication_Package/08_Reproducibility/`  
**Date:** November 2026

---

### 1. Environment Setup

To recreate the exact execution environment, install conda and initialize the environment:
```bash
conda env create -f ENVIRONMENT_SPECIFICATION.yml
conda activate maize_yield_env
```

### 2. Execution Pipeline

All experimental scripts are archived in `15_Final_Confirmatory_Validation/08_Code/` and `16_Final_Publication_Package/`:

1. **Step 1: Evidence and Metric Audit**
   ```bash
   python 16_Final_Publication_Package/01_Evidence_Audit/build_task_a_b1_evidence_audit.py
   ```
2. **Step 2: Numerical and Forecast Reconciliation**
   ```bash
   python 16_Final_Publication_Package/02_Numerical_Reconciliation/build_task_b_c_d_numerical_reconciliation.py
   ```
3. **Step 3: Loss Function Benchmarking**
   ```bash
   python 15_Final_Confirmatory_Validation/08_Code/run_task_d_loss_function_validation.py
   ```
4. **Step 4: Drought Robustness & 2012 Retention Audit**
   ```bash
   python 15_Final_Confirmatory_Validation/08_Code/run_task_e_drought_robustness.py
   ```
5. **Step 5: Model Benchmarking & Stacking Fitting**
   ```bash
   python 15_Final_Confirmatory_Validation/08_Code/run_task_g_final_models.py
   ```
6. **Step 6: Rolling-Origin & Spatial LOSO Validation**
   ```bash
   python 15_Final_Confirmatory_Validation/08_Code/run_task_h_i_confirmatory_validation.py
   ```
7. **Step 7: Publication Tables & Figures Compilation**
   ```bash
   python 16_Final_Publication_Package/06_Tables/build_task_h_publication_tables.py
   python 16_Final_Publication_Package/05_Figures/build_task_g_publication_figures.py
   python 16_Final_Publication_Package/04_Manuscript/build_task_f_complete_manuscript.py
   ```

### 3. Deterministic Seed Control
All algorithms utilize pinned random seeds (`random_state=42`, `random_seed=42`) across data splitting, bootstrapping, and tree fitting.
