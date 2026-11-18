# COVER LETTER DRAFT

[Date: November 2026]

To:  
Editor-in-Chief  
*Computers and Electronics in Agriculture*  
Elsevier Science

**Subject: Submission of Original Research Article**

Dear Editor-in-Chief,

We are pleased to submit our original research manuscript entitled:

**"Integrating Historical Yield Memory and Environmental Stress for County-Level Maize Yield Prediction in the U.S. Corn Belt: Temporal Validation, Geographic Transferability, Drought Robustness, and a Prospective 2026 Forecast"**

for consideration for publication in *Computers and Electronics in Agriculture*.

### Background and Scientific Significance
County-level maize (*Zea mays L.*) yield forecasting is critical for global grain logistics, agricultural trade, and crop insurance underwriting. However, empirical machine learning models in agricultural meteorology frequently suffer from three major methodological flaws: (1) conflating static historical technological trends with genuine environmental weather sensitivity; (2) cross-validation data leakage that inflates reported accuracy; and (3) severe prediction collapse during acute drought events.

### Key Contributions of This Study
In this manuscript, we present a comprehensive, reproducible investigation spanning 16,390 county-year panel records across 788 agricultural counties in the nine U.S. Corn Belt states from 2002 to 2025. Using 69 strictly in-season predictors observable on or before August 31, our study establishes:
1. **Feature Decoupling:** Controlled feature-family ablation demonstrates that environmental stress indices contribute a proven incremental gain of -3.32 bu/acre in RMSE (R² increasing from 0.612 to 0.743) over historical yield memory alone.
2. **Dual-Champion Topologies:** A regularized Ridge Stacking Meta-Learner maximizes global predictive accuracy across the prospective 2023–2025 cohort (RMSE = 14.9206 bu/acre, R² = 0.7429, Bias = +0.2418 bu/acre), while a Calibrated Quad-Tree Blend minimizes operational deployment complexity and severe drought tail error (RMSE = 29.6209 bu/acre).
3. **2012 Drought Preservation Invariant:** We prove that retaining the 2012 severe drought shock in the training partition is mandatory; removing it induces an empirical +1.9821 bu/acre penalty in the severe drought tail of subsequent test harvests.
4. **Spatial Transferability & Irrigation Diagnosis:** Spatial Leave-One-State-Out (LOSO) demonstrates that core rainfed states transfer reliably, while Western Corn Belt transfer (Nebraska) suffers a +2.48 bu/acre penalty, isolating groundwater extraction from the Ogallala Aquifer as a primary source of regional transfer breakdown.
5. **Calibrated Conformal Uncertainty:** Distribution-free split-conformal prediction intervals achieve 90.23% empirical coverage on out-of-time test harvests.

### Ethical Declarations & Open Science
This manuscript has not been published previously and is not under consideration for publication elsewhere. All authors have approved the submission. Full Python training code, feature tables, and evaluation matrices are permanently archived to ensure complete scientific reproducibility.

We believe this work will be of immediate interest to the readership of *Computers and Electronics in Agriculture*. Thank you for your time and consideration.

Sincerely,

**The Authors**  
[Corresponding Author Information Placeholder]  
[Department of Agronomy & Biosystems Engineering Placeholder]  
[University / Research Institute Placeholder]
