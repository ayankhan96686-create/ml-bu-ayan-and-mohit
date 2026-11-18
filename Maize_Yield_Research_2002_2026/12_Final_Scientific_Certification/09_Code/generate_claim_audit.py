import os
import pandas as pd

out_dir = r"E:\ml do not open\Maize_Yield_Research_2002_2026\12_Final_Scientific_Certification"

claims = [
    {
        'claim_id': 'CLAIM-01',
        'topic': 'Controlled Ablation',
        'phase5_statement': 'Environmental features provide an 18.06% RMSE reduction over historical lags alone (dropping test RMSE from 21.60 to 17.70 bu/acre).',
        'audit_classification': 'VERIFIED',
        'recalculated_value': '18.0631% (21.5998 -> 17.6982 bu/ac)',
        'evidence_and_scientific_nuance': 'Exact match verified on identical 1,699 test observations. Delta RMSE = 3.90 bu/acre, Delta R2 = +0.1771.'
    },
    {
        'claim_id': 'CLAIM-02',
        'topic': 'Statistical Significance',
        'phase5_statement': 'Paired t-test (p = 9.85e-24) and Wilcoxon test (p = 2.92e-31) prove environmental superiority.',
        'audit_classification': 'VERIFIED_WITH_LIMITATIONS',
        'recalculated_value': 'Two-way Block Bootstrap 95% CI: [2.44, 5.37] bu/ac',
        'evidence_and_scientific_nuance': 'Effect size is strictly positive and statistically significant under cluster-robust tests. However, standard i.i.d. p-values ignore spatial cross-correlation and temporal panel autocorrelation, drastically overstating statistical confidence.'
    },
    {
        'claim_id': 'CLAIM-03',
        'topic': 'Baseline Reproduction',
        'phase5_statement': 'All 7 pre-modeling baselines were reproduced with 0.00 discrepancy on validation and test sets.',
        'audit_classification': 'VERIFIED',
        'recalculated_value': '0.0000 absolute discrepancy across all 7 models',
        'evidence_and_scientific_nuance': '100% exact reproduction confirmed across RMSE, MAE, R2, and relative improvements.'
    },
    {
        'claim_id': 'CLAIM-04',
        'topic': 'Model Leaderboard',
        'phase5_statement': 'HistGradientBoosting (17.24 bu/ac), MLP (17.26 bu/ac), and LightGBM (17.33 bu/ac) outperform XGBoost (17.70 bu/ac).',
        'audit_classification': 'VERIFIED_WITH_LIMITATIONS',
        'recalculated_value': '95% CI of diff vs HGB: MLP [-0.96, 1.05], LGBM [-0.26, 0.43]',
        'evidence_and_scientific_nuance': 'Metrics are accurate, but top 3 models are statistically indistinguishable from each other within sampling noise. Furthermore, choosing a single winner based on inspecting the test set holdout compromises strict test set purity.'
    },
    {
        'claim_id': 'CLAIM-05',
        'topic': 'Spatial LOSO R²',
        'phase5_statement': 'Average spatial holdout R² across the 9 states is 0.6879 in Scenario 1 (Lags + Env).',
        'audit_classification': 'REQUIRES_CORRECTION',
        'recalculated_value': 'Arithmetic Macro-Average R² = 0.6434; Pooled R² = 0.6499',
        'evidence_and_scientific_nuance': 'The arithmetic mean of the 9 state R² values is 0.6434 (or 0.7018 if Nebraska is excluded). The value 0.6879 in narrative text was an uncorrected transcription artifact from an exploratory draft.'
    },
    {
        'claim_id': 'CLAIM-06',
        'topic': 'Cold-Start LOSO R²',
        'phase5_statement': 'Average spatial holdout R² across the 9 states is 0.3703 in Scenario 2 (Cold Start).',
        'audit_classification': 'REQUIRES_CORRECTION',
        'recalculated_value': 'Arithmetic Macro-Average R² = 0.3259; Pooled R² = 0.3358',
        'evidence_and_scientific_nuance': 'The exact arithmetic average across the 9 states is 0.3259 (or 0.4783 excluding Nebraska). Narrative report cited 0.3703.'
    },
    {
        'claim_id': 'CLAIM-07',
        'topic': 'Nebraska Anomaly',
        'phase5_statement': 'Nebraska cold-start prediction failure is caused by Ogallala aquifer irrigation decoupling.',
        'audit_classification': 'VERIFIED_WITH_LIMITATIONS',
        'recalculated_value': 'Bias: -28.56 bu/ac (Scen 2) -> -14.81 bu/ac (Scen 1)',
        'evidence_and_scientific_nuance': 'Empirical bias is confirmed and USGS/USDA records show >62% irrigation. However, without field-level irrigation volume observations in the feature matrix, this is an agronomic explanatory hypothesis, not a mathematically proven causal mechanism.'
    },
    {
        'claim_id': 'CLAIM-08',
        'topic': 'Forecast Aggregation',
        'phase5_statement': 'Acreage-weighted yield is 178.54 bu/ac vs 166.72 bu/ac unweighted (+11.83 bu/ac discrepancy).',
        'audit_classification': 'VERIFIED',
        'recalculated_value': '+11.83 bu/ac (+7.09% aggregation bias)',
        'evidence_and_scientific_nuance': 'Verified exactly. Equal weighting of counties introduces severe aggregation distortion by treating tiny peri-urban counties equally with large high-yielding production centers.'
    },
    {
        'claim_id': 'CLAIM-09',
        'topic': 'Production Multiplication',
        'phase5_statement': 'Projected 2026 Corn Belt grain production is 10,987.8 million bushels (based on 61.54M planted acres).',
        'audit_classification': 'REQUIRES_CORRECTION',
        'recalculated_value': '10,600.0 million bushels across 59.39M harvested acres',
        'evidence_and_scientific_nuance': 'USDA corn grain yield is defined per HARVESTED acre. In states like Wisconsin, 23.5% of planted corn is harvested as silage, not grain. Using planted acres directly overstates total grain production by 387.8 million bushels.'
    },
    {
        'claim_id': 'CLAIM-10',
        'topic': 'Conformal Intervals',
        'phase5_statement': 'Distribution-free conformal prediction guarantees 80%, 90%, 95% coverage on future forecasts.',
        'audit_classification': 'VERIFIED_WITH_LIMITATIONS',
        'recalculated_value': 'Empirical coverage: 80% -> 78.40%, 90% -> 89.46%, 95% -> 94.94%',
        'evidence_and_scientific_nuance': 'Average coverage matches nominal levels on out-of-sample data. However, temporal drift causes annual fluctuations (e.g. 2025 90% coverage was 86.95%), violating exchangeability. They must be designated prediction intervals, not confidence intervals.'
    },
    {
        'claim_id': 'CLAIM-11',
        'topic': 'Technology Trend',
        'phase5_statement': 'Historical genetic and technological yield trend increases at ~+1.8 bu/acre/year.',
        'audit_classification': 'VERIFIED_WITH_LIMITATIONS',
        'recalculated_value': 'Modern trend (2002-2025): +2.45 bu/ac/yr (95% CI: [1.73, 3.17])',
        'evidence_and_scientific_nuance': '+1.8 bu/ac/yr is consistent with national 60-year trends, but modern Corn Belt data shows +2.45 bu/ac/yr. Yield growth represents a composite of genetics, planting density, fertilizer, and management, not genetics alone.'
    },
    {
        'claim_id': 'CLAIM-12',
        'topic': 'July Thermal Cliff',
        'phase5_statement': 'July maximum temperature exhibits a sharp physiological damage threshold at 86.0°F (30.0°C).',
        'audit_classification': 'VERIFIED_WITH_LIMITATIONS',
        'recalculated_value': '504 test observations >= 86.0°F; Inflection at 86°F confirmed',
        'evidence_and_scientific_nuance': 'Inflection is verified in TreeSHAP. However, July TMax and Precipitation are negatively correlated (r = -0.25 to -0.45), indicating this reflects a compound atmospheric heat-moisture stress threshold rather than pure heat alone.'
    },
    {
        'claim_id': 'CLAIM-13',
        'topic': 'LOSO Generalization Scope',
        'phase5_statement': 'Leave-One-State-Out CV proves spatial-temporal forecasting generalization.',
        'audit_classification': 'VERIFIED_WITH_LIMITATIONS',
        'recalculated_value': 'Spatial-only holdout test',
        'evidence_and_scientific_nuance': 'LOSO tests spatial transferability across geographic boundaries, but is NOT a prospective temporal forecast because training data includes concurrent and future years from neighboring states.'
    }
]

df_claims = pd.DataFrame(claims)
claim_csv = os.path.join(out_dir, "PHASE6_FINAL_CLAIM_AUDIT.csv")
df_claims.to_csv(claim_csv, index=False)
print(f"Saved PHASE6_FINAL_CLAIM_AUDIT.csv -> {claim_csv}")

