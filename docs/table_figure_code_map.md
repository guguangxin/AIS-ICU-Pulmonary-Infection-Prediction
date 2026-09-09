# Table and figure to code map

This file maps every table and figure in the current revision to a public generating or transparent reconstruction route. Patient-level inputs and outputs remain restricted.

## Main manuscript

| Item | Public code / reconstruction route | Notes |
|---|---|---|
| Table 1 | `code/00_table1/Table1_baseline_characteristics.py` | Generates the concise final main table; additional intervention/medication candidates are emitted separately as Table S32. |
| Figure 1 | `code/01_feature_selection/primary_feature_selection.R` | LASSO path/CV, Boruta, and current 11-variable correlation outputs used for the feature-selection figure. |
| Figure 2 | `code/02_primary_analysis/primary_11_predictor_models_and_shap.py` | Training/test ROC, full-range descriptive DCA, and test precision-recall. |
| Figure 3 | `code/02_primary_analysis/primary_11_predictor_models_and_shap.py` | Apparent training/test radial summaries and pre-/post-Platt calibration panels. |
| Figure 4 | `code/02_primary_analysis/primary_11_predictor_models_and_shap.py` | Current Primary11 GBDT SHAP beeswarm, global importance, interaction matrix, and waterfall cases. |

## Supplementary tables

| Item | Public code / reconstruction route |
|---|---|
| Table S1 | `code/08_reporting_reconstruction/reconstruct_reporting_tables_and_flowchart.py` + `reporting_metadata.json` (archived aggregate missingness) |
| Table S2 | `code/02_primary_analysis/primary_11_predictor_models_and_shap.py` (`models_config`, Bayesian search, selected `best_params`) |
| Table S3 | `code/08_reporting_reconstruction/reconstruct_reporting_tables_and_flowchart.py` (historical aggregate Boruta stability) |
| Table S4 | `code/08_reporting_reconstruction/reconstruct_reporting_tables_and_flowchart.py` (historical aggregate LASSO stability) |
| Table S5 | `code/02_primary_analysis/primary_11_predictor_models_and_shap.py` (training metrics + 1,000 bootstrap intervals) |
| Table S6 | `code/02_primary_analysis/primary_11_predictor_models_and_shap.py` (fixed-test metrics at locked cross-fitted thresholds) |
| Table S7 | `code/02_primary_analysis/primary_11_predictor_models_and_shap.py` (pre-Platt fixed-test calibration) |
| Table S8 | `code/04_calibration/Primary11_bootstrap_calibration_ECE_MCE.py` |
| Table S9 | `code/02_primary_analysis/primary_11_predictor_models_and_shap.py` (apparent training calibration) |
| Table S10 | `code/02_primary_analysis/primary_11_predictor_models_and_shap.py` (focused 0.25-0.40 DCA bootstrap analysis) |
| Table S11 | `code/06_historical_bsa_inclusive/historical_BSA_TableS11_S12.py` |
| Table S12 | `code/06_historical_bsa_inclusive/historical_BSA_TableS11_S12.py` |
| Table S13 | `code/02_primary_analysis/primary_11_predictor_models_and_shap.py` (paired DeLong + Holm) |
| Table S14 | `code/07_historical_nine_predictor_traceability/historical_9predictor_bootstrap_optimism.py` |
| Table S15 | `code/02_primary_analysis/export_primary_lr_coefficients.py` |
| Table S16 | `code/08_reporting_reconstruction/reconstruct_reporting_tables_and_flowchart.py` (aggregate component-availability metadata) |
| Table S17 | `code/08_reporting_reconstruction/reconstruct_reporting_tables_and_flowchart.py` (reporting/transparency metadata) |
| Table S18 | `code/03_sensitivity_analyses/temporal_robustness.py` |
| Table S19 | `code/02_primary_analysis/primary_11_predictor_models_and_shap.py` (strict cross-fitted OOF threshold summary) |
| Table S20 | `code/03_sensitivity_analyses/hap_only_sensitivity.py` |
| Table S21 | `code/03_sensitivity_analyses/hap_only_sensitivity.py` |
| Table S22 | `code/03_sensitivity_analyses/followup_opportunity_sensitivity.py` |
| Table S23 | `code/03_sensitivity_analyses/followup_opportunity_sensitivity.py` |
| Table S24 | `code/08_reporting_reconstruction/reconstruct_reporting_tables_and_flowchart.py` (aggregate screening-flow metadata) |
| Table S25 | `code/01_feature_selection/primary_feature_selection.R` |
| Table S26 | `code/03_sensitivity_analyses/age_forced_sensitivity.py` |
| Table S27 | `code/05_repeated_nested_validation/01_repeated_nested_feature_selection.R` |
| Table S28 | `code/05_repeated_nested_validation/02_repeated_nested_model_validation.py` |
| Table S29 | `code/08_reporting_reconstruction/reconstruct_reporting_tables_and_flowchart.py` (operational coding/timing metadata) |
| Table S30 | `code/03_sensitivity_analyses/parsimonious_comparator_training.py` + `parsimonious_comparator.py` |
| Table S31 | `code/08_reporting_reconstruction/reconstruct_reporting_tables_and_flowchart.py` (aggregate sparse-field/VAP audit metadata) |
| Table S32 | `code/00_table1/Table1_baseline_characteristics.py` (separate final supplementary output) |

## Supplementary figures

| Item | Public code / reconstruction route |
|---|---|
| Figure S1 | `code/08_reporting_reconstruction/reconstruct_reporting_tables_and_flowchart.py` |
| Figure S2 | `code/06_historical_bsa_inclusive/historical_BSA_LightGBM_SHAP.py` |
| Figure S3 | `code/04_calibration/Primary11_bootstrap_calibration_ECE_MCE.py` |
| Figure S4 | `code/06_historical_bsa_inclusive/historical_BSA_10predictor_models.py` |
| Figure S5 | `code/06_historical_bsa_inclusive/historical_BSA_10predictor_models.py` |
| Figure S6 | `code/06_historical_bsa_inclusive/historical_BSA_LightGBM_SHAP.py` |
| Figure S7 | `code/02_primary_analysis/primary_11_predictor_models_and_shap.py` (focused DCA pointwise bootstrap bands) |

## Important reconstruction boundary

Some reporting items cannot be derived de novo from the retained analysis-ready participant matrix because the revision archive does not contain the pre-imputation raw predictor matrix, excluded-patient covariates, full reason-specific exclusion subcounts, or exact first-48-hour specimen/treatment timestamps. For those items, the repository uses version-controlled **aggregate revision metadata** rather than inventing unavailable patient-level history. This limitation is separate from the current Primary11 statistical analyses, which are reconstructed from the authorized analysis-ready cohort.
