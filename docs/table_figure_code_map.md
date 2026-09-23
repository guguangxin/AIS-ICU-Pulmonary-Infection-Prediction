# Table and figure to code map

This file maps the tables and figures in the current revised manuscript and Supplementary Material to their public generating or transparent reconstruction routes.

Patient-level inputs and generated patient-level outputs remain restricted and are not distributed in the public repository.

## Main manuscript

| Item | Public code / reconstruction route | Notes |
|---|---|---|
| Table 1 | `code/00_table1/Table1_baseline_characteristics.py` | Generates the concise final main baseline table. Additional first-48-hour intervention/medication candidate variables are emitted separately as Table S32. |
| Figure 1 | `code/01_feature_selection/primary_feature_selection.R` | Generates the revised feature-selection outputs after prespecified/data-quality exclusions: LASSO coefficient path, cross-validation plot, Boruta importance output, and correlation matrix for the final 11 predictors. Thirty-four variables enter LASSO/Boruta after the training-set zero-variance rule. |
| Figure 2 | `code/02_primary_analysis/primary_11_predictor_models_and_shap.py` | Generates apparent-training and fixed-test ROC panels, full-range descriptive decision-curve analysis, and fixed-test precision-recall. ROC labels use empirical AUC point estimates calculated directly from the evaluated probabilities; displayed values are rounded to three decimals and ordering uses unrounded values. |
| Figure 3 | `code/02_primary_analysis/primary_11_predictor_models_and_shap.py` | Generates apparent-training/fixed-test radial metric summaries and pre-/post-Platt calibration panels. |
| Figure 4 | `code/02_primary_analysis/primary_11_predictor_models_and_shap.py` | Generates the Primary11 GBDT SHAP components used in the manuscript, including beeswarm/global importance, main/interaction effects, and representative waterfall cases. The final manuscript figure harmonizes display precision of the common SHAP expected value as `E[f(X)] = -0.475`; this is a display-only correction and does not alter SHAP values or model results. |

## Supplementary tables

| Item | Public code / reconstruction route | Notes |
|---|---|---|
| Table S1 | `code/08_reporting_reconstruction/reconstruct_reporting_tables_and_flowchart.py` + `reporting_metadata.json` | Uses archived aggregate missingness metadata because the complete pre-imputation participant-level matrix was not retained. |
| Table S2 | `code/02_primary_analysis/primary_11_predictor_models_and_shap.py` | Hyperparameter search spaces, Bayesian optimization, selected `best_params`, and boundary status. |
| Table S3 | `code/08_reporting_reconstruction/reconstruct_reporting_tables_and_flowchart.py` | Historical aggregate Boruta stability retained for traceability. |
| Table S4 | `code/08_reporting_reconstruction/reconstruct_reporting_tables_and_flowchart.py` | Historical aggregate LASSO stability retained for traceability. |
| Table S5 | `code/02_primary_analysis/primary_11_predictor_models_and_shap.py` | Apparent training discrimination/Brier summaries with bootstrap intervals. |
| Table S6 | `code/02_primary_analysis/primary_11_predictor_models_and_shap.py` | Fixed internal test metrics at locked cross-fitted training-derived thresholds. |
| Table S7 | `code/02_primary_analysis/primary_11_predictor_models_and_shap.py` | Pre-Platt fixed-test calibration. |
| Table S8 | `code/04_calibration/Primary11_bootstrap_calibration_ECE_MCE.py` | Post-Platt calibration, including ECE/MCE descriptive bootstrap percentile distributions. |
| Table S9 | `code/02_primary_analysis/primary_11_predictor_models_and_shap.py` | Apparent training calibration. |
| Table S10 | `code/02_primary_analysis/primary_11_predictor_models_and_shap.py` | Focused 0.25-0.40 decision-curve analysis with hypothetical flagging/missed-event burden. |
| Table S11 | `code/06_historical_bsa_inclusive/historical_BSA_TableS11_S12.py` | Historical BSA-inclusive versus current Primary11 discrimination comparison; descriptive only because the predictor sets differ beyond BSA. |
| Table S12 | `code/06_historical_bsa_inclusive/historical_BSA_TableS11_S12.py` | Historical BSA-inclusive versus current Primary11 calibration comparison; descriptive only. |
| Table S13 | `code/02_primary_analysis/primary_11_predictor_models_and_shap.py` | Paired fixed-test DeLong comparisons using GBDT as reference, with Holm adjustment. |
| Table S14 | `code/07_historical_nine_predictor_traceability/historical_9predictor_bootstrap_optimism.py` | Historical fixed-hyperparameter optimism reconstruction retained for traceability only. |
| Table S15 | `code/02_primary_analysis/export_primary_lr_coefficients.py` | Base Primary11 logistic-regression coefficients back-transformed to original measurement units. |
| Table S16 | `code/08_reporting_reconstruction/reconstruct_reporting_tables_and_flowchart.py` | Aggregate component-availability metadata for A2DS2/ISAN comparison constraints. |
| Table S17 | `code/08_reporting_reconstruction/reconstruct_reporting_tables_and_flowchart.py` | Reporting, reproducibility, and methodological-transparency metadata. |
| Table S18 | `code/03_sensitivity_analyses/temporal_robustness.py` | 2020-2023 development versus 2024-2025 later-period robustness analysis. |
| Table S19 | `code/02_primary_analysis/primary_11_predictor_models_and_shap.py` | Strict cross-fitted out-of-fold threshold derivation summary. |
| Table S20 | `code/03_sensitivity_analyses/hap_only_sensitivity.py` | HAP-only fixed-test performance after exclusion of VAP cases. |
| Table S21 | `code/03_sensitivity_analyses/hap_only_sensitivity.py` | Paired HAP-only change after removal of MV and intubation/tracheotomy. |
| Table S22 | `code/03_sensitivity_analyses/followup_opportunity_sensitivity.py` | Recorded post-landmark observation opportunity, crude event proportions, and in-hospital mortality summaries. Exact death timing relative to the landmark/ICU exit is unavailable, and discharge/transfer was not modeled as a time-to-event competing process. |
| Table S23 | `code/03_sensitivity_analyses/followup_opportunity_sensitivity.py` | Primary11 GBDT/LR performance across remaining-ICU-stay strata and restriction subsets. |
| Table S24 | `code/08_reporting_reconstruction/reconstruct_reporting_tables_and_flowchart.py` | Archived aggregate screening-flow metadata; the original person-level screening log is unavailable. |
| Table S25 | `code/01_feature_selection/primary_feature_selection.R` | Revised primary feature selection after exclusion of broad-spectrum antibiotic exposure, the all-zero immunosuppressant field, and the nonfunctional standalone heart-disease field; the training-set zero-variance rule subsequently leaves 34 variables entering LASSO/Boruta. |
| Table S26 | `code/03_sensitivity_analyses/cci_age_decomposition_sensitivity.py` | Compares Primary11 age-adjusted CCI with a 12-predictor decomposition replacing CCI by the unadjusted Charlson disease score plus chronological age. This supersedes the earlier age-forced sensitivity analysis. |
| Table S27 | `code/05_repeated_nested_validation/01_repeated_nested_feature_selection.R` | Outer-fold feature-selection frequency across the 5 × 5 repeated nested analysis. This run preceded the later data-quality exclusion of the standalone heart-disease field; that field was selected in 0/25 outer folds. |
| Table S28 | `code/05_repeated_nested_validation/02_repeated_nested_model_validation.py` | Outer-fold LR and GBDT performance for the recoverable post-preprocessing development pipeline. |
| Table S29 | `code/08_reporting_reconstruction/reconstruct_reporting_tables_and_flowchart.py` | Operational coding and timing metadata for the 11 primary predictors. |
| Table S30 | `code/03_sensitivity_analyses/parsimonious_comparator_training.py` + `code/03_sensitivity_analyses/parsimonious_comparator.py` | Four simple LR comparator specifications versus Primary11 LR in the fixed internal test partition. |
| Table S31 | `code/08_reporting_reconstruction/reconstruct_reporting_tables_and_flowchart.py` | Aggregate sparse-field metadata and reconciliation of first-48-hour airway indicators with re-audited VAP subtype. |
| Table S32 | `code/00_table1/Table1_baseline_characteristics.py` | Additional first-48-hour intervention/medication candidate variables moved out of main Table 1 for readability. |
| Table S33 | `code/03_sensitivity_analyses/lasso_order_predictor_count_curve.py` | Post hoc cumulative Top1-through-Top11 LR analysis using the revised R/glmnet `lambda.1se` absolute-coefficient ordering. The fixed test set is used descriptively and no reduced predictor count is selected from test performance. |
| Table S34 | `code/03_sensitivity_analyses/rcs_logistic_regression_sensitivity.py` | Restricted-cubic-spline sensitivity comparing the primary linear LR with RCS-LR; six continuous laboratory predictors use four training-derived knots, while CCI remains linear and binary predictors retain their original form. |

## Supplementary figures

| Item | Public code / reconstruction route | Notes |
|---|---|---|
| Figure S1 | `code/08_reporting_reconstruction/reconstruct_reporting_tables_and_flowchart.py` | Archived patient-inclusion flow using retained historical aggregate counts; the two late-stage exclusion blocks cannot be decomposed further from the retained records. |
| Figure S2 | `code/06_historical_bsa_inclusive/historical_BSA_LightGBM_SHAP.py` | Historical BSA-inclusive LightGBM SHAP dependence plots. |
| Figure S3 | `code/04_calibration/Primary11_bootstrap_calibration_ECE_MCE.py` | Bootstrap calibration plots for the eight current Primary11 models. |
| Figure S4 | `code/06_historical_bsa_inclusive/historical_BSA_10predictor_models.py` | Historical BSA-inclusive discrimination and decision-analytic figure. |
| Figure S5 | `code/06_historical_bsa_inclusive/historical_BSA_10predictor_models.py` | Historical BSA-inclusive metric overview and calibration figure. |
| Figure S6 | `code/06_historical_bsa_inclusive/historical_BSA_LightGBM_SHAP.py` | Historical exploratory BSA-inclusive SHAP interpretation. |
| Figure S7 | `code/02_primary_analysis/primary_11_predictor_models_and_shap.py` | Focused decision-curve reanalysis with pointwise bootstrap uncertainty. |
| Figure S8 | `code/03_sensitivity_analyses/lasso_order_predictor_count_curve.py` | Fixed-test AUC versus cumulative predictor count in the revised R/glmnet coefficient-magnitude order. The figure is descriptive and was not used to select a reduced model. |
| Figure S9 | `code/03_sensitivity_analyses/rcs_logistic_regression_sensitivity.py` | Fixed-test calibration comparison of primary linear LR, RCS-LR, and locked Primary11 GBDT. |

## Important reconstruction boundary

Some reporting items cannot be derived de novo from the retained analysis-ready participant matrix because the revision archive does not contain the complete pre-imputation raw predictor matrix, uncapped predictor matrix, excluded-patient covariates, complete person-level screening log, reliable reason-specific decomposition of the two historical grouped exclusion blocks, or exact first-48-hour specimen/treatment timestamps.

For those items, the repository uses version-controlled **aggregate revision metadata** rather than reconstructing or inventing unavailable patient-level history.

The repeated nested-resampling analysis similarly starts from the archived analysis-ready matrix, which had already undergone the original single-imputation and winsorization procedures. Consequently, the original imputation models and winsorization limits cannot be re-estimated within each outer training fold, and uncertainty from those preprocessing steps is not propagated into the repeated-resampling estimates.

The repeated nested analysis should therefore be interpreted as an evaluation of the **recoverable post-preprocessing development pipeline**, not as full internal validation of the complete raw-data-to-model pipeline.

The fixed internal test partition is used as a standardized within-study comparison set. Because revision-specific analyses were performed after its results had already been examined, it should not be interpreted as an untouched validation cohort for the entire revised workflow.

## Historical-analysis labeling

Files under:

```text
code/06_historical_bsa_inclusive/
code/07_historical_nine_predictor_traceability/
```

are retained for historical traceability and are not part of the current primary model specification.

In these historical files, `BSA` refers to broad-spectrum antibiotic exposure, not body surface area.

The superseded age-forced sensitivity analysis is not part of the final revised analysis and should not be used as the reconstruction route for Table S26.
