# Model reconstruction guide

## Purpose

This document describes how to reconstruct the analyses reported for the current 11-predictor primary model set using an authorized local copy of the analysis-ready dataset.

Patient-level data are not distributed in this public repository.

The repository is intended to document the statistical and modeling logic used for the revised manuscript and to distinguish analyses that can be reconstructed from the retained revision archive from historical source-level operations that can no longer be reproduced.

## Prediction target

The operational prediction target is a later-documented hospital-acquired pneumonia (HAP) or ventilator-associated pneumonia (VAP) event occurring more than 48 hours after ICU admission and before ICU exit.

The target therefore has a variable follow-up horizon after the 48-hour landmark rather than a fixed 7-day or 14-day prediction window.

## Cohort anchors

The archived analysis-ready cohort contains:

- 3,368 patients;
- 1,357 documented pulmonary-infection events.

The preserved fixed internal comparison split contains:

- training set: n = 2,357; events = 950;
- fixed internal test set: n = 1,011; events = 407.

The preserved `Primary_split` field should be used whenever available.

Split identity should not be reconstructed or inferred from patient identifiers.

The current fixed test partition is used as a standardized within-study comparison set. Because several revision-specific analyses were conducted after results from this partition had already been examined, it should not be described as untouched validation of the entire revised workflow.

## Required metadata fields

At minimum, the current primary workflow requires:

- `Study_row_id`
- `Primary_split`
- `Pulmonary_infection`

The current primary 11 predictors are:

| Manuscript label | Analysis field | Type |
|---|---|---|
| NEU | `NEUT_abs` | continuous |
| Intubation/tracheotomy | `Intubation_tracheotomy` | binary |
| MV | `Mechanical_ventilation` | binary |
| LDH | `LDH` | continuous |
| LYM | `LYMPH_abs` | continuous |
| BUN | `BUN` | continuous |
| CCI | `CCI` | continuous age-adjusted Charlson score |
| FIB | `FIB` | continuous |
| Surgery | `Surgery` | binary |
| Diuretics | `Diuretics` | binary |
| TCO2 | `CO2` | continuous |

Laboratory predictors represent the first available result during the first 48 hours after ICU admission.

Procedure, organ-support, and medication predictors represent any documented exposure during the first 48 hours.

They do not represent exposure duration or status active exactly at the 48-hour landmark.

The revision archive does not retain exact laboratory sampling-time distributions, admission-versus-later sampling status, treatment start/stop times, cumulative exposure duration, or active-at-landmark status.

## Current primary feature-selection reconstruction

Run:

```bash
Rscript code/01_feature_selection/primary_feature_selection.R
```

The current primary feature-selection workflow uses prespecified and data-quality exclusions before data-driven selection.

The workflow:

1. excludes early broad-spectrum antibiotic exposure before data-driven selection because exposure was recorded during the same first-48-hour prediction window and was considered particularly susceptible to treatment-response behavior, confounding by indication, and protopathic bias;
2. excludes immunosuppressant use because it is all zero in the archived cohort;
3. excludes the standalone `heart disease` field on data-quality grounds because it did not validly represent overall cardiac comorbidity and was distinct from the ICD-10-coded diagnoses used to construct the Charlson index;
4. leaves 35 candidate fields after these prespecified and data-quality exclusions;
5. removes training-set zero-variance fields, including cholinesterase-inhibitor exposure, leaving 34 variables entering data-driven selection;
6. runs binomial LASSO (`alpha = 1`) with 10-fold cross-validation and `lambda.1se`;
7. runs Boruta with 1,000 trees and `maxRuns = 100`, followed by `TentativeRoughFix`;
8. defines the primary selected set as the LASSO-Boruta intersection.

The revised 34-candidate selection retained the same 11 predictors listed above.

LASSO selected 11 variables and Boruta confirmed 28 variables. Their intersection contained the same 11 predictors as the locked Primary11 specification, with no predictors added or removed.

In 20 repeated runs using the same fixed training cohort, all 11 retained predictors were selected by LASSO in 20/20 runs.

The LASSO absolute-coefficient ordering was also unchanged after exclusion of the standalone heart-disease field.

## Age-adjusted Charlson Comorbidity Index

The `CCI` field used in the primary analysis is an age-adjusted Charlson Comorbidity Index rather than a comorbidity-only score.

The age component is:

- 0 points for age <50 years;
- 1 point for age 50-59 years;
- 2 points for age 60-69 years;
- 3 points for age 70-79 years;
- 4 points for age >=80 years.

Disease components are derived from ICD-10-coded diagnoses using the Charlson scoring framework, including the cerebrovascular-disease component.

The standalone `heart disease` candidate field is distinct from the ICD-10-coded diagnoses used to derive myocardial-infarction and congestive-heart-failure components of the Charlson index.

Its exclusion from revised feature selection therefore does not alter construction of the age-adjusted CCI.

## Primary eight-model reconstruction

Run:

```bash
python code/02_primary_analysis/primary_11_predictor_models_and_shap.py
```

The primary workflow evaluates:

- logistic regression;
- naive Bayes;
- decision tree;
- random forest;
- gradient boosting decision tree;
- XGBoost;
- LightGBM;
- multilayer perceptron.

Key reconstruction rules are:

- preserve the fixed training/test identities;
- fit MinMax scaling parameters on the fixed training set for continuous variables only and apply them unchanged to the fixed test set;
- optimize hyperparameters by 10-fold stratified Bayesian search in the training set;
- fit 10-fold sigmoid/Platt calibration using training data;
- derive model-specific Youden thresholds from strict cross-fitted calibrated training probabilities with the already selected full-training hyperparameters held fixed;
- lock thresholds before applying the final calibrated models to the fixed internal test partition;
- use fixed-test probabilities for ROC, precision-recall, calibration, decision-curve, DeLong/Holm, and related performance summaries;
- use the uncalibrated GBDT base estimator for the primary TreeSHAP interpretation while using calibrated probabilities for probability-based performance evaluation.

The observed fixed-test AUC anchors are:

- GBDT: 0.8362
- LightGBM: 0.8347
- XGBoost: 0.8345
- random forest: 0.8340
- decision tree: 0.8278
- MLP: 0.8274
- logistic regression: 0.8273
- naive Bayes: 0.8232

These empirical point estimates are the statistical-analysis anchors.

The current ROC figure displays empirical fixed-test AUC point estimates rounded to three decimal places for presentation. Model ordering is based on the unrounded AUC values.

## Base logistic-regression coefficient reconstruction

Supplementary Table S15 reports the uncalibrated base logistic-regression coefficients in original measurement units.

Run:

```bash
python code/02_primary_analysis/export_primary_lr_coefficients.py \
  --input /path/to/authorized/analysis_ready.csv
```

The script uses the selected primary LR hyperparameters (`C = 0.2493730`, L2 penalty, `lbfgs`) and back-transforms coefficients for continuous predictors from the MinMax-scaled fitting space to original measurement units.

The reported CCI coefficient corresponds to the age-adjusted CCI used in the primary model.

## Calibration uncertainty

Run:

```bash
python code/04_calibration/Primary11_bootstrap_calibration_ECE_MCE.py
```

The calibration workflow uses approximately equal-frequency risk groups for ECE/MCE and patient-level bootstrap resampling.

Bootstrap distributions of binned calibration-error statistics are treated as descriptive percentile distributions rather than conventional confidence intervals when reported as such in the Supplementary Material.

## Sensitivity and robustness analyses

Repository-facing scripts are located under:

```text
code/03_sensitivity_analyses/
```

### Age-adjusted CCI decomposition sensitivity

Script:

```text
cci_age_decomposition_sensitivity.py
```

This analysis replaces the primary age-adjusted CCI with:

1. the unadjusted Charlson disease score; and
2. chronological age as a separate predictor.

The remaining 10 primary predictors, fixed training/test identities, modeling procedures, and evaluation framework are retained.

For three participants whose archived CCI values had previously been imputed, the unadjusted score is reconstructed by subtracting the deterministic age component from the imputed age-adjusted CCI without rounding.

This analysis supersedes the earlier age-forced sensitivity analysis.

For logistic regression, the primary age-adjusted-CCI specification had a fixed-test AUC of approximately 0.8273 and the decomposed specification approximately 0.8274.

Across the eight algorithms, no AUC difference between the primary and decomposed specifications remained statistically significant after Holm correction.

### Restricted-cubic-spline logistic-regression sensitivity

Script:

```text
rcs_logistic_regression_sensitivity.py
```

This analysis evaluates whether nonlinear functional forms materially improve the primary logistic-regression model.

Restricted cubic spline terms are applied to:

- NEU;
- LDH;
- LYM;
- BUN;
- FIB;
- TCO2.

The age-adjusted CCI is retained as a linear term, and the four binary predictors are retained unchanged.

Four training-derived knots are used at the:

- 5th percentile;
- 35th percentile;
- 65th percentile;
- 95th percentile.

Hyperparameter tuning and Platt calibration use training data only.

The fixed internal test set is used for final comparison with the primary linear LR model.

This analysis addresses whether nonlinear functional form materially changes LR discrimination or calibration.

### LASSO-ordered cumulative predictor analysis

Script:

```text
lasso_order_predictor_count_curve.py
```

Predictors are ordered by decreasing absolute coefficient magnitude from the revised R/glmnet LASSO solution at `lambda.1se` and entered cumulatively from Top 1 through Top 11.

The retained order is:

1. intubation/tracheotomy;
2. mechanical ventilation;
3. surgery;
4. LYM;
5. NEU;
6. CCI;
7. FIB;
8. diuretics;
9. TCO2;
10. BUN;
11. LDH.

The LASSO ordering remained unchanged after the standalone heart-disease field was excluded on data-quality grounds.

The observed fixed-test AUCs for the cumulative LR models were approximately:

- Top 1: 0.603
- Top 2: 0.627
- Top 3: 0.705
- Top 4: 0.753
- Top 5: 0.802
- Top 6: 0.817
- Top 7: 0.824
- Top 8: 0.827
- Top 9: 0.826
- Top 10: 0.828
- Top 11: 0.827

The Top-11 result reproduces the locked Primary11 LR AUC.

The analysis is post hoc and descriptive.

The fixed internal test partition was not used to select an alternative reduced final model.

### HAP-only sensitivity

Script:

```text
hap_only_sensitivity.py
```

This analysis excludes adjudicated VAP cases and evaluates prediction of non-VAP HAP.

It compares:

- the current 11-predictor specification; and
- a corresponding 9-predictor specification after removal of mechanical ventilation and intubation/tracheotomy.

The analysis evaluates whether predictive performance remains present after exclusion of VAP cases and removal of the two airway-support predictors.

### Temporal robustness analysis

Script:

```text
temporal_robustness.py
```

This analysis uses:

- 2020-2023 as the development period;
- 2024-2025 as the later-period evaluation cohort.

The current 11-predictor specification and the eight primary algorithms are evaluated.

Because the archived matrix had already undergone the original imputation and winsorization procedures and the predictor-selection history preceded the chronological split, this analysis is interpreted as a post hoc temporal robustness analysis rather than independent temporal validation of the complete raw-data-to-model pipeline.

### Observation-opportunity sensitivity

Script:

```text
followup_opportunity_sensitivity.py
```

This analysis evaluates discrimination and calibration across strata of recorded remaining ICU stay after the 48-hour landmark using locked Primary11 GBDT and LR probabilities.

Models are not refitted within strata.

Remaining ICU stay is treated descriptively because pulmonary infection itself may prolong ICU stay.

Available mortality information records in-hospital death but does not establish the timing of death relative to the 48-hour landmark or ICU exit.

ICU discharge or transfer was not modeled as a time-to-event competing process.

### Parsimonious comparator reconstruction

When an authorized analysis-ready cohort is available, run:

```text
parsimonious_comparator_training.py
```

before:

```text
parsimonious_comparator.py
```

The four simple LR comparator specifications are:

1. mechanical ventilation + intubation/tracheotomy;
2. neutrophil-to-lymphocyte ratio alone (`NEU / LYM`);
3. age + sex as a component-level floor;
4. NEU + LYM + mechanical ventilation.

Continuous variables are min-max scaled using training-set parameters only.

L2 logistic regression is tuned with 10-fold stratified Bayesian cross-validation and followed by 10-fold Platt calibration in training.

The resulting fixed-test probabilities are then passed to the downstream comparator script for paired performance comparisons.

These simple specifications are not established stroke-pneumonia scores.

A2DS2 and ISAN could not be validly reconstructed because several required components were unavailable in standardized analyzable form.

See the README in `code/03_sensitivity_analyses/` for additional restricted inputs and environment variables.

## Historical exploratory BSA-inclusive reconstruction

Repository-facing historical scripts are located under:

```text
code/06_historical_bsa_inclusive/
```

Here, BSA refers to the archived broad-spectrum-antibiotic-exposure field rather than body surface area.

These scripts reconstruct the archived BSA-inclusive eight-model analysis, historical LightGBM SHAP figures, and descriptive BSA-inclusive-versus-current comparisons.

These analyses are historical and exploratory.

The historical BSA-inclusive predictor set and the current 11-predictor BSA-free set differ by more than the BSA field alone, so the comparison is descriptive and must not be interpreted as an isolated incremental-BSA or causal-antibiotic effect.

The public historical scripts preserve the archived workflow while replacing workstation-specific absolute paths.

Patient-level data, patient-level predictions, and fitted model objects are not distributed publicly.

## Historical nine-predictor optimism reconstruction

Run:

```bash
python code/07_historical_nine_predictor_traceability/historical_9predictor_bootstrap_optimism.py \
  --input /path/to/authorized/analysis_ready.csv \
  --bootstrap 1000
```

This is a reconstruction of a historical traceability analysis, not a recovered copy of the original generation script.

It uses the earlier nine-predictor BSA-free specification:

- NEU;
- intubation/tracheotomy;
- MV;
- LDH;
- LYM;
- BUN;
- CCI;
- FIB;
- surgery.

Selected GBDT/LightGBM hyperparameter values retained in the archived Supplementary Table S2 are used.

Hyperparameters are held fixed.

Each bootstrap-fitted base estimator is evaluated in both its bootstrap sample and the complete original training cohort, and optimism is defined as the difference in AUC.

The bundled historical parameter JSON uses continuous values at the precision displayed in the archived table because a full-precision historical `best_params_` export was not located.

The script therefore uses archived Supplementary Table S14 values as QA anchors and documents possible small deviations from parameter rounding or software-version differences.

This historical analysis is retained for traceability only and is not an uncertainty estimate for the current Primary11 models.

## Repeated nested resampling

The repeated nested analysis is intentionally split into R and Python stages under:

```text
code/05_repeated_nested_validation/
```

### Stage 1

Script:

```text
01_repeated_nested_feature_selection.R
```

Across 5 repeats × 5 outer folds, feature selection is repeated within each outer training fold.

The historical repeated-resampling run excluded broad-spectrum antibiotic exposure before selection, removed outer-training zero-variance fields, reran LASSO and Boruta, and applied the predefined intersection rule.

### Stage 2

Script:

```text
02_repeated_nested_model_validation.py
```

Within each outer training fold, the recoverable downstream modeling steps include:

- training-only scaling;
- inner 5-fold Bayesian hyperparameter optimization;
- 5-fold Platt calibration;
- cross-fitted training-only Youden-threshold derivation;
- evaluation in the untouched outer fold for LR and GBDT.

The repeated nested analysis evaluates only the recoverable post-preprocessing development pipeline.

## Critical preprocessing limitation

The revision archive retains an already imputed and winsorized analysis-ready matrix.

The original pre-imputation and uncapped predictor matrix was not retained in the revision archive.

Consequently, the original imputation models and winsorization limits cannot be re-estimated within each repeated outer training fold.

Uncertainty arising from these preprocessing steps is therefore not propagated into the repeated-resampling performance estimates.

Accordingly, this analysis evaluates the recoverable **post-preprocessing development pipeline** rather than providing full internal validation of the complete raw-data-to-model pipeline.

The repeated nested-resampling analysis was completed before the later revision-specific data-quality exclusion of the standalone heart-disease field.

That field was not retained in the reported outer-fold final predictor sets.

The primary feature-selection workflow was subsequently rerun after exclusion of the standalone heart-disease field and recovered exactly the same locked 11-predictor specification.

The historical nested-resampling results are therefore retained as originally generated rather than presented as though every later revision-specific data-quality decision had been rerun retrospectively.

## Restricted outputs

Do not commit any patient-level input or generated patient-level output to the public repository, including:

- CSV/Excel participant datasets;
- row-level predicted probabilities;
- outer-fold assignments containing participant identifiers;
- exact patient-level dates;
- source medical-record identifiers;
- patient-level SHAP outputs;
- serialized fitted models or checkpoints if institutional policy treats them as restricted.

The repository `.gitignore` is intended to reduce the risk of accidental upload of these materials.

## Final reporting reconstruction module

The directory:

```text
code/08_reporting_reconstruction/
```

contains aggregate, non-patient-level metadata and code used to reconstruct selected reporting items that cannot all be derived de novo from the retained analysis-ready participant matrix.

These include items dependent on historical aggregate screening information, missingness summaries, or other revision metadata.

The pre-imputation raw matrix, excluded-patient covariates, exact predictor sampling timestamps, and reliable reason-specific subcounts within the two historical grouped exclusion blocks are not available.

The repository therefore preserves available aggregate revision metadata explicitly rather than inventing unavailable patient-level history.

Where applicable, run:

```bash
python code/08_reporting_reconstruction/reconstruct_reporting_tables_and_flowchart.py
```

The reporting reconstruction module does not infer unavailable patient-level information.

## Reproducibility boundary

The public repository reconstructs analyses that can be reproduced from the retained analysis-ready archive and explicitly documents limitations for analyses that depend on unavailable source-level information.

The original pre-imputation matrix, uncapped predictor matrix, complete person-level screening log, exact reason-specific decomposition of the two grouped exclusion blocks, complete excluded-patient covariates, and exact predictor timing information were not retained in the revision archive.

No unavailable patient-level information should be reconstructed by assumption.

The fixed internal test set should be interpreted as a standardized internal comparison set rather than as an untouched validation cohort for every revision-specific analysis.

## Final interpretation anchors

The observed fixed-test discrimination of the leading algorithms is similar.

GBDT has the highest observed fixed-test AUC and more favorable calibration than LR.

LR is retained as a transparent equation-based benchmark rather than designated as an overall preferred model.

No leading algorithm shows a statistically significant discriminative advantage after multiplicity adjustment in the current fixed-test comparison.

The restricted-cubic-spline sensitivity analysis does not show a material improvement sufficient to replace the primary linear LR specification.

The cumulative LASSO-ordered predictor analysis shows that most of the increase in AUC occurs before the full 11-predictor specification is reached; however, this post hoc fixed-test analysis is not used to select a reduced final model.

Positive decision-curve net benefit should not be interpreted as demonstrated clinical utility.

No model-specific management intervention has been prospectively evaluated.

SHAP values and regression coefficients describe fitted-model behavior and should not be interpreted causally.

## Release and citation

The final submitted manuscript should cite the exact GitHub release tag and commit corresponding to the final public analysis code.

Citation metadata are provided in:

```text
CITATION.cff
```

The immutable archival DOI or another persistent identifier may be added after the final versioned release is archived.

See:

```text
docs/table_figure_code_map.md
```

for the mapping between reported manuscript and supplementary outputs and public code.
