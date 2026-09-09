# Model reconstruction guide

## Purpose

This document describes how to reconstruct the analyses reported for the current BSA-free 11-predictor primary model set using an authorized local copy of the analysis-ready dataset. Patient-level data are not distributed in this repository.

## Prediction target

The operational target is a later-documented hospital-acquired pneumonia (HAP) or ventilator-associated pneumonia (VAP) event after the 48-hour landmark and before ICU exit. It is not a fixed 7-day or 14-day cumulative-incidence target.

## Cohort anchors

The archived analysis-ready cohort contains 3,368 patients and 1,357 documented pulmonary-infection events. The fixed internal comparison split contains:

- training set: n = 2,357; events = 950
- fixed internal test set: n = 1,011; events = 407

The preserved `Primary_split` field should be used whenever available. Do not infer split identity from patient identifiers.

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
| CCI | `CCI` | continuous score |
| FIB | `FIB` | continuous |
| Surgery | `Surgery` | binary |
| Diuretics | `Diuretics` | binary |
| TCO2 | `CO2` | continuous |

Treatment/support predictors represent documented occurrence during the first 48 hours after ICU admission, not full-ICU-course exposure.

## Current primary feature-selection reconstruction

Run:

```bash
Rscript code/01_feature_selection/primary_feature_selection.R
```

The current primary selection workflow:

1. excludes broad-spectrum antibiotics (BSA) before data-driven selection;
2. excludes immunosuppressant use because it is all zero in the archived cohort;
3. removes training-set zero-variance fields before selection;
4. runs binomial LASSO (`alpha=1`) with 10-fold cross-validation and `lambda.1se`;
5. runs Boruta with 1,000 trees, `maxRuns=100`, followed by `TentativeRoughFix`;
6. uses the LASSO-Boruta intersection as the primary selected set.

The current selection retains the 11 predictors listed above.

## Primary eight-model reconstruction

Run:

```bash
python code/02_primary_analysis/primary_11_predictor_models_and_shap.py
```

The primary workflow evaluates logistic regression, naive Bayes, decision tree, random forest, GBDT, XGBoost, LightGBM, and multilayer perceptron.

Key reconstruction rules:

- use the preserved fixed training/test identities;
- fit MinMax scaling parameters on the fixed training set for continuous variables only and apply them unchanged to the fixed test set;
- optimize hyperparameters by 10-fold stratified Bayesian search in the training set;
- fit 10-fold sigmoid/Platt calibration in the training set;
- derive model-specific Youden thresholds from strict cross-fitted calibrated training probabilities with the already selected full-training hyperparameters held fixed;
- lock thresholds before applying the final calibrated models to the fixed internal test partition;
- use the fixed-test probabilities for ROC, precision-recall, calibration, decision-curve, DeLong/Holm, and related performance summaries;
- use the uncalibrated GBDT base estimator for the primary TreeSHAP interpretation while using calibrated probabilities for probability-based performance evaluation.

The observed fixed-test AUC anchors are approximately:

- GBDT: 0.8362
- LightGBM: 0.8347
- XGBoost: 0.8345
- Random forest: 0.8340
- Decision tree: 0.8278
- MLP: 0.8274
- Logistic regression: 0.8273
- Naive Bayes: 0.8232

These exact empirical values are statistical-analysis anchors; figure display strips may round or use separately documented display values.

## Base logistic-regression coefficient reconstruction

Supplementary Table S15 reports the uncalibrated base logistic-regression coefficients in original measurement units. Run:

```bash
python code/02_primary_analysis/export_primary_lr_coefficients.py --input /path/to/authorized/analysis_ready.csv
```

The script uses the selected primary LR hyperparameters (`C=0.2493730`, L2 penalty, `lbfgs`) and back-transforms coefficients for continuous predictors from the MinMax-scaled fitting space to original measurement units.

## Calibration uncertainty

Run:

```bash
python code/04_calibration/Primary11_bootstrap_calibration_ECE_MCE.py
```

The calibration workflow uses 10 approximately equal-frequency risk groups for ECE/MCE and 1,000 patient-level bootstrap resamples for pointwise calibration intervals.

## Sensitivity and robustness analyses

Repository-facing scripts are under `code/03_sensitivity_analyses/`:

- age-forced 12-predictor analysis;
- HAP-only 11-predictor versus 9-predictor airway-removed analysis;
- 2020-2023 versus 2024-2025 temporal robustness analysis;
- remaining-ICU-stay observation-opportunity analysis;
- upstream reconstruction of the four parsimonious LR comparators;
- downstream parsimonious-comparator fixed-test comparison.

For the parsimonious analysis, run `parsimonious_comparator_training.py` first when an authorized analysis-ready cohort is available, then use `parsimonious_comparator.py` for the reported paired fixed-test comparisons. The four simple specifications are MV + intubation/tracheotomy, NLR alone (`NEU / LYM`), age + sex as a component-level floor, and NEU + LYM + MV. Continuous variables are min-max scaled using training-set parameters only; L2 logistic regression is tuned with 10-fold stratified Bayesian cross-validation and followed by 10-fold Platt calibration in training.

See the README in that directory for required restricted inputs and environment variables.


## Historical exploratory BSA-inclusive reconstruction

Repository-facing historical scripts are under `code/06_historical_bsa_inclusive/`. They reconstruct the archived 10-predictor BSA-inclusive eight-model analysis, the historical LightGBM SHAP figures, and the descriptive BSA-inclusive-versus-current Table S11/S12 comparisons.

These analyses are historical and exploratory. The BSA-inclusive 10-predictor set and the current 11-predictor BSA-free set differ by more than BSA alone, so the comparison is descriptive and must not be interpreted as an isolated incremental-BSA or causal-antibiotic effect.

The public historical scripts preserve the archived workflow while replacing workstation-specific absolute paths. Patient-level data, patient-level predictions, and fitted model objects are not distributed publicly.

## Historical nine-predictor optimism reconstruction (Supplementary Table S14)

Run:

```bash
python code/07_historical_nine_predictor_traceability/historical_9predictor_bootstrap_optimism.py \
  --input /path/to/authorized/analysis_ready.csv \
  --bootstrap 1000
```

This is a reconstruction of a historical traceability analysis, not a recovered copy of the original generation script. It uses the earlier nine-predictor BSA-free specification (NEU, intubation/tracheotomy, MV, LDH, LYM, BUN, CCI, FIB, and surgery) and the selected GBDT/LightGBM hyperparameter values retained in the archived Supplementary Table S2. Hyperparameters are held fixed; each bootstrap-fitted base estimator is evaluated both in its bootstrap sample and in the complete original training cohort, and optimism is the difference in AUC. Mean optimism is subtracted from the original apparent training AUC.

The bundled historical parameter JSON uses continuous values at the precision displayed in the archived table because a full-precision historical `best_params_` export was not located. The script therefore reports archived S14 values only as QA anchors and documents possible small deviations from parameter rounding or library-version differences. S14 is retained for historical traceability only and is not an uncertainty estimate for the current Primary11 models.

## Repeated nested resampling

The repeated nested analysis is intentionally split into R and Python stages under `code/05_repeated_nested_validation/`.

Stage 1 repeats BSA-free LASSO/Boruta feature selection within each of 25 outer training folds (5 repeats x 5 folds). Stage 2 repeats training-only scaling, 5-fold Bayesian hyperparameter optimization, 5-fold Platt calibration, cross-fitted training-only Youden-threshold derivation, and evaluation in the untouched outer validation fold for LR and GBDT.

This analysis evaluates only the recoverable post-preprocessing development pipeline.

## Critical preprocessing limitation

The revision archive retains an already imputed and winsorized analysis-ready matrix. The original pre-imputation and uncapped predictor matrix was not retained in the revision archive. Consequently, the original imputation models and winsorization limits cannot be re-estimated within each repeated outer fold. The public repository must not describe the repeated nested analysis as validation of the complete raw-data preprocessing pipeline.

## Restricted outputs

Do not commit any patient-level input or generated patient-level output to the public repository, including:

- CSV/Excel participant datasets;
- row-level predicted probabilities;
- outer-fold assignments containing participant identifiers;
- serialized fitted models/checkpoints if institutional policy treats them as restricted;
- exact patient dates or source medical-record identifiers.

The repository `.gitignore` is intended to prevent accidental upload of these materials.


## Final reporting reconstruction module

`code/08_reporting_reconstruction/` contains only aggregate, non-patient-level metadata and reconstructs Tables S1, S3, S4, S16, S17, S24, S29, S31, and Figure S1. These items cannot all be derived de novo from the retained analysis-ready matrix because the pre-imputation raw matrix, excluded-patient covariates, exact sampling timestamps, and reliable reason-specific subcounts within the two historical grouped exclusion blocks are unavailable. The repository therefore preserves the archived aggregate revision metadata explicitly rather than inventing missing patient-level history.

The final Table 1 script also emits Supplementary Table S32 separately, implementing the reviewer-requested move of additional first-48-hour intervention/medication candidate variables out of the concise main table.

See `docs/table_figure_code_map.md` for the complete mapping of every reported main and supplementary table/figure to public code.
