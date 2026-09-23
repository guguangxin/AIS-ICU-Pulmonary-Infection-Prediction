# Repeated nested resampling (5 x 5 outer folds)

This folder contains the two-stage implementation used for revision-specific repeated nested resampling of the **recoverable post-preprocessing development pipeline**.

## Stage 1: feature selection in each outer training fold

Run:

```bash
Rscript 01_repeated_nested_feature_selection.R \
  --input=/path/to/restricted/analysis_ready.csv \
  --output-dir=outputs/repeated_nested
```

Stage 1 creates:

- `00_outer_fold_assignments.csv`
- `01_nested_outer_fold_feature_selection.csv`
- `02_nested_feature_selection_frequency.csv`

Within each of the 25 outer training folds, BSA is excluded before selection, zero-variance fields are removed, LASSO is rerun with 10-fold CV and `lambda.1se`, and Boruta is rerun with 1,000 trees, `maxRuns=100`, followed by `TentativeRoughFix`. The final feature rule is the LASSO-Boruta intersection, with the union used only if the intersection contains fewer than three variables.

## Stage 2: LR and GBDT outer-fold evaluation

Run:

```bash
python 02_repeated_nested_model_validation.py \
  --input /path/to/restricted/analysis_ready.csv \
  --stage1-dir outputs/repeated_nested \
  --output-dir outputs/repeated_nested
```

Stage 2 repeats training-only scaling, inner 5-fold Bayesian hyperparameter tuning, 5-fold Platt calibration, cross-fitted training-only Youden-threshold derivation, and untouched outer-fold evaluation for LR and GBDT.

## Critical limitation

The archived revision input is an analysis-ready matrix that had already undergone the original single-imputation and winsorization procedures before outer resampling. The original pre-imputation and uncapped predictor matrix was not retained in the revision archive. Therefore, neither the imputation model nor the winsorization limits could be re-estimated within each outer training fold, and uncertainty from these preprocessing steps is not propagated into the repeated-resampling performance estimates.

Within each outer training fold, zero-variance screening, LASSO/Boruta feature selection, scaling, Bayesian hyperparameter tuning, Platt calibration, and cross-fitted training-only Youden-threshold derivation were repeated before evaluation in the untouched outer fold.

Accordingly, this analysis evaluates the recoverable **post-preprocessing development pipeline** rather than the complete raw-data-to-model pipeline and should not be interpreted as full internal validation of all preprocessing and modeling steps.

## Relation to the revised primary feature-selection analysis

This repeated nested-resampling analysis was completed before the later revision-specific data-quality exclusion of the standalone heart-disease field from the primary candidate pool. The field was not retained in the reported outer-fold final predictor sets. The primary feature-selection analysis was subsequently rerun after excluding this field and recovered the same locked 11-predictor specification.

The nested-resampling results are therefore retained as originally generated and are reported as a robustness analysis of the recoverable post-preprocessing pipeline rather than as a complete rerun of every later revision-specific data-quality decision.
## Public-repository data note

The scripts are public. Patient-level input data and generated patient-level outer-fold assignments/predictions are not included in the public repository and remain subject to institutional and ethical data-access restrictions.
