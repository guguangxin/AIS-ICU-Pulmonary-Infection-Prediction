# Repeated nested validation (5 x 5 outer resampling)

This folder contains the two-stage implementation used to document the revision-specific repeated nested validation of the **recoverable post-preprocessing development pipeline**.

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

## Stage 2: LR and GBDT model validation

Run:

```bash
python 02_repeated_nested_model_validation.py \
  --input /path/to/restricted/analysis_ready.csv \
  --stage1-dir outputs/repeated_nested \
  --output-dir outputs/repeated_nested
```

Stage 2 repeats training-only scaling, inner 5-fold Bayesian hyperparameter tuning, 5-fold Platt calibration, cross-fitted training-only Youden-threshold derivation, and untouched outer-fold evaluation for LR and GBDT.

## Critical limitation

The archived revision input is already imputed and winsorized under the original preprocessing workflow. The original pre-imputation and uncapped predictor matrix was not retained in the revision archive. Therefore the imputation model and winsorization limits cannot be re-estimated within each outer training fold. This analysis validates the recoverable **post-preprocessing** development pipeline, not the complete raw-data pipeline.

## Public-repository data note

The scripts are public. Patient-level input data and generated patient-level outer-fold assignments/predictions are not included in the public repository and remain subject to institutional and ethical data-access restrictions.
