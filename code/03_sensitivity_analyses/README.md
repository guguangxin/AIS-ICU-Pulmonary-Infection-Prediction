# Sensitivity and robustness analyses

This directory contains repository-facing copies of the revision-specific sensitivity analyses for the current 11-predictor primary analysis.

## Scripts

- `age_forced_sensitivity.py` — compares the current Primary11 models with an AgeForced12 specification on the locked fixed test partition.
- `hap_only_sensitivity.py` — excludes all adjudicated VAP cases and compares the 11-predictor HAP-only models with the corresponding 9-predictor models after removal of mechanical ventilation and intubation/tracheotomy.
- `temporal_robustness.py` — development-period (2020-2023) versus later-period (2024-2025) temporal robustness analysis using the eight primary algorithms.
- `followup_opportunity_sensitivity.py` — evaluates discrimination and calibration across recorded remaining-ICU-stay strata using locked Primary11 GBDT and LR probabilities; models are not refitted within strata.
- `parsimonious_comparator_training.py` — reconstructs the upstream fitting of the four simple logistic-regression comparator specifications from the governed analysis-ready cohort.
- `parsimonious_comparator.py` — reproduces the downstream fixed-test performance comparisons, paired bootstrap differences, and paired DeLong/Holm tests from the locked comparator prediction file.

## Parsimonious comparator reconstruction

The four simple LR specifications are:

1. mechanical ventilation + intubation/tracheotomy;
2. NLR alone (`NEU / LYM`);
3. age + sex as a component-level floor (not a reconstructed A2DS2 or ISAN score);
4. NEU + LYM + mechanical ventilation.

The upstream reconstruction preserves the fixed original internal split, applies training-derived min-max scaling to continuous variables, tunes L2 logistic regression by 10-fold stratified Bayesian cross-validation, and applies 10-fold Platt calibration in training. The resulting fixed-test probabilities can then be passed to `parsimonious_comparator.py` for the reported paired comparisons.

## Restricted data

Patient-level data and patient-level prediction files are **not** included in the public repository. By default, scripts look for restricted inputs under:

```text
<repository>/restricted_data/
```

You may instead set:

```text
AIS_ICU_DATA_DIR=/path/to/restricted_data
AIS_ICU_OUTPUT_DIR=/path/to/analysis_outputs
```

Additional optional variables are supported by individual scripts, including `AIS_ICU_EXACT_SPLIT_FILE`, `AIS_ICU_PRIMARY11_OUTPUT_DIR`, `AIS_ICU_TEMPORAL_INPUT_FILE`, and `AIS_ICU_PARSIMONIOUS_PREDICTIONS`.

Generated patient-level CSV files, serialized model objects, checkpoints, and result folders should remain outside version control and are intended to be covered by the repository `.gitignore`.

## Reproducibility scope

These public copies preserve the statistical/modeling logic of the revision analyses while replacing workstation-specific absolute paths with portable restricted-data and output locations. Reconstructed outputs may differ slightly across software versions because Bayesian optimization, calibration, and stochastic estimators can be version-sensitive; archived point estimates in the manuscript remain the reporting reference.
