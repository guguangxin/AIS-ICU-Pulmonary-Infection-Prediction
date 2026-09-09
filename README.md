# AIS-ICU-Pulmonary-Infection-Prediction

Versioned public code and reproducibility documentation for the development and internal evaluation of machine-learning models predicting a later-documented ICU-acquired hospital-acquired pneumonia/ventilator-associated pneumonia (HAP/VAP) event after a 48-hour landmark and before ICU exit in adults with acute ischemic stroke.

> **Research use only.** This repository documents an internal model-development study and is not a clinical decision-support system. Prospective external validation and recalibration are required before clinical use.

## Current primary specification

The current primary analysis excludes early broad-spectrum antibiotic (BSA) exposure **before** feature selection. The BSA-free LASSO-Boruta intersection contains 11 predictors:

1. absolute neutrophil count (NEU)
2. endotracheal intubation/tracheotomy
3. mechanical ventilation (MV)
4. lactate dehydrogenase (LDH)
5. absolute lymphocyte count (LYM)
6. blood urea nitrogen (BUN)
7. Charlson Comorbidity Index (CCI)
8. fibrinogen (FIB)
9. surgery
10. diuretics
11. total carbon dioxide (TCO2)

Eight algorithms are evaluated: logistic regression, naive Bayes, decision tree, random forest, gradient boosting decision tree, XGBoost, LightGBM, and multilayer perceptron.

## Repository structure

```text
code/
  00_table1/
  01_feature_selection/
  02_primary_analysis/
  03_sensitivity_analyses/
  04_calibration/
  05_repeated_nested_validation/
  06_historical_bsa_inclusive/
  07_historical_nine_predictor_traceability/
  08_reporting_reconstruction/
data/
  README.md
docs/
  data_and_feature_definitions.md
  reproducibility_notes.md
  model_reconstruction.md
  revision_analysis_inventory.md
  table_figure_code_map.md
CITATION.cff
environment.yml
requirements.txt
LICENSE
```

The complete mapping from each reported main/supplementary table and figure to its public generation/reconstruction route is in [`docs/table_figure_code_map.md`](docs/table_figure_code_map.md).

## Data governance

Participant-level clinical data are **not** distributed in this repository. The deidentified analysis-ready dataset may be available from the corresponding author on reasonable request, subject to institutional and ethical requirements. The public repository must not contain patient identifiers, hospital record numbers, exact patient-level dates, row-level prediction files, or source medical-record exports.

Most current analyses expect an authorized local CSV with:

- `Study_row_id`
- `Primary_split`
- `Pulmonary_infection`
- the candidate/predictor fields documented in `docs/data_and_feature_definitions.md`

Set the main restricted input path with `AIS_ICU_DATA_FILE` where supported, or use the script-specific command-line input described in each module README.

## Environment

A machine-readable Conda environment is provided in `environment.yml`. The revision record used Python 3.13.5 and R 4.5.1, including:

- scikit-learn 1.6.1
- scikit-optimize 0.10.2
- LightGBM 4.6.0
- XGBoost 3.2.0
- SHAP 0.51.0
- statsmodels 0.14.4
- glmnet 4.1-10
- Boruta 9.0.0
- ranger 0.17.0

Create the environment with:

```bash
conda env create -f environment.yml
conda activate ais-icu-pulmonary-infection
```

## Reconstructing the current analysis

Primary feature selection:

```bash
Rscript code/01_feature_selection/primary_feature_selection.R
```

Primary eight-model analysis, strict cross-fitted threshold derivation, DCA, calibration panels, and GBDT SHAP:

```bash
python code/02_primary_analysis/primary_11_predictor_models_and_shap.py
```

Additional modules reproduce calibration uncertainty, HAP-only and age-forced analyses, temporal robustness, observation-opportunity analyses, parsimonious comparators, repeated nested resampling, historical BSA-inclusive traceability, and historical nine-predictor optimism reconstruction. See `docs/model_reconstruction.md`.

Aggregate reporting/audit items that cannot be regenerated from the retained analysis-ready matrix are reconstructed transparently from non-patient-level revision metadata:

```bash
python code/08_reporting_reconstruction/reconstruct_reporting_tables_and_flowchart.py
```

## Reproducibility boundary

The revision archive retains the completed analysis-ready participant matrix but not the original pre-imputation/raw uncapped matrix. Consequently, the repeated nested resampling analysis can re-estimate BSA-free feature selection, scaling, tuning, calibration, and thresholding within outer folds, but it cannot re-estimate the original imputation models or winsorization limits. The repository therefore describes that analysis as validation of the **recoverable post-preprocessing pipeline**, not full raw-data pipeline validation.

Similarly, historical missingness, grouped exclusion-flow detail, exact predictor timestamps, and selected historical summaries are retained only as aggregate revision metadata; no unavailable patient-level information is invented.

## Historical analyses

`code/06_historical_bsa_inclusive/` preserves the exploratory BSA-inclusive 10-predictor analysis used for historical supplementary figures and descriptive comparisons. Because that historical predictor set differs from the current 11-predictor set by more than BSA alone, the comparison must not be interpreted as an isolated causal or incremental BSA effect.

`code/07_historical_nine_predictor_traceability/` reconstructs the earlier fixed-hyperparameter bootstrap optimism analysis retained as Supplementary Table S14 solely for traceability.

## Citation and release

Citation metadata are provided in `CITATION.cff`. The immutable archival DOI/persistent identifier should be added after the versioned GitHub release is archived.

## License

Code is released under the MIT License. See `LICENSE`.
