# AIS-ICU-Pulmonary-Infection-Prediction

Versioned public code and reproducibility documentation for the development and internal evaluation of machine-learning models predicting later-documented ICU-acquired hospital-acquired pneumonia/ventilator-associated pneumonia (HAP/VAP) after a 48-hour landmark and before ICU exit in adults with acute ischemic stroke.

> **Research use only.** This repository documents an internal model-development study and is not a clinical decision-support system. Prospective external validation and recalibration are required before clinical use.

## Current primary specification

The current primary analysis uses a prespecified and data-quality-refined candidate pool before feature selection.

Early broad-spectrum antibiotic (BSA) exposure was excluded before feature selection because exposure was recorded during the same first-48-hour window used for prediction and was considered particularly susceptible to treatment-response behavior, confounding by indication, and protopathic bias.

The all-zero immunosuppressant field was excluded as non-informative.

The standalone `heart disease` field was excluded on data-quality grounds because it did not validly represent overall cardiac comorbidity. This field was distinct from the ICD-10-coded diagnoses used to construct the cardiac components of the Charlson Comorbidity Index.

After these exclusions, 35 candidate fields remained. A training-set zero-variance check subsequently removed cholinesterase-inhibitor exposure, leaving 34 variables entering LASSO and Boruta.

The revised LASSO-Boruta intersection recovered the same locked 11-predictor specification:

1. absolute neutrophil count (NEU)
2. endotracheal intubation/tracheotomy
3. mechanical ventilation (MV)
4. lactate dehydrogenase (LDH)
5. absolute lymphocyte count (LYM)
6. blood urea nitrogen (BUN)
7. age-adjusted Charlson Comorbidity Index (CCI)
8. fibrinogen (FIB)
9. surgery
10. diuretics
11. total carbon dioxide (TCO2)

The revised 34-candidate feature-selection analysis recovered exactly the same 11-predictor specification as the locked primary model.

Eight algorithms are evaluated:

- logistic regression
- naive Bayes
- decision tree
- random forest
- gradient boosting decision tree
- XGBoost
- LightGBM
- multilayer perceptron

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
