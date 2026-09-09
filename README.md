# AIS-ICU-Pulmonary-Infection-Prediction

Code and reproducible analysis workflow for the development and internal evaluation of machine-learning models predicting later-documented hospital-acquired pneumonia (HAP) or ventilator-associated pneumonia (VAP) during the remaining ICU stay after a 48-hour landmark in patients with acute ischemic stroke (AIS).

## Study overview

This repository accompanies a single-center retrospective prediction study conducted in a neurology intensive care unit.

The analytic cohort included 3,368 eligible patients with AIS. The primary outcome was later-documented ICU-acquired pulmonary infection after the first 48 hours of ICU admission and before ICU exit, operationalized as non-VAP HAP or VAP.

The current primary analysis uses 11 predictors selected after excluding early broad-spectrum antibiotic exposure before feature selection.

## Primary predictors

The 11 predictors used in the current primary models are:

1. Absolute neutrophil count (NEU)
2. Endotracheal intubation/tracheotomy
3. Mechanical ventilation (MV)
4. Lactate dehydrogenase (LDH)
5. Absolute lymphocyte count (LYM)
6. Blood urea nitrogen (BUN)
7. Charlson Comorbidity Index (CCI)
8. Fibrinogen (FIB)
9. Surgery
10. Diuretics
11. Total carbon dioxide (TCO2)

## Models

Eight algorithms were evaluated:

- Logistic regression
- Naive Bayes
- Decision tree
- Random forest
- Gradient boosting decision tree
- XGBoost
- LightGBM
- Multilayer perceptron

The 11-predictor logistic regression model is retained as the preferred transparent candidate for prospective external validation. Tree-based and neural-network models are retained for comparative performance assessment and model-behavior exploration.

## Repository structure

```text
.
├── code/
│   └── analysis scripts
├── data/
│   └── README describing data-access restrictions
├── docs/
│   ├── data and feature definitions
│   ├── reproducibility notes
│   └── revision analysis inventory
├── environment.yml
├── requirements.txt
├── LICENSE
└── README.md
