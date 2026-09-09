# Revision analysis inventory

## Current public-repository coverage

The repository now contains public code or reconstruction documentation for the main current analysis modules:

- Table 1 descriptive/statistical reconstruction;
- BSA-free LASSO + Boruta primary feature selection;
- current 11-predictor eight-model primary analysis;
- strict cross-fitted threshold derivation within the primary workflow;
- Platt calibration and primary performance evaluation;
- focused decision-curve analysis within the primary workflow;
- GBDT SHAP interpretation within the primary workflow;
- Primary11 bootstrap calibration / ECE / MCE / Supplementary Figure S3 workflow;
- AgeForced12 sensitivity analysis;
- HAP-only 11-predictor versus 9-predictor airway-removed sensitivity analysis;
- calendar-time robustness analysis (2020-2023 development versus 2024-2025 later cohort);
- remaining-ICU-stay / observation-opportunity sensitivity analysis;
- downstream parsimonious-comparator fixed-test comparison;
- repeated 5 x 5 outer nested validation of the recoverable post-preprocessing pipeline;
- current 11-predictor base logistic-regression coefficient reconstruction for Supplementary Table S15.

## Items that remain open before the final archived release

### 1. Historical exploratory BSA-inclusive analyses

The current Supplementary Material retains historical exploratory BSA-inclusive results, including Tables S11-S12 and Figures S2 and S4-S6. The original generation script for those historical figures/tables has not yet been identified in the current public package.

Before the final GitHub Release / Zenodo archive, do one of the following:

1. add the archived BSA-inclusive analysis/figure-generation script if it can be located; or
2. add a clearly labeled reconstruction script that has been rerun against the authorized archived data and checked against the reported historical results; or
3. remove the historical BSA-inclusive outputs from the manuscript/supplement if they are no longer required.

Do not claim that newly written reconstruction code was the original code used if that cannot be verified.

### 2. Upstream fitting of the four parsimonious LR comparators

The public `parsimonious_comparator.py` reproduces the downstream fixed-test comparisons from the locked patient-level comparator-prediction file. That patient-level file is restricted and is not included publicly. The upstream fitting code that originally generated the four simple comparator probabilities is not currently present in the public package.

Before the final archived release, preferably add/reconstruct and verify the upstream fitting workflow for:

- MV + intubation/tracheotomy;
- NLR alone (`NEU / LYM`);
- age + sex component-level floor;
- NEU + LYM + MV.

The manuscript specifies training-only scaling of continuous variables, L2 logistic regression with 10-fold stratified tuning, 10-fold Platt calibration, and evaluation on the same fixed 1,011-patient internal test partition.

### 3. Historical nine-predictor optimism traceability (Supplementary Table S14)

Supplementary Table S14 retains a historical fixed-hyperparameter bootstrap optimism analysis for GBDT and LightGBM. If the original script is available, add it to a clearly labeled historical/traceability directory. If it cannot be recovered, document that limitation explicitly and avoid implying that the current repository contains original generation code for S14.

## Release rule

Do not create the final GitHub Release or Zenodo archive until the three open traceability items above have been resolved or explicitly removed from the reported manuscript/supplementary outputs.
