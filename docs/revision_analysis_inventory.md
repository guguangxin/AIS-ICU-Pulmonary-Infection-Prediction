# Revision analysis inventory

## Current public-repository coverage

The repository now contains public code or reconstruction documentation for the main current and revision-specific analysis modules:

- Table 1 descriptive/statistical reconstruction;
- BSA-free LASSO + Boruta primary feature selection;
- current 11-predictor eight-model primary analysis;
- strict cross-fitted threshold derivation within the primary workflow;
- Platt calibration and primary performance evaluation;
- focused decision-curve analysis within the primary workflow;
- current-primary GBDT SHAP interpretation within the primary workflow;
- Primary11 bootstrap calibration / ECE / MCE / Supplementary Figure S3 workflow;
- AgeForced12 sensitivity analysis;
- HAP-only 11-predictor versus 9-predictor airway-removed sensitivity analysis;
- calendar-time robustness analysis (2020-2023 development versus 2024-2025 later cohort);
- remaining-ICU-stay / observation-opportunity sensitivity analysis;
- upstream fitting and downstream fixed-test comparison for the four parsimonious LR comparators;
- repeated 5 x 5 outer nested validation of the recoverable post-preprocessing pipeline;
- current 11-predictor base logistic-regression coefficient reconstruction for Supplementary Table S15;
- historical exploratory BSA-inclusive 10-predictor eight-model reconstruction for Supplementary Figures S4-S5;
- historical exploratory BSA-inclusive LightGBM SHAP reconstruction for Supplementary Figures S2 and S6;
- historical-versus-current descriptive reconstruction for Supplementary Tables S11-S12.

## Historical BSA-inclusive scope

The BSA-inclusive analyses are retained as historical exploratory analyses. Their predictor composition differs from the current 11-predictor primary analysis by more than BSA alone, so Tables S11-S12 are descriptive model-set contrasts rather than isolated tests of the incremental value of BSA. The public scripts preserve the archived historical workflow and replace workstation-specific absolute paths with portable local paths. Patient-level inputs and outputs are not distributed.

## Parsimonious comparator reconstruction

The public repository now includes both stages of the parsimonious analysis:

1. `parsimonious_comparator_training.py` reconstructs the four simple LR specifications from an authorized analysis-ready cohort using the fixed original split, training-derived min-max scaling for continuous variables, 10-fold stratified Bayesian tuning of L2 logistic regression, and 10-fold Platt calibration.
2. `parsimonious_comparator.py` reproduces fixed-test AUC/AP/Brier summaries, paired bootstrap differences, paired DeLong tests, and Holm adjustment from the locked comparator prediction file.

The four specifications are MV + intubation/tracheotomy, NLR alone (NEU/LYM), age + sex as a component-level floor, and NEU + LYM + MV. The age-sex model is not presented as a reconstructed A2DS2 or ISAN score.

## Remaining traceability item before the final archived release

### Historical nine-predictor optimism analysis (Supplementary Table S14)

Supplementary Table S14 retains a historical fixed-hyperparameter bootstrap optimism analysis for the earlier nine-predictor GBDT and LightGBM models. It is explicitly retained only for analysis traceability and is not used as the uncertainty estimate for the current 11-predictor primary analysis.

Before the final GitHub Release / Zenodo archive, do one of the following:

1. add the archived S14 optimism script if it can be recovered;
2. add a clearly labeled reconstruction script that reproduces the historical procedure from the authorized archive; or
3. remove S14 from the Supplementary Material if the historical traceability analysis is no longer needed.

Do not imply that a newly written reconstruction script is the original historical generation code unless that can be verified.

## Release rule

Do not create the final GitHub Release or Zenodo archive until the S14 traceability item above has been resolved or explicitly removed from the reported Supplementary Material.
