# Reproducibility notes

## Prediction target

The operational prediction target is a later-documented hospital-acquired pneumonia (HAP) or ventilator-associated pneumonia (VAP) event occurring more than 48 hours after ICU admission and before ICU exit.

The target therefore has a variable post-landmark observation horizon rather than a fixed 7-day or 14-day cumulative-incidence horizon.

The available data do not support a time-to-event competing-risk analysis. The mortality variable records in-hospital death only and does not establish the timing of death relative to the 48-hour landmark or ICU exit. ICU discharge or transfer was not modeled as a time-to-event competing process.

## Fixed training/test split

The original 70:30 training/test identities are retained through the current primary analysis:

- training set: n = 2,357; events = 950;
- fixed internal test set: n = 1,011; events = 407.

The fixed test partition is used as a standardized within-study comparison set.

Because revision-specific feature re-selection and several subsequent sensitivity analyses were conducted after results from this partition had already been examined, it should not be described as untouched validation of the entire revised workflow.

## Primary feature selection

The archived candidate set originally contained 38 fields.

Before the revised data-driven feature-selection step:

- early broad-spectrum antibiotic exposure was excluded because it was recorded during the same first-48-hour prediction window and was particularly susceptible to treatment-response behavior, confounding by indication, and protopathic bias;
- the all-zero immunosuppressant field was excluded as non-informative;
- the standalone `heart disease` field was excluded on data-quality grounds because it did not validly represent overall cardiac comorbidity and was distinct from the ICD-10-coded diagnoses used in construction of the Charlson index.

These exclusions left 35 candidate fields.

A training-set zero-variance rule subsequently removed cholinesterase-inhibitor exposure, leaving 34 variables entering LASSO and Boruta.

LASSO used binomial regression with `alpha = 1`, 10-fold cross-validation, and `lambda.1se`.

Boruta used 1,000 trees, `maxRuns = 100`, followed by `TentativeRoughFix`.

The primary feature-selection rule used the LASSO-Boruta intersection.

The revised analysis recovered exactly the same locked 11-predictor Primary11 specification.

## Primary model development

The primary modeling script retains the fixed original training/test identities.

Hyperparameter optimization and probability calibration are confined to training data.

Continuous-variable scaling parameters are fitted in the training data and applied unchanged to the fixed test partition.

Model-specific operating thresholds are derived from strict cross-fitted calibrated training probabilities and are locked before application to the fixed internal test set.

The fixed test set is then used for standardized within-study comparison of discrimination, calibration, decision-curve quantities, and related performance measures.

## Probability calibration and interpretation

Probability calibration uses training-based sigmoid/Platt calibration.

Calibration performance is evaluated separately from discrimination.

The logistic-regression model is retained as a transparent equation-based benchmark rather than designated as an overall preferred model.

GBDT has the highest observed fixed-test AUC and more favorable calibration than LR in the current primary comparison.

Pairwise discrimination differences among the leading algorithms are interpreted with multiplicity-adjusted testing.

## Repeated nested-resampling scope

The repeated nested analysis is implemented under:

```text
code/05_repeated_nested_validation/
