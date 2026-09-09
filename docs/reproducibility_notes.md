# Reproducibility notes

## Prediction target

The operational target is the probability of a later-documented HAP/VAP event after the 48-hour landmark and before ICU exit. It is not interpreted as a fixed 7-day or 14-day incidence probability.

## Primary feature selection

BSA was excluded before current feature selection. Immunosuppressant use was excluded because it was all zero in the cohort. A training-set zero-variance rule removed non-informative fields before LASSO/Boruta. LASSO used 10-fold cross-validation and `lambda.1se`; Boruta used 1,000 trees, `maxRuns=100`, and `TentativeRoughFix`. The primary rule used the LASSO-Boruta intersection.

## Primary model development

The primary script retains the fixed original Train/Test identities. Hyperparameter optimization and calibration are confined to training data. Model-specific operating thresholds are obtained from strict cross-fitted calibrated training probabilities and then locked before application to the fixed internal test set.

## Important revision limitation

The revision archive retains the already imputed and winsorized analysis-ready data but not the original pre-imputation/raw uncapped matrix. Therefore, repeated nested validation of the revision pipeline cannot reproduce imputation-model fitting and winsorization-limit estimation inside every outer fold. This limitation should remain explicit in manuscript and repository documentation.
