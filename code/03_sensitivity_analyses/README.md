# Sensitivity and robustness analyses

This directory contains repository-facing copies of revision-specific sensitivity and robustness analyses for the current 11-predictor primary analysis.

These analyses were performed to address reviewer questions regarding model parsimony, nonlinear functional form, the age-adjusted Charlson Comorbidity Index (CCI), pneumonia subtype, unequal observation opportunity, temporal separation around the 48-hour landmark, and calendar-time robustness. Unless otherwise stated, they are post hoc analyses and were not used to redefine the locked primary predictor set on the basis of fixed-test performance.

## Scripts

### `cci_age_decomposition_sensitivity.py`

Evaluates the overlap between chronological age and the age-adjusted CCI used in the primary analysis.

The primary age-adjusted CCI is replaced by:

1. the unadjusted Charlson comorbidity score; and
2. chronological age as a separate predictor.

The remaining 10 primary predictors, the original fixed training/test identities, and the modeling/evaluation framework are retained.

For the three participants whose archived CCI values had previously been imputed, the unadjusted Charlson score is reconstructed by subtracting the deterministic age component from the imputed age-adjusted CCI without rounding.

This analysis supersedes the earlier age-forced sensitivity analysis. It is intended to assess whether separating age from comorbidity materially changes predictive performance rather than to estimate a causal or independent biological effect of age.

---

### `rcs_logistic_regression_sensitivity.py`

Evaluates whether nonlinear functional forms materially improve the primary logistic-regression model.

Restricted cubic spline terms are applied to the six continuous laboratory predictors:

- absolute neutrophil count (NEU);
- lactate dehydrogenase (LDH);
- absolute lymphocyte count (LYM);
- blood urea nitrogen (BUN);
- fibrinogen (FIB); and
- total carbon dioxide (TCO2).

The age-adjusted CCI is retained as a linear term, and the four binary predictors are retained unchanged.

Four spline knots are estimated from the training cohort only at the 5th, 35th, 65th, and 95th percentiles. Hyperparameter tuning and Platt calibration are performed using training data only. The locked fixed test set is used only for final comparison with the primary linear LR model.

This analysis addresses whether misspecified linear functional form could explain the calibration difference between LR and the leading tree-based models.

---

### `lasso_order_predictor_count_curve.py`

Performs a descriptive cumulative-predictor analysis using the predictor order defined by the revised R/glmnet LASSO solution.

Predictors are ordered by decreasing absolute coefficient magnitude at `lambda.1se` in the fixed training cohort and entered cumulatively into logistic-regression models from Top 1 through Top 11.

The final revised order is:

1. intubation/tracheotomy;
2. mechanical ventilation;
3. surgery;
4. LYM;
5. NEU;
6. CCI;
7. FIB;
8. diuretics;
9. TCO2;
10. BUN; and
11. LDH.

The fixed internal test set is used to describe how test AUC changes as predictors are added. This analysis is post hoc and descriptive and was not used to select an alternative reduced predictor set.

The LASSO coefficient ordering remained unchanged after exclusion of the standalone heart-disease field on data-quality grounds.

---

### `hap_only_sensitivity.py`

Excludes all adjudicated ventilator-associated pneumonia (VAP) cases and evaluates prediction of non-VAP hospital-acquired pneumonia (HAP).

The analysis compares:

- the current 11-predictor specification; and
- a corresponding 9-predictor specification after removal of mechanical ventilation and intubation/tracheotomy.

This analysis evaluates whether the predictive signal is explained primarily by early airway-support variables or remains present for non-VAP HAP.

---

### `followup_opportunity_sensitivity.py`

Evaluates discrimination and calibration across strata of recorded remaining ICU stay after the 48-hour prediction landmark using locked Primary11 GBDT and LR probabilities.

Models are not refitted within strata.

The analysis is descriptive because remaining ICU stay may itself be prolonged by pulmonary infection and therefore should not be interpreted as an exogenous follow-up duration.

Available mortality information records in-hospital death but does not establish exact post-landmark death timing, and ICU discharge/transfer was not modeled as a time-to-event competing process.

---

### `day3_temporal_sensitivity_reproducible.py`

Performs the hour-level temporal-separation audit and ICU-day-3 exclusion sensitivity analysis reported in Supplementary Table S35.

The script combines the locked analysis-ready modeling matrix with the archived timing dataset and verifies that qualifying pulmonary-infection events in the final analytic cohort occurred strictly after the 48-hour landmark and before ICU exit.

Qualifying events are summarized according to elapsed time from ICU admission as:

- ICU day 3: >48 to <72 hours;
- ICU day 4: 72 to <96 hours; and
- ICU day 5+: >=96 hours.

The script also summarizes the number of pulmonary-infection events among patients with exactly 0 recorded remaining ICU days and among those with 0-1 recorded remaining ICU days. The archived `ICU_LOS_days` variable is integer day-based, so 0 recorded remaining days does not imply zero elapsed post-landmark observation time.

For the reviewer-requested ICU-day-3 exclusion sensitivity analysis, patients with a qualifying positive event during >48 to <72 hours are removed from both the original training and fixed test partitions. The locked 11-predictor LR and GBDT specifications and previously selected hyperparameters are retained. Continuous-variable scaling, model fitting, and 10-fold Platt calibration are then refitted using the reduced training partition before evaluation in the corresponding reduced fixed test partition.

The script additionally performs a locked-prediction subset check in which the original primary model predictions are retained and ICU-day-3-positive patients are removed only from the fixed test evaluation subset.

Reported outputs include event-timing summaries, AUC with bootstrap confidence intervals, average precision, Brier score, calibration intercept, and calibration slope.

This analysis addresses temporal proximity between the first-48-hour predictor window and early post-landmark pneumonia documentation. It does not establish the exact biological onset of infection or exclude incipient infection already developing at the 48-hour landmark.

---

### `temporal_robustness.py`

Performs a calendar-time robustness analysis using:

- 2020-2023 as the development period; and
- 2024-2025 as the later-period evaluation cohort.

The analysis uses the eight primary algorithms and the current 11-predictor specification.

Because the archived matrix had already undergone the original imputation and winsorization procedures and the predictor-selection history preceded the chronological split, this analysis is interpreted as a post hoc temporal robustness analysis rather than independent temporal validation of the complete raw-data-to-model pipeline.

---

### `parsimonious_comparator_training.py`

Reconstructs the upstream fitting of four simple logistic-regression comparator specifications from the governed analysis-ready cohort.

The four comparator specifications are:

1. mechanical ventilation + intubation/tracheotomy;
2. neutrophil-to-lymphocyte ratio (NLR) alone;
3. age + sex as a component-level floor, not a reconstructed A2DS2 or ISAN score; and
4. NEU + LYM + mechanical ventilation.

The reconstruction preserves the fixed original internal split, applies training-derived min-max scaling to continuous variables, tunes L2 logistic regression by 10-fold stratified Bayesian cross-validation, and applies 10-fold Platt calibration in training.

---

### `parsimonious_comparator.py`

Performs the downstream fixed-test comparison between the current 11-predictor LR model and the four parsimonious comparator models.

Reported comparisons include fixed-test performance, paired bootstrap differences, and paired DeLong tests with Holm adjustment where applicable.

These simple comparators do not represent established stroke-pneumonia scores. A2DS2 and ISAN could not be validly reconstructed because several required components were unavailable in standardized analyzable form.

## Restricted data

Patient-level clinical data and patient-level prediction files are not included in the public repository.

By default, scripts look for restricted inputs under:

```text
<repository>/restricted_data/
```
