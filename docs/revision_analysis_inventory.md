# Revision analysis inventory

## Current public-repository coverage

The repository contains public generating code or transparent reconstruction routes for the main and supplementary tables and figures reported in the current revised manuscript.

The item-level mapping is maintained in:

```text
docs/table_figure_code_map.md
```

The current public modules cover:

- final main Table 1 and the additional first-48-hour intervention/medication variables reported separately in Supplementary Table S32;
- the revised primary feature-selection workflow in which broad-spectrum antibiotic exposure, the all-zero immunosuppressant field, and the nonfunctional standalone heart-disease field are excluded before data-driven selection, followed by the training-set zero-variance rule, leaving 34 variables entering LASSO and Boruta;
- the current locked 11-predictor eight-model primary analysis;
- training-only Bayesian hyperparameter optimization, Platt probability calibration, strict cross-fitted threshold derivation, fixed internal test evaluation, DeLong/Holm comparison, decision-curve analysis, precision-recall analysis, and Primary11 GBDT SHAP interpretation;
- bootstrap calibration assessment, including ECE/MCE summaries and Figure S3;
- age-adjusted Charlson Comorbidity Index decomposition sensitivity;
- restricted-cubic-spline logistic-regression sensitivity;
- cumulative LASSO-ordered Top1-through-Top11 predictor-count analysis;
- HAP-only and airway-removed sensitivity analyses;
- calendar-time robustness analysis;
- remaining-ICU-stay observation-opportunity analysis;
- parsimonious logistic-regression comparator analyses;
- repeated 5 x 5 outer-fold resampling of the recoverable post-preprocessing development pipeline;
- historical BSA-inclusive performance and SHAP analyses retained for traceability;
- historical nine-predictor fixed-hyperparameter bootstrap-optimism reconstruction retained for traceability;
- aggregate reporting reconstruction for historical missingness, screening-flow, score-component availability, predictor timing/coding, sparse-field/VAP audit, and other reporting items that cannot be regenerated from the retained participant-level analysis-ready matrix.

## Current primary feature-selection status

The archived candidate set originally contained 38 fields.

For the current revised primary feature-selection workflow:

1. broad-spectrum antibiotic exposure was excluded before data-driven selection;
2. the all-zero immunosuppressant field was excluded as non-informative;
3. the standalone `heart disease` field was excluded on data-quality grounds because it did not validly represent overall cardiac comorbidity and was distinct from the ICD-10-coded diagnoses used in construction of the Charlson index.

These exclusions left 35 candidate fields.

A training-set zero-variance check subsequently removed cholinesterase-inhibitor exposure, leaving 34 variables entering LASSO and Boruta.

LASSO selected 11 predictors and Boruta confirmed 28. Their intersection contained the same locked 11-predictor Primary11 specification, with no predictors added or removed.

The revised LASSO coefficient-magnitude ordering was also unchanged.

## Current revision-specific sensitivity analyses

The final revision-specific sensitivity-analysis directory is:

```text
code/03_sensitivity_analyses/
```

Current analyses include:

- `cci_age_decomposition_sensitivity.py`
- `rcs_logistic_regression_sensitivity.py`
- `lasso_order_predictor_count_curve.py`
- `hap_only_sensitivity.py`
- `followup_opportunity_sensitivity.py`
- `temporal_robustness.py`
- `parsimonious_comparator_training.py`
- `parsimonious_comparator.py`

The earlier age-forced sensitivity analysis has been superseded by the age-adjusted CCI decomposition analysis and is not part of the final revised analysis.

### Age-adjusted CCI decomposition

The primary `CCI` variable is age-adjusted.

The decomposition sensitivity replaces it with:

- the unadjusted Charlson disease score; and
- chronological age as a separate predictor.

The remaining 10 primary predictors and the fixed training/test identities are retained.

Across the eight algorithms, no AUC difference between the primary age-adjusted-CCI specification and the decomposed specification remained statistically significant after Holm correction.

### Restricted-cubic-spline logistic regression

The RCS sensitivity evaluates nonlinear functional forms for:

- NEU;
- LDH;
- LYM;
- BUN;
- FIB;
- TCO2.

The age-adjusted CCI remains linear and binary predictors retain their original form.

Four training-derived knots are used at the 5th, 35th, 65th, and 95th percentiles.

The analysis does not show a material improvement sufficient to replace the primary linear LR specification.

### Cumulative predictor-count analysis

The LASSO-ordered cumulative analysis enters predictors sequentially in the revised absolute-coefficient order:

1. intubation/tracheotomy;
2. mechanical ventilation;
3. surgery;
4. LYM;
5. NEU;
6. CCI;
7. FIB;
8. diuretics;
9. TCO2;
10. BUN;
11. LDH.

The fixed internal test set is used descriptively to show how AUC changes from Top 1 through Top 11.

This post hoc analysis is not used to select an alternative reduced final model.

## Repeated nested-resampling scope

The repeated nested analysis is implemented under:

```text
code/05_repeated_nested_validation/
```

The directory name is retained for repository continuity, but the analysis should be described in reporting as repeated nested resampling or evaluation of the recoverable post-preprocessing development pipeline.

Across 5 repeats x 5 outer folds, the recoverable workflow repeats outer-training feature selection and downstream model-development operations.

The archived revision input had already undergone the original single-imputation and winsorization procedures before outer resampling.

The original pre-imputation and uncapped predictor matrix was not retained.

Therefore:

- the imputation model cannot be re-estimated within each outer training fold;
- winsorization limits cannot be re-estimated within each outer training fold;
- uncertainty from these preprocessing steps is not propagated into repeated-resampling performance estimates.

Accordingly, this analysis is not full internal validation of the complete raw-data-to-model pipeline.

The repeated nested-resampling run was also completed before the later revision-specific data-quality exclusion of the standalone heart-disease field.

That field was selected in 0/25 reported outer-fold final predictor sets.

The historical nested results are therefore retained as originally generated rather than presented as though every later revision-specific data-quality decision had been rerun retrospectively.

## Fixed internal test-set status

The original 70:30 training/test split is preserved throughout the primary analysis.

The fixed internal test partition contains:

- n = 1,011 participants;
- 407 events.

It remains a standardized within-study comparison set.

However, revision-specific feature re-selection and several subsequent sensitivity analyses were conducted after results from this partition had already been examined.

The fixed test set should therefore not be described as untouched validation of the entire revised workflow.

## Data and privacy boundary

Participant-level clinical data are not distributed publicly.

The public repository should not contain:

- participant-level CSV or Excel datasets;
- hospital record numbers or other identifiers;
- exact patient-level dates;
- row-level predicted probabilities;
- patient-level outer-fold assignments;
- source medical-record exports;
- patient-level SHAP outputs;
- fitted objects or checkpoints if institutional policy treats them as restricted.

The deidentified analysis-ready dataset may be available from the corresponding author on reasonable request, subject to applicable institutional and ethical approvals.

Public scripts are designed to reconstruct the analyses when run against an authorized local copy of the restricted analysis-ready dataset.

## Historical and reconstruction labeling

Files under:

```text
code/06_historical_bsa_inclusive/
code/07_historical_nine_predictor_traceability/
```

are retained for historical traceability.

They are not the current primary model-development specification.

Historical BSA-inclusive comparisons are descriptive because the historical and current predictor sets differ by more than the broad-spectrum-antibiotic field alone.

They must not be interpreted as an isolated incremental effect of broad-spectrum antibiotic exposure or as a causal antibiotic effect.

In historical repository labels, `BSA` refers to broad-spectrum antibiotic exposure, not body surface area.

Aggregate reporting tables that cannot be regenerated from the retained participant-level matrix are reconstructed using version-controlled non-patient-level revision metadata.

Unavailable patient-level history is not inferred or invented.

## Known reproducibility limitations

The revision archive does not retain:

- the complete pre-imputation predictor matrix;
- the uncapped predictor matrix;
- the complete person-level original screening log;
- reliable reason-specific decomposition of the two historical grouped exclusion blocks;
- complete covariates for excluded participants;
- exact first-48-hour laboratory sampling timestamps;
- treatment start/stop times or cumulative exposure duration;
- active-at-landmark treatment status;
- exact death timing relative to the 48-hour landmark or ICU exit.

Available mortality information records in-hospital death only.

ICU discharge or transfer was not modeled as a time-to-event competing process.

These limitations should remain explicit in both manuscript reporting and repository documentation.

## Final release-readiness checks

Before creating the final revised release:

1. confirm that the canonical primary feature-selection script is:
   ```text
   code/01_feature_selection/primary_feature_selection.R
   ```

2. confirm that the canonical primary modeling script is:
   ```text
   code/02_primary_analysis/primary_11_predictor_models_and_shap.py
   ```

3. confirm that the superseded age-forced sensitivity script is not part of the final analysis route;

4. confirm that the final sensitivity-analysis directory contains the CCI decomposition, RCS-LR, and LASSO-ordered cumulative predictor-count scripts;

5. confirm that the repeated nested-resampling README explicitly distinguishes the recoverable post-preprocessing pipeline from the complete raw-data pipeline;

6. confirm that:
   ```text
   README.md
   docs/model_reconstruction.md
   docs/table_figure_code_map.md
   docs/revision_analysis_inventory.md
   CITATION.cff
   environment.yml
   requirements.txt
   ```
   are synchronized with the final revised analysis;

7. verify that the public repository contains no patient-level datasets, row-level predictions, patient identifiers, exact patient-level dates, restricted fitted objects, cache directories, or workstation-specific absolute paths;

8. run the repository-facing R scripts under the documented R environment where feasible to confirm that the public versions execute as expected;

9. retain the existing `v1.0.0` historical release and create a new final revision release rather than overwriting or deleting the previous release;

10. create the final release as:
    ```text
    v1.1.0
    ```

11. after all final repository commits are complete, record the exact commit SHA corresponding to `v1.1.0`;

12. insert the final repository URL, release tag, and exact commit SHA into the manuscript Code Availability statement and reviewer response.

An archival DOI or other persistent identifier may additionally be created after the final versioned release is frozen, but it should not be invented or cited before it actually exists.
