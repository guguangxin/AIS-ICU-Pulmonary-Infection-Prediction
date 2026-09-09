# Revision analysis inventory

## Current public-repository coverage

The repository now contains public generating code or transparent reconstruction code for **all currently reported main and supplementary tables and figures**. The item-level map is `docs/table_figure_code_map.md`.

Core modules cover:

- final main Table 1 plus the reviewer-requested Supplementary Table S32 split;
- BSA-free LASSO + Boruta primary feature selection;
- current 11-predictor eight-model primary analysis;
- strict cross-fitted threshold derivation, Platt calibration, fixed-test performance, DeLong/Holm comparison, DCA, PR analysis, and Primary11 GBDT SHAP;
- bootstrap calibration uncertainty/ECE/MCE and Figure S3;
- age-forced, HAP-only airway-removed, calendar-time, remaining-ICU-stay, and parsimonious-comparator analyses;
- repeated 5 x 5 nested validation of the recoverable post-preprocessing pipeline;
- historical BSA-inclusive 10-predictor performance and SHAP traceability;
- historical nine-predictor fixed-hyperparameter bootstrap-optimism traceability;
- aggregate reporting reconstruction for historical missingness/selection tables, score-component availability, transparency summary, screening flow, predictor timing/coding, sparse-field/VAP audit, and Figure S1.

## Data and privacy boundary

No participant-level CSV/Excel file, row-level prediction output, source medical record, exact patient-level date, or model checkpoint is required to be committed publicly. Fitted objects are reconstructable from the authorized analysis-ready dataset using the public code and machine-readable environment.

## Historical/reconstruction labeling

Historical BSA-inclusive and earlier nine-predictor outputs remain clearly labeled as historical/exploratory or traceability analyses. Aggregate reporting tables that cannot be regenerated from the retained completed participant matrix are reconstructed from version-controlled non-patient-level revision metadata; unavailable raw history is not inferred.

## Known reproducibility limitation

The pre-imputation/raw uncapped predictor matrix was not retained in the revision archive. Therefore, repeated nested validation does not re-estimate imputation models or winsorization limits within outer folds and is correctly described as validation of the recoverable post-preprocessing pipeline.

## Release-readiness checks still requiring the repository owner

Before freezing `v1.0.0`:

1. run the two R scripts once under R 4.5.1 / Boruta 9.0.0 to confirm local execution;
2. verify the public GitHub tree contains no patient-level files, cache directories, or workstation-specific paths;
3. confirm `README.md`, `CITATION.cff`, `environment.yml`, and this inventory are the latest versions;
4. create the versioned release and archival DOI, then insert the persistent identifier into the manuscript Data Availability statement and reviewer response.
