# Aggregate reporting reconstruction

This directory covers manuscript/supplement items that are **reporting, audit, or historical aggregate summaries** rather than analyses that can be regenerated from the retained analysis-ready participant matrix.

Covered here:

- Table S1: archived missing-data summary;
- Tables S3-S4: historical BSA-inclusive selection-stability summaries;
- Table S16: A2DS2/ISAN component-availability comparison;
- Table S17: reporting/reproducibility transparency summary;
- Table S24: sequential screening-flow summary;
- Table S29: operational timing/coding rules for the 11 primary predictors;
- Table S31: sparse-field and HAP/VAP airway-timing audit;
- Figure S1: patient inclusion/exclusion flowchart.

The bundled `reporting_metadata.json` contains only aggregate, non-patient-level revision metadata. This approach is intentional because the revision archive does not retain the pre-imputation raw matrix, complete excluded-patient records, or exact predictor timestamps needed to derive some of these reporting items de novo. No unsupported values are reconstructed.

Run:

```bash
python code/08_reporting_reconstruction/reconstruct_reporting_tables_and_flowchart.py
```

See `docs/table_figure_code_map.md` for the generating/reconstruction route for every reported main and supplementary table and figure.
