# Historical nine-predictor optimism analysis (Supplementary Table S14)

This directory provides a **reconstruction**, not a recovered copy of the original historical generation script, for Supplementary Table S14.

## Scope

Table S14 is retained only for analysis traceability. It summarizes a 1,000-resample fixed-hyperparameter bootstrap optimism analysis of the **base (uncalibrated)** GBDT and LightGBM estimators from the earlier nine-predictor BSA-free model specification:

- NEU (`NEUT_abs`)
- intubation/tracheotomy (`Intubation_tracheotomy`)
- mechanical ventilation (`Mechanical_ventilation`)
- LDH
- LYM (`LYMPH_abs`)
- BUN
- CCI
- FIB
- surgery (`Surgery`)

It is **not** used as the uncertainty estimate for the current 11-predictor primary models.

## Files

- `historical_9predictor_bootstrap_optimism.py` — reconstruction of the archived S14 bootstrap procedure.
- `historical_9predictor_selected_params.json` — machine-readable GBDT/LightGBM selected values retained in the archived Supplementary Table S2.

## Historical procedure reconstructed

The authorized analysis-ready cohort is restricted to the original training set (n=2,357; 950 events). Continuous predictors are min-max scaled using the original training cohort. Previously selected model hyperparameters are held fixed.

For each bootstrap replicate, the training cohort is sampled with replacement, a fresh base estimator is fitted to that bootstrap sample, and two AUCs are calculated: the AUC in the bootstrap sample and the AUC obtained when the same bootstrap-fitted estimator is applied to the complete original training cohort. Optimism is the former minus the latter. Mean optimism is subtracted from the original apparent training AUC. Percentile intervals are calculated from the replicate distributions.

The archived Supplementary Table S14 reported:

| Model | Apparent AUC | Mean optimism | Archived "SE" | Optimism 95% interval | Corrected AUC | Corrected archived "SE" | Corrected 95% interval |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GBDT | 0.876 | 0.035 | 0.006 | 0.023-0.046 | 0.841 | 0.006 | 0.830-0.853 |
| LightGBM | 0.863 | 0.029 | 0.007 | 0.016-0.041 | 0.835 | 0.007 | 0.823-0.848 |

These values are embedded in the script only as **QA anchors**. They are not used to calculate the reconstructed results.

## Run

Use an authorized local analysis-ready file; patient-level data are not included in this repository.

```bash
python code/07_historical_nine_predictor_traceability/historical_9predictor_bootstrap_optimism.py \
  --input /path/to/authorized/analysis_ready.csv \
  --bootstrap 1000
```

For a quick smoke test only:

```bash
python code/07_historical_nine_predictor_traceability/historical_9predictor_bootstrap_optimism.py \
  --input /path/to/authorized/analysis_ready.csv \
  --bootstrap 5 \
  --models LightGBM
```

The 1,000-resample GBDT reconstruction can take substantial time on a laptop.

## Fixed split

If the authorized file includes `Primary_split`, that field is used directly. If it is absent, the script can reconstruct the archived 70:30 stratified split using `random_state=42`, but this fallback assumes that the authorized analysis-ready file preserves the original row order. The script stops unless the cohort anchors are exactly:

- total n=3,368; events=1,357
- training n=2,357; events=950
- test n=1,011; events=407

## Precision / software caveat

The full-precision historical `BayesSearchCV.best_params_` export was not located in the revision archive. The bundled JSON therefore uses the selected continuous values **as displayed** in the archived Supplementary Table S2 (for example, GBDT learning rate 0.0135 and LightGBM learning rate 0.0213). Those displayed values were rounded. Small numerical deviations from the archived S14 numbers can therefore occur because of parameter rounding and library-version differences.

If a full-precision archived parameter export is later recovered, pass it using `--params-json` without changing the reconstruction algorithm.

## Privacy

The public repository must not contain the patient-level input. The script writes only aggregate table results, aggregate bootstrap-replicate AUC values, and run metadata. Repository `.gitignore` rules should keep local CSV outputs out of GitHub.
