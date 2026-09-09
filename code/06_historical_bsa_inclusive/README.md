# Historical BSA-inclusive exploratory analysis

This directory preserves the archived **BSA-inclusive 10-predictor** analysis retained in the Supplementary Material for traceability. It is secondary/exploratory and is not the current primary prognostic model.

## Files

- `historical_BSA_10predictor_models.py`  
  Historical eight-model workflow using the 10 predictors: BSA, NEU, intubation/tracheotomy, MV, LDH, LYM, BUN, CCI, FIB, and surgery. It reconstructs the historical model-comparison figures and summary outputs.

- `historical_BSA_LightGBM_SHAP.py`  
  Historical LightGBM SHAP workflow used to reconstruct the BSA-inclusive interpretation figures.

- `historical_BSA_TableS11_S12.py`  
  Table-only reconstruction of Supplementary Tables S11 and S12 from locally generated fixed-test calibrated prediction files.

## Supplementary figure mapping

The archived scripts correspond to the historical supplementary figures as follows:

- **Figure S2**: SHAP dependence-style feature scatter plots from `historical_BSA_LightGBM_SHAP.py` (`图4_SHAP散点图_全部10个特征.png`).
- **Figure S4A**: historical training ROC comparison from `historical_BSA_10predictor_models.py`.
- **Figure S4B**: historical fixed-test ROC comparison from `historical_BSA_10predictor_models.py`.
- **Figure S4C**: historical fixed-test decision-curve analysis from `historical_BSA_10predictor_models.py`.
- **Figure S4D**: historical fixed-test precision-recall analysis from `historical_BSA_10predictor_models.py`.
- **Figure S5A-B**: historical apparent-training and fixed-test multi-metric radial summaries.
- **Figure S5C-D**: historical fixed-test calibration before and after Platt scaling.
- **Figure S6A**: SHAP beeswarm.
- **Figure S6B**: mean absolute SHAP feature importance.
- **Figure S6C**: SHAP main-effect / interaction-effect matrix.
- **Figure S6D-F**: representative waterfall plots. The plotting script writes high-, low-, and borderline-output cases as separate files; the final supplementary composite orders them as **low, borderline, high**.

## Historical implementation note

The code is intentionally preserved as a reconstruction of the archived exploratory analysis rather than silently modernized.

In particular, the historical script:

1. performs its archived missing-value handling before the 70/30 stratified split;
2. uses `random_state=42` for the stratified split;
3. fits MinMax scaling on the historical training partition;
4. performs Bayesian hyperparameter tuning and Platt calibration within the training partition; and
5. derives its operating threshold from calibrated predictions on the training set in the archived implementation.

The last step is **not** the strict cross-fitted OOF threshold procedure used in the current Primary11 workflow. This distinction is documented rather than retroactively changing the historical analysis, because changing the implementation would no longer reproduce the archived supplementary results.

## Restricted data and patient-level outputs

No patient-level data are included in this repository.

The public versions of the scripts disable patient-level CSV exports by default. To generate the local prediction file needed for Table S11/S12 reconstruction, explicitly add `--export-patient-level`. Generated patient-level CSV files must remain in a restricted local workspace and must not be committed to the public repository.

## Example: historical model workflow

```bash
python historical_BSA_10predictor_models.py \
  --data-file /restricted/path/analysis_dataset.csv \
  --output outputs/historical_bsa_inclusive \
  --export-patient-level
```

The opt-in calibrated fixed-test prediction file used for Table S11/S12 is:

`historical_BSA10_test_predictions_calibrated_with_StudyRowID.csv`

Model/scaler artifacts are also written locally for the SHAP workflow. Repository `.gitignore` rules should exclude `*.pkl`, `outputs/`, and patient-level CSV files.

## Example: SHAP workflow

```bash
python historical_BSA_LightGBM_SHAP.py \
  --data-file /restricted/path/analysis_dataset.csv \
  --model-dir outputs/historical_bsa_inclusive \
  --output outputs/historical_bsa_shap
```

Patient-level SHAP values are not exported unless `--export-patient-level` is explicitly supplied. Aggregate SHAP importance and interaction summaries can still be generated locally.

## Example: Tables S11-S12

After running the historical model workflow with `--export-patient-level`, compare the historical probabilities with the current Primary11 fixed-test probabilities:

```bash
python historical_BSA_TableS11_S12.py \
  --historical outputs/historical_bsa_inclusive/historical_BSA10_test_predictions_calibrated_with_StudyRowID.csv \
  --primary11 /restricted/path/Primary11_test_predictions_calibrated_with_StudyRowID.csv \
  --output outputs/historical_bsa_table_s11_s12
```

The table script checks the locked fixed-test sample size (`n=1,011`) and event count (`407`) and stops if patient/outcome alignment is unsafe.

## Interpretation

The BSA-inclusive and current Primary11 predictor sets differ beyond the presence/absence of BSA. Therefore, Supplementary Tables S11-S12 are descriptive model-set comparisons and are **not** interpreted as an isolated causal or incremental effect of broad-spectrum antibiotic exposure.
