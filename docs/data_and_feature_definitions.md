# Data and feature definitions

## Required metadata fields

- `Study_row_id`: study-specific row identifier used by the analysis workflow.
- `Primary_split`: preserved fixed original training/test identity.
- `Pulmonary_infection`: binary endpoint indicating a later-documented hospital-acquired pneumonia (HAP) or ventilator-associated pneumonia (VAP) event occurring more than 48 hours after ICU admission and before ICU exit.

The endpoint has a variable post-landmark observation horizon and is not a fixed 7-day or 14-day cumulative-incidence outcome.

## Current primary 11 predictors

| Manuscript label | Analysis field | Type / scale |
|---|---|---|
| NEU | `NEUT_abs` | continuous, ×10^9/L |
| Intubation/tracheotomy | `Intubation_tracheotomy` | binary |
| MV | `Mechanical_ventilation` | binary |
| LDH | `LDH` | continuous, U/L |
| LYM | `LYMPH_abs` | continuous, ×10^9/L |
| BUN | `BUN` | continuous, mmol/L |
| CCI | `CCI` | continuous age-adjusted Charlson score |
| FIB | `FIB` | continuous, g/L |
| Surgery | `Surgery` | binary |
| Diuretics | `Diuretics` | binary |
| TCO2 | `CO2` | continuous, mmol/L |

## Predictor timing

Laboratory predictors represent the first available result during the first 48 hours after ICU admission.

This applies to the retained laboratory predictors:

- NEU;
- LDH;
- LYM;
- BUN;
- FIB;
- TCO2.

Procedure, organ-support, and medication predictors represent any documented exposure at any time during the first 48 hours after ICU admission.

This applies to:

- intubation/tracheotomy;
- mechanical ventilation;
- surgery;
- diuretics;

and to other first-48-hour intervention/medication candidate variables used during candidate screening.

These fields do not encode:

- exposure duration;
- treatment start/stop time;
- cumulative dose or duration;
- whether exposure was active exactly at the 48-hour landmark.

The revision archive does not retain the exact distribution of laboratory sampling times within the first 48 hours or whether a result was obtained on admission versus later during that window.

## Age-adjusted Charlson Comorbidity Index

The analysis field:

```text
CCI
