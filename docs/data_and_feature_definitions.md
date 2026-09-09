# Data and feature definitions

## Required metadata fields

- `Study_row_id`: study-specific row identifier used by the analysis workflow.
- `Primary_split`: fixed original development/test identity retained from the primary analysis.
- `Pulmonary_infection`: binary model outcome representing later-documented ICU-acquired HAP/VAP after the 48-hour landmark and before ICU exit.

## Current primary 11 predictors

| Manuscript label | Analysis field | Type / scale |
|---|---|---|
| NEU | `NEUT_abs` | continuous, ×10^9/L |
| Intubation/tracheotomy | `Intubation_tracheotomy` | binary |
| MV | `Mechanical_ventilation` | binary |
| LDH | `LDH` | continuous, U/L |
| LYM | `LYMPH_abs` | continuous, ×10^9/L |
| BUN | `BUN` | continuous, mmol/L |
| CCI | `CCI` | continuous score |
| FIB | `FIB` | continuous, g/L |
| Surgery | `Surgery` | binary |
| Diuretics | `Diuretics` | binary |
| TCO2 | `CO2` | continuous, mmol/L |

Treatment-support variables encode documented occurrence during the first 48 hours after ICU admission rather than full-ICU-course exposure, treatment duration, or activity exactly at the 48-hour landmark.
