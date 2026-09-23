# Model reconstruction guide

## Purpose

This document describes how to reconstruct the analyses reported for the current 11-predictor primary model set using an authorized local copy of the analysis-ready dataset.

Patient-level data are not distributed in this public repository.

The repository is intended to document the statistical and modeling logic used for the revised manuscript and to make clear which analyses can be reconstructed from the retained revision archive and which historical source-level operations can no longer be reproduced.

## Prediction target

The operational prediction target is a later-documented hospital-acquired pneumonia (HAP) or ventilator-associated pneumonia (VAP) event occurring more than 48 hours after ICU admission and before ICU exit.

The estimand is therefore a variable-horizon event probability conditional on remaining in the analytic cohort at the 48-hour landmark.

It is not a fixed 7-day or 14-day cumulative-incidence target.

## Cohort anchors

The archived analysis-ready cohort contains:

- 3,368 patients;
- 1,357 documented pulmonary-infection events.

The preserved fixed internal comparison split contains:

- training set: n = 2,357; events = 950;
- fixed internal test set: n = 1,011; events = 407.

The preserved `Primary_split` field should be used whenever available.

Split identity should not be reconstructed or inferred from patient identifiers.

The current fixed test partition is used as a standardized within-study comparison set. Because several revision-specific analyses were conducted after results from this partition had already been examined, it should not be described as untouched validation of the entire revised workflow.

## Required metadata fields

At minimum, the current primary workflow requires:

- `Study_row_id`
- `Primary_split`
- `Pulmonary_infection`

The current primary 11 predictors are:

| Manuscript label | Analysis field | Type |
|---|---|---|
| NEU | `NEUT_abs` | continuous |
| Intubation/tracheotomy | `Intubation_tracheotomy` | binary |
| MV | `Mechanical_ventilation` | binary |
| LDH | `LDH` | continuous |
| LYM | `LYMPH_abs` | continuous |
| BUN | `BUN` | continuous |
| CCI | `CCI` | continuous age-adjusted Charlson score |
| FIB | `FIB` | continuous |
| Surgery | `Surgery` | binary |
| Diuretics | `Diuretics` | binary |
| TCO2 | `CO2` | continuous |

Laboratory predictors represent the first available result during the first 48 hours after ICU admission.

Procedure, organ-support, and medication predictors represent any documented exposure during the first 48 hours.

They do not represent exposure duration or status active exactly at the 48-hour landmark.

The revision archive does not retain exact laboratory sampling-time distributions, admission-versus-later sampling status, treatment start/stop times, cumulative exposure duration, or active-at-landmark status.

## Current primary feature-selection reconstruction

Run:

```bash
Rscript code/01_feature_selection/primary_feature_selection.R
