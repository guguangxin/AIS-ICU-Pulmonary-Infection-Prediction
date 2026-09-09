# -*- coding: utf-8 -*-
"""Export Supplementary Table S15 coefficients for the current Primary11 base LR.

This script reconstructs the *uncalibrated* base logistic-regression coefficients in
original measurement units. It uses the selected primary LR hyperparameters reported
for the current 11-predictor analysis and the fixed training partition.

Preferred input: the authorized analysis-ready CSV with a preserved `Primary_split`
column. Patient-level data are not included in the public repository.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

OUTCOME = "Pulmonary_infection"
SPLIT_COL = "Primary_split"
ID_COL = "Study_row_id"

FEATURES = [
    "NEUT_abs",
    "Intubation_tracheotomy",
    "Mechanical_ventilation",
    "LDH",
    "LYMPH_abs",
    "BUN",
    "CCI",
    "FIB",
    "Surgery",
    "Diuretics",
    "CO2",
]

CONTINUOUS = ["NEUT_abs", "LDH", "LYMPH_abs", "BUN", "CCI", "FIB", "CO2"]
BINARY = ["Intubation_tracheotomy", "Mechanical_ventilation", "Surgery", "Diuretics"]

DISPLAY = {
    "NEUT_abs": "NEU",
    "Intubation_tracheotomy": "Intubation/tracheotomy",
    "Mechanical_ventilation": "MV",
    "LDH": "LDH",
    "LYMPH_abs": "LYM",
    "BUN": "BUN",
    "CCI": "CCI",
    "FIB": "FIB",
    "Surgery": "Surgery",
    "Diuretics": "Diuretics",
    "CO2": "TCO2",
}

CODING = {
    "NEUT_abs": "Continuous, x10^9/L",
    "Intubation_tracheotomy": "Binary: 1=yes, 0=no",
    "Mechanical_ventilation": "Binary: 1=yes, 0=no",
    "LDH": "Continuous, U/L",
    "LYMPH_abs": "Continuous, x10^9/L",
    "BUN": "Continuous, mmol/L",
    "CCI": "Continuous score",
    "FIB": "Continuous, g/L",
    "Surgery": "Binary: 1=yes, 0=no",
    "Diuretics": "Binary: 1=yes, 0=no",
    "CO2": "Continuous, mmol/L",
}

# Selected current Primary11 LR hyperparameters.
LR_C = 0.2493730
LR_SOLVER = "lbfgs"
RANDOM_STATE = 42

EXPECTED = {
    "Intercept": -2.47312750,
    "NEU": 0.16384220,
    "Intubation/tracheotomy": 1.05995301,
    "MV": 0.93980262,
    "LDH": 0.00193951,
    "LYM": -0.40344548,
    "BUN": 0.02949279,
    "CCI": 0.21036992,
    "FIB": 0.20418390,
    "Surgery": 0.73937104,
    "Diuretics": 0.42106097,
    "TCO2": -0.04858044,
}


def read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = df.columns.astype(str).str.replace("\ufeff", "", regex=False).str.strip()
    return df


def select_training_rows(df: pd.DataFrame, recreate_split: bool) -> pd.DataFrame:
    if SPLIT_COL in df.columns:
        s = df[SPLIT_COL].astype(str).str.strip().str.lower()
        train_mask = s.isin({"train", "training", "0"})
        test_mask = s.isin({"test", "testing", "1"})
        if not train_mask.any() or not test_mask.any():
            raise ValueError("Primary_split exists but Train/Test labels could not be recognized.")
        train = df.loc[train_mask].copy()
        test = df.loc[test_mask].copy()
        if len(train) != 2357 or int(train[OUTCOME].sum()) != 950:
            raise ValueError("Preserved Train partition does not match n=2357/events=950.")
        if len(test) != 1011 or int(test[OUTCOME].sum()) != 407:
            raise ValueError("Preserved Test partition does not match n=1011/events=407.")
        return train

    if not recreate_split:
        raise ValueError(
            "Input does not contain Primary_split. Use the preserved exact-split file. "
            "Only use --recreate-split for an authorized archived cohort known to retain "
            "the original row order used for the 70/30 stratified random_state=42 split."
        )

    idx = np.arange(len(df))
    y = df[OUTCOME].astype(int).to_numpy()
    tr_idx, te_idx = train_test_split(
        idx,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    train = df.iloc[tr_idx].copy()
    test = df.iloc[te_idx].copy()
    if len(train) != 2357 or int(train[OUTCOME].sum()) != 950:
        raise ValueError("Recreated Train partition does not match n=2357/events=950.")
    if len(test) != 1011 or int(test[OUTCOME].sum()) != 407:
        raise ValueError("Recreated Test partition does not match n=1011/events=407.")
    print("WARNING: Primary_split was recreated from archived row order; preserved split identity is preferred.")
    return train


def backtransform_coefficients(model: LogisticRegression, scaler: MinMaxScaler) -> tuple[float, dict[str, float]]:
    beta_scaled = np.asarray(model.coef_[0], dtype=float)
    intercept_original = float(model.intercept_[0])
    beta_original: dict[str, float] = {}

    for i, feature in enumerate(FEATURES):
        b = float(beta_scaled[i])
        if feature in CONTINUOUS:
            j = CONTINUOUS.index(feature)
            data_range = float(scaler.data_range_[j])
            data_min = float(scaler.data_min_[j])
            if data_range == 0:
                raise ValueError(f"Zero range for continuous feature: {feature}")
            beta_original[feature] = b / data_range
            intercept_original -= b * data_min / data_range
        else:
            beta_original[feature] = b

    return intercept_original, beta_original


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=os.environ.get("AIS_ICU_DATA_FILE", "restricted_data/BSAfree_feature_selection_input_exactsplit.csv"),
        help="Authorized analysis-ready CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("AIS_ICU_OUTPUT_DIR", "outputs/primary_analysis"),
    )
    parser.add_argument(
        "--recreate-split",
        action="store_true",
        help="Fallback only for the authorized archived cohort with original row order when Primary_split is absent.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = read_csv(input_path)
    required = {OUTCOME, *FEATURES}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if len(df) != 3368 or int(df[OUTCOME].sum()) != 1357:
        raise ValueError("Cohort anchor check failed: expected n=3368/events=1357.")
    if df[FEATURES].isna().any().any():
        raise ValueError("Primary11 fields contain missing values; this script expects the archived analysis-ready matrix.")

    train = select_training_rows(df, recreate_split=args.recreate_split)
    X = train[FEATURES].copy()
    y = train[OUTCOME].astype(int).to_numpy()

    scaler = MinMaxScaler()
    X_scaled = X.copy()
    X_scaled.loc[:, CONTINUOUS] = scaler.fit_transform(X[CONTINUOUS])

    model = LogisticRegression(
        C=LR_C,
        penalty="l2",
        solver=LR_SOLVER,
        random_state=RANDOM_STATE,
        max_iter=3000,
    )
    model.fit(X_scaled.to_numpy(), y)

    intercept, beta = backtransform_coefficients(model, scaler)

    rows = [{"Predictor": "Intercept", "Coefficient": intercept, "exp_beta": np.nan, "Coding_scale": "Intercept"}]
    for feature in FEATURES:
        b = beta[feature]
        rows.append({
            "Predictor": DISPLAY[feature],
            "Coefficient": b,
            "exp_beta": float(np.exp(b)),
            "Coding_scale": CODING[feature],
        })

    result = pd.DataFrame(rows)
    out_file = out_dir / "Table_S15_primary_LR_coefficients.csv"
    result.to_csv(out_file, index=False, encoding="utf-8-sig")

    max_abs_diff = 0.0
    for _, row in result.iterrows():
        name = row["Predictor"]
        if name in EXPECTED:
            max_abs_diff = max(max_abs_diff, abs(float(row["Coefficient"]) - EXPECTED[name]))

    print(result.to_string(index=False))
    print(f"\nSaved: {out_file}")
    print(f"Maximum absolute difference from reported Table S15 anchors: {max_abs_diff:.3e}")
    if max_abs_diff > 5e-6:
        raise RuntimeError("Coefficient audit did not reproduce the reported Table S15 values within tolerance.")


if __name__ == "__main__":
    main()
