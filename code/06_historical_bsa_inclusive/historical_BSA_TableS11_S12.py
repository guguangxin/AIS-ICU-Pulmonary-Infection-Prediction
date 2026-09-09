#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reproduce Supplementary Tables S11 and S12 for the historical exploratory
BSA-inclusive 10-predictor analysis versus the current BSA-free Primary11 analysis.

This script is intentionally table-only. The archived 10-predictor analysis code can
continue to generate historical Figures S2/S4/S5/S6. This file reconstructs the two
side-by-side comparison tables from locally generated fixed-test calibrated prediction
files, without publishing patient-level data.

Expected inputs
---------------
1) Historical BSA-inclusive 10-predictor fixed-test calibrated probabilities.
2) Current Primary11 fixed-test calibrated probabilities.

The preferred input from each modeling script is the patient-level CSV containing:
- Study_row_id
- True Label (or equivalent)
- one calibrated-probability column for each of the eight algorithms

Typical filename from the modeling workflow:
  测试集逐患者预测概率_校准后_带StudyRowID.csv

The script aligns the two prediction files by Study_row_id when available. If IDs are
absent, row-order alignment is permitted only when sample size, event count, and all
outcome labels are identical; a warning is written to the QA output.

Table S11
---------
- Historical BSA-inclusive AUC and 95% patient-level bootstrap interval
- Current Primary11 AUC and 95% patient-level bootstrap interval
- Descriptive delta AUC = Primary11 - BSA-inclusive
- No isolated-BSA paired hypothesis test is reported, because the predictor sets differ
  beyond BSA after the BSA-free re-selection.

Table S12
---------
- Brier score
- Calibration intercept
- Calibration slope
for historical BSA-inclusive and current Primary11 calibrated predictions.

Calibration definitions match the current primary analysis code:
- intercept: slope fixed to 1 in logit space, intercept estimated by maximum likelihood
- slope: unpenalized logistic regression of outcome on logit(predicted probability)

Bootstrap convention matches the primary modeling code:
- 1,000 patient-level resamples by default
- random seed 42
- percentile 2.5th and 97.5th percentiles

No patient-level data are bundled with this repository script. Generated CSV outputs
should remain under outputs/ and are ignored by the public repository .gitignore.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

RANDOM_STATE = 42
N_BOOTSTRAP = 1000
EXPECTED_TEST_N = 1011
EXPECTED_TEST_EVENTS = 407

MODEL_ORDER = [
    "GBDT",
    "LightGBM",
    "RF",
    "XGBoost",
    "MLP",
    "LR",
    "DT",
    "NB",
]

MODEL_DISPLAY = {
    "GBDT": "Gradient Boosting Decision Tree",
    "LightGBM": "LightGBM",
    "RF": "Random Forest",
    "XGBoost": "XGBoost",
    "MLP": "Multilayer Perceptron",
    "LR": "Logistic Regression",
    "DT": "Decision Tree",
    "NB": "Naive Bayes",
}

# Candidate labels accepted from the original/current modeling scripts.
MODEL_ALIASES = {
    "GBDT": [
        "Gradient Boosting Decision Tree Prob",
        "GradientBoostingDecisionTree Prob",
        "GBDT Prob",
        "GBDT",
        "Primary11_GBDT",
    ],
    "LightGBM": ["LightGBM Prob", "LightGBM"],
    "RF": ["Random Forest Prob", "RandomForest Prob", "RF Prob", "RF"],
    "XGBoost": ["XGBoost Prob", "XGB Prob", "XGBoost", "XGB"],
    "MLP": ["Multilayer Perceptron Prob", "MLP Prob", "MLP"],
    "LR": [
        "Logistic Regression Prob",
        "LogisticRegression Prob",
        "LR Prob",
        "LR",
        "Primary11_LR",
    ],
    "DT": ["Decision Tree Prob", "DecisionTree Prob", "DT Prob", "DT"],
    "NB": ["Naive Bayes Prob", "NaiveBayes Prob", "NB Prob", "NB"],
}

ID_ALIASES = ["Study_row_id", "StudyRowID", "Study row id", "study_id", "ID"]
Y_ALIASES = ["True Label", "True_Label", "TrueLabel", "Pulmonary_infection", "Actual", "Outcome"]

# Archived values from the final revised Supplementary Tables S11-S12.
# These values are QA targets only; they are NOT used to generate the tables.
ARCHIVED_QA = {
    "GBDT":      {"hist_auc": 0.864, "hist_brier": 0.1453, "hist_int": -0.0415, "hist_slope": 1.0280,
                  "curr_auc": 0.8362, "curr_brier": 0.1620, "curr_int": -0.0643, "curr_slope": 1.0491},
    "LightGBM":  {"hist_auc": 0.865, "hist_brier": 0.1460, "hist_int": -0.0326, "hist_slope": 1.0255,
                  "curr_auc": 0.8347, "curr_brier": 0.1632, "curr_int": -0.0630, "curr_slope": 1.0312},
    "RF":        {"hist_auc": 0.863, "hist_brier": 0.1465, "hist_int": -0.0284, "hist_slope": 1.0232,
                  "curr_auc": 0.8340, "curr_brier": 0.1635, "curr_int": -0.0656, "curr_slope": 1.0371},
    "XGBoost":   {"hist_auc": 0.864, "hist_brier": 0.1459, "hist_int": -0.0329, "hist_slope": 1.0204,
                  "curr_auc": 0.8345, "curr_brier": 0.1635, "curr_int": -0.0735, "curr_slope": 1.0296},
    "MLP":       {"hist_auc": 0.864, "hist_brier": 0.1444, "hist_int": -0.0380, "hist_slope": 1.0252,
                  "curr_auc": 0.8274, "curr_brier": 0.1649, "curr_int": -0.0682, "curr_slope": 1.0103},
    "LR":        {"hist_auc": 0.861, "hist_brier": 0.1467, "hist_int": -0.0151, "hist_slope": 1.0459,
                  "curr_auc": 0.8273, "curr_brier": 0.1664, "curr_int": -0.0594, "curr_slope": 1.0186},
    "DT":        {"hist_auc": 0.849, "hist_brier": 0.1526, "hist_int":  0.0016, "hist_slope": 1.1995,
                  "curr_auc": 0.8278, "curr_brier": 0.1695, "curr_int": -0.0606, "curr_slope": 1.4307},
    "NB":        {"hist_auc": 0.857, "hist_brier": 0.1622, "hist_int":  0.0425, "hist_slope": 1.0169,
                  "curr_auc": 0.8232, "curr_brier": 0.1847, "curr_int":  0.0145, "curr_slope": 0.9987},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--historical", type=Path, required=True,
                   help="Historical BSA-inclusive fixed-test calibrated prediction CSV.")
    p.add_argument("--primary11", type=Path, required=True,
                   help="Current Primary11 fixed-test calibrated prediction CSV.")
    p.add_argument("--output", type=Path,
                   default=Path("outputs/historical_bsa_table_s11_s12"),
                   help="Local output directory (default: outputs/historical_bsa_table_s11_s12).")
    p.add_argument("--bootstrap", type=int, default=N_BOOTSTRAP,
                   help="Patient-level bootstrap resamples for AUC intervals (default: 1000).")
    p.add_argument("--seed", type=int, default=RANDOM_STATE,
                   help="Bootstrap random seed (default: 42).")
    return p.parse_args()


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = df.columns.astype(str).str.replace("\ufeff", "", regex=False).str.strip()
    return df


def find_column(df: pd.DataFrame, aliases: Iterable[str], *, required: bool = True) -> str | None:
    norm_to_actual: Dict[str, str] = {_norm(c): c for c in df.columns}
    for alias in aliases:
        hit = norm_to_actual.get(_norm(alias))
        if hit is not None:
            return hit
    if required:
        raise KeyError(f"Could not identify required column. Tried aliases={list(aliases)}. Available={list(df.columns)}")
    return None


def resolve_model_columns(df: pd.DataFrame) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for key in MODEL_ORDER:
        out[key] = find_column(df, MODEL_ALIASES[key], required=True)  # type: ignore[assignment]
    return out


def clean_probability(x: pd.Series, label: str) -> np.ndarray:
    p = pd.to_numeric(x, errors="coerce").to_numpy(float)
    if np.isnan(p).any():
        raise ValueError(f"{label}: probability column contains missing/non-numeric values.")
    if np.any((p < 0) | (p > 1)):
        raise ValueError(f"{label}: probability values must lie in [0, 1].")
    return p


def clean_outcome(x: pd.Series, label: str) -> np.ndarray:
    y = pd.to_numeric(x, errors="coerce").to_numpy(float)
    if np.isnan(y).any():
        raise ValueError(f"{label}: outcome column contains missing/non-numeric values.")
    if not np.all(np.isin(y, [0, 1])):
        raise ValueError(f"{label}: outcome must be binary 0/1.")
    return y.astype(int)


def align_inputs(hist: pd.DataFrame, curr: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    hist_id = find_column(hist, ID_ALIASES, required=False)
    curr_id = find_column(curr, ID_ALIASES, required=False)
    hist_y = find_column(hist, Y_ALIASES, required=True)
    curr_y = find_column(curr, Y_ALIASES, required=True)

    if hist_id is not None and curr_id is not None:
        h = hist.copy()
        c = curr.copy()
        h["__join_id"] = h[hist_id].astype(str)
        c["__join_id"] = c[curr_id].astype(str)
        if h["__join_id"].duplicated().any() or c["__join_id"].duplicated().any():
            raise ValueError("Study_row_id contains duplicates; patient-level alignment is unsafe.")
        if set(h["__join_id"]) != set(c["__join_id"]):
            only_h = len(set(h["__join_id"]) - set(c["__join_id"]))
            only_c = len(set(c["__join_id"]) - set(h["__join_id"]))
            raise ValueError(f"Historical and Primary11 IDs differ (historical-only={only_h}, primary-only={only_c}).")
        c = c.set_index("__join_id").loc[h["__join_id"]].reset_index(drop=True)
        h = h.reset_index(drop=True)
        alignment = "Study_row_id"
    else:
        if len(hist) != len(curr):
            raise ValueError("IDs are unavailable and row counts differ; cannot align safely.")
        h = hist.reset_index(drop=True).copy()
        c = curr.reset_index(drop=True).copy()
        alignment = "row_order_after_exact_label_check"

    yh = clean_outcome(h[hist_y], "historical")
    yc = clean_outcome(c[curr_y], "primary11")
    if not np.array_equal(yh, yc):
        raise ValueError("Outcome labels are not identical after alignment; stop rather than compare mismatched patients.")
    if len(yh) != EXPECTED_TEST_N or int(yh.sum()) != EXPECTED_TEST_EVENTS:
        raise ValueError(
            f"Fixed-test identity mismatch: expected n={EXPECTED_TEST_N}, events={EXPECTED_TEST_EVENTS}; "
            f"observed n={len(yh)}, events={int(yh.sum())}."
        )
    return h, c, alignment


def calibration_intercept_slope(y_true: np.ndarray, y_proba: np.ndarray, eps: float = 1e-6) -> Tuple[float, float]:
    """Match the definitions used in the current primary modeling script."""
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(y_proba, dtype=float), eps, 1 - eps)
    lp = logit(p)

    try:
        lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000)
        lr.fit(lp.reshape(-1, 1), y)
    except (TypeError, ValueError):
        lr = LogisticRegression(penalty="l2", C=1e12, solver="lbfgs", max_iter=2000)
        lr.fit(lp.reshape(-1, 1), y)
    slope = float(lr.coef_[0, 0])

    def neg_ll(b: np.ndarray) -> float:
        eta = lp + b[0]
        log_p = -np.logaddexp(0, -eta)
        log_1mp = -np.logaddexp(0, eta)
        return float(-(y * log_p + (1 - y) * log_1mp).sum())

    res = minimize(neg_ll, x0=np.array([0.0]), method="L-BFGS-B")
    if not res.success:
        raise RuntimeError(f"Calibration-intercept optimization failed: {res.message}")
    return float(res.x[0]), slope


def bootstrap_auc(y: np.ndarray, p: np.ndarray, n_boot: int, seed: int) -> Tuple[float, float, float, int]:
    # Intentionally mirrors the original primary bootstrap convention: seed reset for
    # each model so all models use the same sequence of patient-level resamples.
    np.random.seed(seed)
    n = len(y)
    vals: List[float] = []
    for _ in range(n_boot):
        idx = np.random.randint(0, n, n)
        yy = y[idx]
        if np.unique(yy).size < 2:
            continue
        vals.append(float(roc_auc_score(yy, p[idx])))
    if not vals:
        raise RuntimeError("No valid bootstrap resamples were produced.")
    v = np.asarray(vals, float)
    return float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), len(v)


def markdown_table_s11(df: pd.DataFrame) -> str:
    lines = [
        "| Model | Historical BSA-inclusive AUC (95% CI) | Current Primary11 AUC (95% CI) | Descriptive ΔAUC (Primary11 − BSA+) | Paired-test status | Interpretation |",
        "|---|---:|---:|---:|---|---|",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['Model']} | {r['Historical_BSA_AUC']:.3f} ({r['Historical_BSA_AUC_CI_low']:.3f}–{r['Historical_BSA_AUC_CI_high']:.3f}) "
            f"| {r['Primary11_AUC']:.3f} ({r['Primary11_AUC_CI_low']:.3f}–{r['Primary11_AUC_CI_high']:.3f}) "
            f"| {r['Delta_AUC_Primary11_minus_BSAplus']:+.3f} | {r['Paired_test_status']} | {r['Interpretation']} |"
        )
    return "\n".join(lines) + "\n"


def markdown_table_s12(df: pd.DataFrame) -> str:
    lines = [
        "| Model | Brier BSA+ | Brier Primary11 | ΔBrier (Primary11 − BSA+) | Intercept BSA+ | Slope BSA+ | Intercept Primary11 | Slope Primary11 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['Model']} | {r['Brier_BSAplus']:.4f} | {r['Brier_Primary11']:.4f} "
            f"| {r['Delta_Brier_Primary11_minus_BSAplus']:+.4f} | {r['Intercept_BSAplus']:+.4f} "
            f"| {r['Slope_BSAplus']:.4f} | {r['Intercept_Primary11']:+.4f} | {r['Slope_Primary11']:.4f} |"
        )
    return "\n".join(lines) + "\n"


def build_qa(s11: pd.DataFrame, s12: pd.DataFrame) -> pd.DataFrame:
    s11i = s11.set_index("Model_key")
    s12i = s12.set_index("Model_key")
    rows = []
    specs = [
        ("Historical_AUC", "Historical_BSA_AUC", "hist_auc", 0.003),
        ("Current_AUC", "Primary11_AUC", "curr_auc", 0.003),
        ("Historical_Brier", "Brier_BSAplus", "hist_brier", 0.003),
        ("Current_Brier", "Brier_Primary11", "curr_brier", 0.003),
        ("Historical_intercept", "Intercept_BSAplus", "hist_int", 0.020),
        ("Current_intercept", "Intercept_Primary11", "curr_int", 0.020),
        ("Historical_slope", "Slope_BSAplus", "hist_slope", 0.030),
        ("Current_slope", "Slope_Primary11", "curr_slope", 0.030),
    ]
    for key in MODEL_ORDER:
        for metric, col, target_key, tol in specs:
            source = s11i if col in s11i.columns else s12i
            computed = float(source.loc[key, col])
            archived = float(ARCHIVED_QA[key][target_key])
            rows.append({
                "Model": MODEL_DISPLAY[key],
                "Metric": metric,
                "Computed": computed,
                "Archived_reference": archived,
                "Difference": computed - archived,
                "Tolerance": tol,
                "Within_tolerance": abs(computed - archived) <= tol,
            })
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    if args.bootstrap < 1:
        raise ValueError("--bootstrap must be >= 1")
    out = args.output.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    hist = read_csv(args.historical.expanduser().resolve())
    curr = read_csv(args.primary11.expanduser().resolve())
    hist, curr, alignment = align_inputs(hist, curr)

    hist_y_col = find_column(hist, Y_ALIASES, required=True)
    curr_y_col = find_column(curr, Y_ALIASES, required=True)
    y = clean_outcome(hist[hist_y_col], "historical")
    # Explicit second check after alignment.
    if not np.array_equal(y, clean_outcome(curr[curr_y_col], "primary11")):
        raise RuntimeError("Outcome alignment changed unexpectedly.")

    hist_cols = resolve_model_columns(hist)
    curr_cols = resolve_model_columns(curr)

    s11_rows = []
    s12_rows = []
    bootstrap_rows = []

    for key in MODEL_ORDER:
        ph = clean_probability(hist[hist_cols[key]], f"historical/{key}")
        pc = clean_probability(curr[curr_cols[key]], f"primary11/{key}")

        auc_h = float(roc_auc_score(y, ph))
        auc_c = float(roc_auc_score(y, pc))
        mean_h, lo_h, hi_h, n_h = bootstrap_auc(y, ph, args.bootstrap, args.seed)
        mean_c, lo_c, hi_c, n_c = bootstrap_auc(y, pc, args.bootstrap, args.seed)

        brier_h = float(brier_score_loss(y, ph))
        brier_c = float(brier_score_loss(y, pc))
        int_h, slope_h = calibration_intercept_slope(y, ph)
        int_c, slope_c = calibration_intercept_slope(y, pc)

        s11_rows.append({
            "Model_key": key,
            "Model": MODEL_DISPLAY[key],
            "Historical_BSA_AUC": auc_h,
            "Historical_BSA_AUC_bootstrap_mean": mean_h,
            "Historical_BSA_AUC_CI_low": lo_h,
            "Historical_BSA_AUC_CI_high": hi_h,
            "Primary11_AUC": auc_c,
            "Primary11_AUC_bootstrap_mean": mean_c,
            "Primary11_AUC_CI_low": lo_c,
            "Primary11_AUC_CI_high": hi_c,
            "Delta_AUC_Primary11_minus_BSAplus": auc_c - auc_h,
            "Paired_test_status": "Not used as isolated-BSA test",
            "Interpretation": "Descriptive only; predictor sets differ beyond BSA",
        })
        s12_rows.append({
            "Model_key": key,
            "Model": MODEL_DISPLAY[key],
            "Brier_BSAplus": brier_h,
            "Brier_Primary11": brier_c,
            "Delta_Brier_Primary11_minus_BSAplus": brier_c - brier_h,
            "Intercept_BSAplus": int_h,
            "Slope_BSAplus": slope_h,
            "Intercept_Primary11": int_c,
            "Slope_Primary11": slope_c,
        })
        bootstrap_rows.extend([
            {"Model": MODEL_DISPLAY[key], "Analysis": "Historical BSA+", "Bootstrap_mean_AUC": mean_h,
             "CI_low": lo_h, "CI_high": hi_h, "Successful_resamples": n_h},
            {"Model": MODEL_DISPLAY[key], "Analysis": "Current Primary11", "Bootstrap_mean_AUC": mean_c,
             "CI_low": lo_c, "CI_high": hi_c, "Successful_resamples": n_c},
        ])

    s11 = pd.DataFrame(s11_rows)
    s12 = pd.DataFrame(s12_rows)
    boot = pd.DataFrame(bootstrap_rows)
    qa = build_qa(s11, s12)

    s11.to_csv(out / "Table_S11_historical_BSA_vs_Primary11.csv", index=False, encoding="utf-8-sig")
    s12.to_csv(out / "Table_S12_historical_BSA_vs_Primary11.csv", index=False, encoding="utf-8-sig")
    boot.to_csv(out / "Table_S11_AUC_bootstrap_audit.csv", index=False, encoding="utf-8-sig")
    qa.to_csv(out / "historical_BSA_TableS11_S12_archived_QA.csv", index=False, encoding="utf-8-sig")
    (out / "Table_S11_historical_BSA_vs_Primary11.md").write_text(markdown_table_s11(s11), encoding="utf-8")
    (out / "Table_S12_historical_BSA_vs_Primary11.md").write_text(markdown_table_s12(s12), encoding="utf-8")

    qa_pass = bool(qa["Within_tolerance"].all())
    note = (
        "Historical BSA-inclusive Table S11/S12 reconstruction\n"
        "======================================================\n"
        f"Historical prediction file: {args.historical.expanduser().resolve()}\n"
        f"Primary11 prediction file : {args.primary11.expanduser().resolve()}\n"
        f"Alignment method          : {alignment}\n"
        f"Fixed test                : n={len(y)}, events={int(y.sum())}\n"
        f"Bootstrap                 : n={args.bootstrap}, seed={args.seed}\n"
        f"Archived-value QA         : {'PASS' if qa_pass else 'CHECK DIFFERENCES'}\n\n"
        "Interpretation: Table S11 is descriptive only. The historical BSA-inclusive\n"
        "10-predictor set and current BSA-free 11-predictor set differ in more than BSA\n"
        "alone, so no isolated-BSA paired hypothesis test is reported.\n"
    )
    (out / "RUN_SUMMARY.txt").write_text(note, encoding="utf-8")

    print("\n" + "=" * 88)
    print("Historical BSA-inclusive Tables S11/S12 complete")
    print("=" * 88)
    print(f"Alignment: {alignment}")
    print(f"Fixed test: n={len(y)}, events={int(y.sum())}")
    print(f"Archived-value QA: {'PASS' if qa_pass else 'CHECK DIFFERENCES'}")
    print("\nTable S11 point estimates:")
    print(s11[["Model", "Historical_BSA_AUC", "Primary11_AUC", "Delta_AUC_Primary11_minus_BSAplus"]].to_string(index=False))
    print("\nTable S12 calibration:")
    print(s12[["Model", "Brier_BSAplus", "Brier_Primary11", "Intercept_BSAplus", "Slope_BSAplus", "Intercept_Primary11", "Slope_Primary11"]].to_string(index=False))
    print(f"\nOutputs: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
