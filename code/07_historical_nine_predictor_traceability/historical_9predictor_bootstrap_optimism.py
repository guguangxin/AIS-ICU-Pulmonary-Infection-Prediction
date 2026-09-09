#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reconstruction of historical Supplementary Table S14
=====================================================

This is a RECONSTRUCTION script, not the recovered original generation script.
It reconstructs the historical fixed-hyperparameter bootstrap optimism analysis
for the earlier BSA-free NINE-predictor GBDT and LightGBM base estimators.

Historical nine-predictor set
-----------------------------
NEUT_abs, Intubation_tracheotomy, Mechanical_ventilation, LDH,
LYMPH_abs, BUN, CCI, FIB, Surgery

Historical procedure reconstructed here
---------------------------------------
1. Use the original fixed training cohort (n=2357; events=950).
2. Fit MinMax scaling on the original training cohort for continuous variables.
3. Hold the previously selected GBDT / LightGBM hyperparameters fixed.
4. Fit the uncalibrated/base estimator on the complete original training cohort
   and calculate the apparent training AUC.
5. For each patient-level bootstrap resample of the training cohort:
   a. fit a fresh clone of the fixed-hyperparameter base estimator to the
      bootstrap sample;
   b. calculate AUC in that same bootstrap sample;
   c. apply that bootstrap-fitted estimator to the full original training cohort
      and calculate AUC there;
   d. optimism = AUC_bootstrap_sample - AUC_original_training.
6. Corrected AUC for a bootstrap replicate = original apparent AUC - optimism.
7. Report mean optimism, sample SD of the bootstrap distribution (the archived
   table labels this quantity "SE"), percentile 95% intervals, and the mean
   optimism-corrected AUC.

Important scope
---------------
- The analysis is retained for historical traceability only.
- It is NOT an uncertainty estimate for the current 11-predictor primary model.
- Patient-level data and predictions are not written by this script.
- The bundled JSON contains the hyperparameter values as displayed in the
  archived Supplementary Table S2. Some continuous values were rounded in that
  table, so very small numerical differences from the archived S14 values can
  occur unless a full-precision archived best_params_ export is supplied.

Example
-------
python historical_9predictor_bootstrap_optimism.py \
    --input /path/to/authorized/analysis_ready.csv \
    --bootstrap 1000
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

try:
    import lightgbm
    from lightgbm import LGBMClassifier
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "lightgbm is required for the historical S14 reconstruction. "
        "Install the repository environment first."
    ) from exc


HERE = Path(__file__).resolve().parent
DEFAULT_PARAMS = HERE / "historical_9predictor_selected_params.json"

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
]

BINARY_FEATURES = [
    "Intubation_tracheotomy",
    "Mechanical_ventilation",
    "Surgery",
]
CONTINUOUS_FEATURES = [c for c in FEATURES if c not in BINARY_FEATURES]

EXPECTED = {
    "total_n": 3368,
    "total_events": 1357,
    "train_n": 2357,
    "train_events": 950,
    "test_n": 1011,
    "test_events": 407,
}

# Archived Table S14 values. These are QA anchors only; they are never used to
# generate predictions, optimism estimates, or confidence intervals.
S14_QA_ANCHORS = {
    "GBDT": {
        "Apparent_AUC": 0.876,
        "Mean_optimism": 0.035,
        "Optimism_SD_archived_label_SE": 0.006,
        "Optimism_CI_low": 0.023,
        "Optimism_CI_high": 0.046,
        "Corrected_AUC": 0.841,
        "Corrected_SD_archived_label_SE": 0.006,
        "Corrected_CI_low": 0.830,
        "Corrected_CI_high": 0.853,
    },
    "LightGBM": {
        "Apparent_AUC": 0.863,
        "Mean_optimism": 0.029,
        "Optimism_SD_archived_label_SE": 0.007,
        "Optimism_CI_low": 0.016,
        "Optimism_CI_high": 0.041,
        "Corrected_AUC": 0.835,
        "Corrected_SD_archived_label_SE": 0.007,
        "Corrected_CI_low": 0.823,
        "Corrected_CI_high": 0.848,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct historical nine-predictor bootstrap optimism (Supplementary Table S14)."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help=(
            "Authorized analysis-ready patient-level CSV. If omitted, the script "
            "uses AIS_ICU_DATA_FILE, or looks in AIS_ICU_DATA_DIR for "
            "BSAfree_feature_selection_input_exactsplit.csv / analysis_ready_exactsplit.csv."
        ),
    )
    parser.add_argument(
        "--params-json",
        type=Path,
        default=DEFAULT_PARAMS,
        help="Fixed historical hyperparameters (default: bundled displayed Supplementary Table S2 values).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "historical_9predictor_S14",
        help="Local aggregate-output directory (default: outputs/historical_9predictor_S14).",
    )
    parser.add_argument("--bootstrap", type=int, default=1000, help="Number of bootstrap resamples (default: 1000).")
    parser.add_argument("--seed", type=int, default=42, help="Bootstrap and split seed (default: 42).")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["GBDT", "LightGBM"],
        default=["GBDT", "LightGBM"],
        help="Models to run (default: both GBDT and LightGBM).",
    )
    parser.add_argument(
        "--qa-tolerance",
        type=float,
        default=0.004,
        help=(
            "Absolute tolerance used only for comparison with archived 3-decimal S14 anchors. "
            "A modest tolerance is intentional because the archived selected continuous hyperparameters "
            "were displayed with rounding and package versions can differ."
        ),
    )
    return parser.parse_args()


def locate_input(cli_path: Path | None) -> Path:
    if cli_path is not None:
        p = cli_path.expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(p)
        return p

    env_file = os.environ.get("AIS_ICU_DATA_FILE")
    if env_file:
        p = Path(env_file).expanduser().resolve()
        if p.exists():
            return p
        raise FileNotFoundError(f"AIS_ICU_DATA_FILE does not exist: {p}")

    env_dir = os.environ.get("AIS_ICU_DATA_DIR")
    if env_dir:
        root = Path(env_dir).expanduser().resolve()
        candidates = [
            root / "BSAfree_feature_selection_input_exactsplit.csv",
            root / "analysis_ready_exactsplit.csv",
        ]
        for p in candidates:
            if p.exists():
                return p

    raise FileNotFoundError(
        "No authorized analysis-ready CSV was found. Pass --input, set AIS_ICU_DATA_FILE, "
        "or set AIS_ICU_DATA_DIR. Patient-level data are intentionally not included in the repository."
    )


def read_input(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = df.columns.astype(str).str.replace("\ufeff", "", regex=False).str.strip()
    return df


def normalize_outcome(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        out = pd.to_numeric(s, errors="coerce")
    else:
        mapping = {
            "No": 0, "no": 0, "NO": 0, "0": 0, "0.0": 0,
            "Yes": 1, "yes": 1, "YES": 1, "1": 1, "1.0": 1,
        }
        out = s.astype(str).str.strip().map(mapping)
    if out.isna().any():
        raise ValueError("Outcome contains values that cannot be mapped to 0/1.")
    out = out.astype(int)
    if not set(out.unique()).issubset({0, 1}):
        raise ValueError("Outcome must be binary 0/1.")
    return out


def split_train_test(df: pd.DataFrame, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """Use preserved Primary_split when available; otherwise reconstruct the archived 70:30 split."""
    required = [OUTCOME] + FEATURES
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    work = df.copy()
    work[OUTCOME] = normalize_outcome(work[OUTCOME])

    if len(work) != EXPECTED["total_n"] or int(work[OUTCOME].sum()) != EXPECTED["total_events"]:
        raise ValueError(
            f"Cohort anchor mismatch: n={len(work)}, events={int(work[OUTCOME].sum())}; "
            f"expected n={EXPECTED['total_n']}, events={EXPECTED['total_events']}."
        )

    if SPLIT_COL in work.columns:
        split_norm = work[SPLIT_COL].astype(str).str.strip().str.lower()
        train = work.loc[split_norm == "train"].copy().reset_index(drop=True)
        test = work.loc[split_norm == "test"].copy().reset_index(drop=True)
        source = "preserved Primary_split"
    else:
        # Historical archive fallback. This only reproduces the original patient identities
        # when the authorized analysis-ready file preserves the original row order.
        idx = np.arange(len(work))
        tr_idx, te_idx = train_test_split(
            idx,
            test_size=0.30,
            random_state=seed,
            stratify=work[OUTCOME].to_numpy(),
        )
        train = work.iloc[tr_idx].copy().reset_index(drop=True)
        test = work.iloc[te_idx].copy().reset_index(drop=True)
        source = "reconstructed 70:30 stratified split (random_state=42)"

    anchors = [
        ("train_n", len(train)),
        ("train_events", int(train[OUTCOME].sum())),
        ("test_n", len(test)),
        ("test_events", int(test[OUTCOME].sum())),
    ]
    bad = [(k, v, EXPECTED[k]) for k, v in anchors if v != EXPECTED[k]]
    if bad:
        raise ValueError(f"Fixed split anchor mismatch: {bad}")

    if train[FEATURES].isna().any().any() or test[FEATURES].isna().any().any():
        train_miss = train[FEATURES].isna().sum()
        test_miss = test[FEATURES].isna().sum()
        raise ValueError(
            "Historical S14 requires the authorized analysis-ready (already completed) matrix. "
            f"Missing values remain. Train={train_miss[train_miss > 0].to_dict()}, "
            f"Test={test_miss[test_miss > 0].to_dict()}"
        )

    return train, test, source


def scale_training(train: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    x = train[FEATURES].copy()
    y = train[OUTCOME].astype(int).to_numpy()
    scaler = MinMaxScaler()
    x.loc[:, CONTINUOUS_FEATURES] = scaler.fit_transform(x[CONTINUOUS_FEATURES])
    return x.to_numpy(dtype=float), y, scaler


def load_params(path: Path) -> Dict[str, dict]:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    missing = [m for m in ("GBDT", "LightGBM") if m not in payload]
    if missing:
        raise ValueError(f"Parameter JSON is missing model blocks: {missing}")
    return payload


def build_model(model_name: str, params: dict):
    if model_name == "GBDT":
        return GradientBoostingClassifier(random_state=42, **params)
    if model_name == "LightGBM":
        return LGBMClassifier(
            random_state=42,
            verbosity=-1,
            n_jobs=1,
            force_col_wise=True,
            **params,
        )
    raise ValueError(model_name)


def percentile_ci(values: Iterable[float]) -> Tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan, np.nan
    lo, hi = np.percentile(arr, [2.5, 97.5])
    return float(lo), float(hi)


def bootstrap_optimism(
    base_model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> Tuple[dict, pd.DataFrame]:
    if n_bootstrap < 1:
        raise ValueError("--bootstrap must be >= 1")

    # Original apparent fit.
    original_model = clone(base_model)
    original_model.fit(x_train, y_train)
    apparent_auc = float(roc_auc_score(y_train, original_model.predict_proba(x_train)[:, 1]))

    rng = np.random.RandomState(seed)
    rows = []
    n = len(y_train)

    for b in range(1, n_bootstrap + 1):
        idx = rng.randint(0, n, size=n)
        y_boot = y_train[idx]
        if np.unique(y_boot).size < 2:
            continue

        model_b = clone(base_model)
        model_b.fit(x_train[idx], y_boot)

        p_boot = model_b.predict_proba(x_train[idx])[:, 1]
        p_orig = model_b.predict_proba(x_train)[:, 1]

        auc_boot = float(roc_auc_score(y_boot, p_boot))
        auc_orig = float(roc_auc_score(y_train, p_orig))
        optimism = auc_boot - auc_orig
        corrected = apparent_auc - optimism

        rows.append({
            "bootstrap_rep": b,
            "auc_bootstrap_sample": auc_boot,
            "auc_original_training": auc_orig,
            "optimism": optimism,
            "corrected_auc": corrected,
        })

    reps = pd.DataFrame(rows)
    if len(reps) == 0:
        raise RuntimeError("No valid bootstrap resamples were generated.")

    optimism = reps["optimism"].to_numpy(dtype=float)
    corrected = reps["corrected_auc"].to_numpy(dtype=float)
    opt_lo, opt_hi = percentile_ci(optimism)
    cor_lo, cor_hi = percentile_ci(corrected)

    # The archived S14 column was labeled "SE". The reconstructed procedure uses
    # the sample SD of the bootstrap distribution, which is numerically on the
    # archived scale; dividing by sqrt(B) would estimate the Monte Carlo SE of the
    # bootstrap mean and would be a different quantity.
    opt_sd = float(np.std(optimism, ddof=1)) if len(optimism) > 1 else np.nan
    cor_sd = float(np.std(corrected, ddof=1)) if len(corrected) > 1 else np.nan

    summary = {
        "Apparent_AUC": apparent_auc,
        "Mean_optimism": float(np.mean(optimism)),
        "Optimism_SD_archived_label_SE": opt_sd,
        "Optimism_CI_low": opt_lo,
        "Optimism_CI_high": opt_hi,
        "Corrected_AUC": float(apparent_auc - np.mean(optimism)),
        "Corrected_SD_archived_label_SE": cor_sd,
        "Corrected_CI_low": cor_lo,
        "Corrected_CI_high": cor_hi,
        "N_bootstrap_valid": int(len(reps)),
    }
    return summary, reps


def qa_against_archived(model_name: str, summary: dict, tolerance: float) -> dict:
    anchor = S14_QA_ANCHORS[model_name]
    diffs = {k: float(summary[k] - v) for k, v in anchor.items()}
    passed = all(abs(v) <= tolerance for v in diffs.values())
    return {
        "QA_status": "PASS" if passed else "REVIEW",
        "QA_tolerance": tolerance,
        "QA_max_abs_difference": float(max(abs(v) for v in diffs.values())),
        "QA_differences_json": json.dumps(diffs, ensure_ascii=False),
    }


def main() -> None:
    args = parse_args()
    input_path = locate_input(args.input)
    params_path = args.params_json.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print("Historical nine-predictor fixed-hyperparameter bootstrap optimism reconstruction")
    print("Supplementary Table S14 traceability analysis")
    print("=" * 88)
    print(f"Input: {input_path}")
    print(f"Parameters: {params_path}")
    print(f"Bootstrap resamples requested: {args.bootstrap}")

    df = read_input(input_path)
    train, test, split_source = split_train_test(df, seed=args.seed)
    x_train, y_train, _ = scale_training(train)
    params_payload = load_params(params_path)

    print(f"Split source: {split_source}")
    print(f"Train n={len(train)}, events={int(y_train.sum())}")
    print(f"Test  n={len(test)}, events={int(test[OUTCOME].sum())} (not used in S14 optimism fitting)")

    summary_rows = []
    # Replicate-level outputs contain only aggregate AUC values by bootstrap index,
    # never patient identifiers or patient-level predictions.
    replicate_outputs = []

    for model_name in args.models:
        print(f"\nRunning {model_name} ...")
        base_model = build_model(model_name, params_payload[model_name])
        summary, reps = bootstrap_optimism(
            base_model,
            x_train=x_train,
            y_train=y_train,
            n_bootstrap=args.bootstrap,
            seed=args.seed,
        )
        qa = qa_against_archived(model_name, summary, tolerance=args.qa_tolerance)
        summary_rows.append({"Model": model_name, **summary, **qa})
        reps.insert(0, "Model", model_name)
        replicate_outputs.append(reps)

        print(
            f"  apparent={summary['Apparent_AUC']:.6f} | "
            f"mean optimism={summary['Mean_optimism']:.6f} | "
            f"corrected={summary['Corrected_AUC']:.6f} | "
            f"QA={qa['QA_status']}"
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / "Table_S14_historical_9predictor_bootstrap_optimism_reconstructed.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    # Useful for auditing the bootstrap arithmetic without revealing patient-level data.
    reps_path = output_dir / "Table_S14_bootstrap_replicate_aggregate_AUCs.csv"
    pd.concat(replicate_outputs, ignore_index=True).to_csv(reps_path, index=False, encoding="utf-8-sig")

    metadata = {
        "script": Path(__file__).name,
        "status": "reconstruction_not_original_generation_script",
        "input_basename": input_path.name,
        "split_source": split_source,
        "features": FEATURES,
        "continuous_features_scaled": CONTINUOUS_FEATURES,
        "binary_features_unscaled": BINARY_FEATURES,
        "train_n": int(len(train)),
        "train_events": int(y_train.sum()),
        "test_n": int(len(test)),
        "test_events": int(test[OUTCOME].sum()),
        "bootstrap_requested": int(args.bootstrap),
        "bootstrap_seed": int(args.seed),
        "params_file": params_path.name,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "lightgbm": lightgbm.__version__,
        "note": (
            "Bundled historical continuous hyperparameters are retained at the precision displayed "
            "in the archived Supplementary Table S2. Small numerical differences from S14 can occur "
            "if full-precision best_params_ values or original package versions are unavailable."
        ),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print("\nSaved aggregate outputs:")
    print(f"  {summary_path}")
    print(f"  {reps_path}")
    print(f"  {output_dir / 'run_metadata.json'}")
    print("\nNo patient-level output was written.")


if __name__ == "__main__":
    main()
