#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reconstruct the four post hoc parsimonious logistic-regression comparators.

This script is intended for the public reproducibility repository. It does not
contain or download patient-level data. Supply the locally governed analysis-ready
cohort with --data or set AIS_ICU_DATA_DIR.

Comparator specifications
-------------------------
1) Airway2_MV_Intubation: Mechanical ventilation + intubation/tracheotomy
2) NLR1: neutrophil-to-lymphocyte ratio (NEU / LYM)
3) AgeSex2_partial_score_components: age + sex (component-level floor only)
4) NEU_LYM_MV3: NEU + LYM + mechanical ventilation

Workflow (matches the revised manuscript)
-----------------------------------------
- Preserve the fixed original 70/30 internal split. If Primary_split is absent,
  recreate the archived split with stratified train_test_split, random_state=42.
- Min-max scale continuous predictors using training-set parameters only.
- L2 logistic regression with C selected by 10-fold stratified Bayesian search.
- 10-fold sigmoid/Platt probability calibration in the training set.
- Evaluate the same fixed 1,011-patient internal test partition.

The script writes patient-level predictions only into the local output directory.
Those files are governed analysis outputs and should NOT be committed to GitHub.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler

try:
    from skopt import BayesSearchCV
    from skopt.space import Categorical, Real
except Exception as exc:  # pragma: no cover - dependency check happens at runtime
    BayesSearchCV = None
    Categorical = None
    Real = None
    SKOPT_IMPORT_ERROR = exc
else:
    SKOPT_IMPORT_ERROR = None

RANDOM_STATE = 42
OUTCOME = "Pulmonary_infection"
ID_COL = "Study_row_id"
SPLIT_COL = "Primary_split"
TUNING_CV = 10
CALIBRATION_CV = 10
N_ITER = 40

EXPECTED_N = 3368
EXPECTED_EVENTS = 1357
EXPECTED_TRAIN_N = 2357
EXPECTED_TRAIN_EVENTS = 950
EXPECTED_TEST_N = 1011
EXPECTED_TEST_EVENTS = 407

# Archived fixed-test point estimates reported in the revised manuscript.
# These are used only as a QA target after reconstruction; they are not used
# during fitting or hyperparameter selection.
ARCHIVED_EXPECTED = {
    "Airway2_MV_Intubation": {"AUC": 0.6265213889, "AP": 0.5355424604, "Brier": 0.2099206171},
    "NLR1": {"AUC": 0.7567425192, "AP": 0.6568805336, "Brier": 0.2017637789},
    "AgeSex2_partial_score_components": {"AUC": 0.6346205477, "AP": 0.5444484800, "Brier": 0.2285962283},
    "NEU_LYM_MV3": {"AUC": 0.7825186716, "AP": 0.7138213242, "Brier": 0.1873630913},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Path to the governed analysis-ready CSV. If omitted, AIS_ICU_DATA_DIR is searched.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/parsimonious_comparator_training"),
        help="Local output directory (ignored by the repository .gitignore).",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=2,
        help="Parallel jobs for Bayesian search. Default: 2.",
    )
    parser.add_argument(
        "--prediction-audit",
        type=Path,
        default=None,
        help=(
            "Optional archived combined test-prediction CSV. If supplied, the script "
            "compares reconstructed probabilities with the archived probabilities."
        ),
    )
    return parser.parse_args()


def resolve_data_path(arg_path: Path | None) -> Path:
    if arg_path is not None:
        return arg_path.expanduser().resolve()

    env = os.environ.get("AIS_ICU_DATA_DIR", "").strip()
    if not env:
        raise FileNotFoundError(
            "No data path supplied. Use --data /path/to/file.csv or set AIS_ICU_DATA_DIR."
        )

    root = Path(env).expanduser().resolve()
    candidates = [
        root / "BSAfree_feature_selection_input_exactsplit.csv",
        root / "英文列名准备变量筛选_HAPVAP复核_待填写.csv",
        root / "analysis_ready.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    csvs = sorted(root.glob("*.csv"))
    if len(csvs) == 1:
        return csvs[0]
    raise FileNotFoundError(
        f"Could not identify the analysis-ready CSV under {root}. Use --data explicitly."
    )


def read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = df.columns.astype(str).str.replace("\ufeff", "", regex=False).str.strip()
    return df


def fixed_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    required = {ID_COL, OUTCOME}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if len(df) != EXPECTED_N or int(df[OUTCOME].sum()) != EXPECTED_EVENTS:
        raise ValueError(
            f"Unexpected cohort identity: n={len(df)}, events={int(df[OUTCOME].sum())}; "
            f"expected n={EXPECTED_N}, events={EXPECTED_EVENTS}."
        )

    if SPLIT_COL in df.columns:
        split = df[SPLIT_COL].astype(str).str.strip().str.lower()
        train_mask = split.isin({"train", "training", "development", "dev", "0"})
        test_mask = split.isin({"test", "testing", "validation", "internal_test", "1"})
        if train_mask.sum() + test_mask.sum() == len(df) and train_mask.sum() > 0 and test_mask.sum() > 0:
            train = df.loc[train_mask].copy()
            test = df.loc[test_mask].copy()
        else:
            raise ValueError(
                "Primary_split exists but its values could not be interpreted as Train/Test."
            )
    else:
        idx = np.arange(len(df))
        train_idx, test_idx = train_test_split(
            idx,
            test_size=0.30,
            random_state=RANDOM_STATE,
            stratify=df[OUTCOME].astype(int).to_numpy(),
        )
        train = df.iloc[train_idx].copy()
        test = df.iloc[test_idx].copy()

    checks = (
        (len(train), int(train[OUTCOME].sum()), EXPECTED_TRAIN_N, EXPECTED_TRAIN_EVENTS, "Train"),
        (len(test), int(test[OUTCOME].sum()), EXPECTED_TEST_N, EXPECTED_TEST_EVENTS, "Test"),
    )
    for n, events, exp_n, exp_events, label in checks:
        if n != exp_n or events != exp_events:
            raise ValueError(
                f"{label} identity mismatch: n={n}, events={events}; expected n={exp_n}, events={exp_events}."
            )

    if train[ID_COL].duplicated().any() or test[ID_COL].duplicated().any():
        raise ValueError("Study_row_id must be unique within Train and Test.")

    return train.reset_index(drop=True), test.reset_index(drop=True)


def binary_or_numeric(series: pd.Series, name: str) -> pd.Series:
    """Preserve numeric coding; otherwise map a two-level categorical variable deterministically."""
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        return numeric.astype(float)

    levels = sorted(series.dropna().astype(str).unique().tolist())
    if len(levels) != 2:
        raise ValueError(f"{name} must be numeric or have exactly two observed levels; got {levels}")
    mapping = {levels[0]: 0.0, levels[1]: 1.0}
    return series.astype(str).map(mapping).astype(float)


def build_design(
    train: pd.DataFrame,
    test: pd.DataFrame,
    model_name: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    if model_name == "Airway2_MV_Intubation":
        cols = ["Mechanical_ventilation", "Intubation_tracheotomy"]
        for col in cols:
            if col not in train.columns:
                raise ValueError(f"Missing predictor: {col}")
        Xtr = pd.DataFrame({col: binary_or_numeric(train[col], col) for col in cols})
        Xte = pd.DataFrame({col: binary_or_numeric(test[col], col) for col in cols})
        continuous: List[str] = []

    elif model_name == "NLR1":
        for col in ["NEUT_abs", "LYMPH_abs"]:
            if col not in train.columns:
                raise ValueError(f"Missing predictor: {col}")
        tr_lym = pd.to_numeric(train["LYMPH_abs"], errors="raise").astype(float)
        te_lym = pd.to_numeric(test["LYMPH_abs"], errors="raise").astype(float)
        if (tr_lym <= 0).any() or (te_lym <= 0).any():
            raise ValueError("LYMPH_abs contains non-positive values; NLR cannot be reconstructed without an explicit rule.")
        Xtr = pd.DataFrame({"NLR": pd.to_numeric(train["NEUT_abs"], errors="raise").astype(float) / tr_lym})
        Xte = pd.DataFrame({"NLR": pd.to_numeric(test["NEUT_abs"], errors="raise").astype(float) / te_lym})
        continuous = ["NLR"]

    elif model_name == "AgeSex2_partial_score_components":
        for col in ["Age", "Sex"]:
            if col not in train.columns:
                raise ValueError(f"Missing predictor: {col}")
        Xtr = pd.DataFrame(
            {
                "Age": pd.to_numeric(train["Age"], errors="raise").astype(float),
                "Sex": binary_or_numeric(train["Sex"], "Sex"),
            }
        )
        Xte = pd.DataFrame(
            {
                "Age": pd.to_numeric(test["Age"], errors="raise").astype(float),
                "Sex": binary_or_numeric(test["Sex"], "Sex"),
            }
        )
        continuous = ["Age"]

    elif model_name == "NEU_LYM_MV3":
        cols = ["NEUT_abs", "LYMPH_abs", "Mechanical_ventilation"]
        for col in cols:
            if col not in train.columns:
                raise ValueError(f"Missing predictor: {col}")
        Xtr = pd.DataFrame(
            {
                "NEUT_abs": pd.to_numeric(train["NEUT_abs"], errors="raise").astype(float),
                "LYMPH_abs": pd.to_numeric(train["LYMPH_abs"], errors="raise").astype(float),
                "Mechanical_ventilation": binary_or_numeric(train["Mechanical_ventilation"], "Mechanical_ventilation"),
            }
        )
        Xte = pd.DataFrame(
            {
                "NEUT_abs": pd.to_numeric(test["NEUT_abs"], errors="raise").astype(float),
                "LYMPH_abs": pd.to_numeric(test["LYMPH_abs"], errors="raise").astype(float),
                "Mechanical_ventilation": binary_or_numeric(test["Mechanical_ventilation"], "Mechanical_ventilation"),
            }
        )
        continuous = ["NEUT_abs", "LYMPH_abs"]

    else:
        raise KeyError(model_name)

    if Xtr.isna().any().any() or Xte.isna().any().any():
        raise ValueError(f"{model_name}: missing values remain in the analysis-ready predictors.")

    return Xtr, Xte, continuous


def scale_training_only(
    Xtr: pd.DataFrame,
    Xte: pd.DataFrame,
    continuous: Iterable[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler | None]:
    Xtr_s = Xtr.copy()
    Xte_s = Xte.copy()
    continuous = list(continuous)
    if not continuous:
        return Xtr_s, Xte_s, None

    scaler = MinMaxScaler()
    Xtr_s.loc[:, continuous] = scaler.fit_transform(Xtr[continuous])
    Xte_s.loc[:, continuous] = scaler.transform(Xte[continuous])
    return Xtr_s, Xte_s, scaler


def fit_one(
    Xtr: pd.DataFrame,
    ytr: np.ndarray,
    Xte: pd.DataFrame,
    n_jobs: int,
):
    if BayesSearchCV is None:
        raise RuntimeError(
            "scikit-optimize is required for the manuscript-matched Bayesian search. "
            f"Original import error: {SKOPT_IMPORT_ERROR}"
        )

    estimator = LogisticRegression(random_state=RANDOM_STATE, max_iter=3000)
    search_space = {
        "C": Real(0.001, 100.0, prior="log-uniform"),
        "penalty": Categorical(["l2"]),
        "solver": Categorical(["lbfgs", "saga", "newton-cg"]),
    }
    inner_cv = StratifiedKFold(n_splits=TUNING_CV, shuffle=True, random_state=RANDOM_STATE)
    search = BayesSearchCV(
        estimator=estimator,
        search_spaces=search_space,
        n_iter=N_ITER,
        cv=inner_cv,
        scoring="roc_auc",
        random_state=RANDOM_STATE,
        n_jobs=n_jobs,
        refit=True,
        verbose=0,
    )
    search.fit(Xtr, ytr)

    try:
        calibrated = CalibratedClassifierCV(
            estimator=search.best_estimator_, method="sigmoid", cv=CALIBRATION_CV, n_jobs=1
        )
    except TypeError:  # compatibility with older scikit-learn
        calibrated = CalibratedClassifierCV(
            base_estimator=search.best_estimator_, method="sigmoid", cv=CALIBRATION_CV, n_jobs=1
        )
    calibrated.fit(Xtr, ytr)
    p_test = calibrated.predict_proba(Xte)[:, 1]
    return search, calibrated, p_test


def metrics(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    return {
        "AUC": float(roc_auc_score(y, p)),
        "AP": float(average_precision_score(y, p)),
        "Brier": float(brier_score_loss(y, p)),
    }


def main() -> int:
    args = parse_args()
    data_path = resolve_data_path(args.data)
    output_dir = args.output.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = read_csv(data_path)
    train, test = fixed_split(df)
    ytr = train[OUTCOME].astype(int).to_numpy()
    yte = test[OUTCOME].astype(int).to_numpy()

    model_names = [
        "Airway2_MV_Intubation",
        "NLR1",
        "AgeSex2_partial_score_components",
        "NEU_LYM_MV3",
    ]

    prediction_df = pd.DataFrame({
        ID_COL: test[ID_COL].to_numpy(),
        "True_Label": yte,
    })
    perf_rows = []
    params = {}
    audit_rows = []

    for model_name in model_names:
        print(f"\n=== {model_name} ===")
        Xtr, Xte, continuous = build_design(train, test, model_name)
        Xtr_s, Xte_s, _ = scale_training_only(Xtr, Xte, continuous)
        search, calibrated, p_test = fit_one(Xtr_s, ytr, Xte_s, args.n_jobs)
        m = metrics(yte, p_test)
        prediction_df[model_name] = p_test
        params[model_name] = {
            "best_cv_auc": float(search.best_score_),
            "best_params": {
                k: (v.item() if hasattr(v, "item") else v) for k, v in search.best_params_.items()
            },
            "continuous_scaled": list(continuous),
        }
        perf_rows.append({
            "Model": model_name,
            "N_predictors": Xtr.shape[1],
            "Train_CV_AUC": float(search.best_score_),
            **m,
        })

        target = ARCHIVED_EXPECTED[model_name]
        audit_rows.append({
            "Model": model_name,
            "Observed_AUC": m["AUC"],
            "Archived_AUC": target["AUC"],
            "Delta_AUC": m["AUC"] - target["AUC"],
            "Observed_AP": m["AP"],
            "Archived_AP": target["AP"],
            "Delta_AP": m["AP"] - target["AP"],
            "Observed_Brier": m["Brier"],
            "Archived_Brier": target["Brier"],
            "Delta_Brier": m["Brier"] - target["Brier"],
        })
        print(
            f"CV AUC={search.best_score_:.4f} | Test AUC={m['AUC']:.4f} | "
            f"AP={m['AP']:.4f} | Brier={m['Brier']:.4f}"
        )

    pd.DataFrame(perf_rows).to_csv(
        output_dir / "parsimonious_model_metrics.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(audit_rows).to_csv(
        output_dir / "parsimonious_archived_target_audit.csv", index=False, encoding="utf-8-sig"
    )
    prediction_df.to_csv(
        output_dir / "parsimonious_test_predictions_RESTRICTED.csv", index=False, encoding="utf-8-sig"
    )
    (output_dir / "parsimonious_best_params.json").write_text(
        json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if args.prediction_audit is not None:
        archived = read_csv(args.prediction_audit)
        required = {ID_COL, "True_Label", *model_names}
        missing = required - set(archived.columns)
        if missing:
            raise ValueError(f"Prediction-audit file is missing columns: {sorted(missing)}")
        merged = prediction_df.merge(
            archived[[ID_COL, "True_Label", *model_names]],
            on=ID_COL,
            how="inner",
            suffixes=("_reconstructed", "_archived"),
            validate="one_to_one",
        )
        if len(merged) != EXPECTED_TEST_N:
            raise ValueError("Prediction audit did not match all fixed-test patients.")
        probability_audit = []
        for model_name in model_names:
            a = merged[f"{model_name}_reconstructed"].to_numpy(float)
            b = merged[f"{model_name}_archived"].to_numpy(float)
            probability_audit.append({
                "Model": model_name,
                "Pearson_r": float(np.corrcoef(a, b)[0, 1]),
                "Mean_abs_probability_difference": float(np.mean(np.abs(a - b))),
                "Max_abs_probability_difference": float(np.max(np.abs(a - b))),
            })
        pd.DataFrame(probability_audit).to_csv(
            output_dir / "parsimonious_probability_audit.csv", index=False, encoding="utf-8-sig"
        )

    print(f"\nOutputs written to: {output_dir}")
    print("Do not commit patient-level prediction files from outputs/ to the public repository.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
