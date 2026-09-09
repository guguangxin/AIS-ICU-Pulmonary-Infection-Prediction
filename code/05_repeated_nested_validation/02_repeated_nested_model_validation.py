#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Repeated nested resampling - stage 2: LR/GBDT model validation.

This is the portable public-repository version of the Reviewer 6 M5 validation
script. It consumes stage-1 outer-fold assignments and BSA-free LASSO/Boruta
feature selections.

Repeated inside each OUTER training fold
----------------------------------------
- selected feature set from stage-1 BSA-free LASSO/Boruta;
- training-only MinMax scaling for continuous selected variables;
- Bayesian hyperparameter optimization (inner 5-fold CV);
- 5-fold Platt calibration within outer training;
- Youden threshold from cross-fitted calibrated outer-training probabilities;
- evaluation in the untouched outer validation fold.

Models
------
- Logistic Regression (transparent benchmark)
- Gradient Boosting Decision Tree (representative leading model)

IMPORTANT LIMITATION
--------------------
The archived revision input has already undergone the original imputation and
winsorization workflow. Those preprocessing stages cannot be re-estimated inside
outer folds and are therefore NOT claimed as part of this repeated nested
resampling. This validates the recoverable post-preprocessing development
pipeline, not the complete raw-data pipeline.

Example
-------
python 02_repeated_nested_model_validation.py \
  --input /path/to/restricted/analysis_ready.csv \
  --stage1-dir outputs/repeated_nested \
  --output-dir outputs/repeated_nested
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import time
import warnings

import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.optimize import minimize

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss, roc_curve,
    confusion_matrix, accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score, matthews_corrcoef,
)
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

OUTCOME = "Pulmonary_infection"
ID_COL = "Study_row_id"
EXPECTED_N = 3368
EXPECTED_EVENTS = 1357
EXPECTED_OUTER_FOLDS = 25

INNER_FOLDS = 5
CALIBRATION_FOLDS = 5
THRESHOLD_OUTER_FOLDS = 5
RANDOM_STATE = 20260901
N_ITER_LR = 25
N_ITER_GBDT = 25
N_JOBS = 2
MODELS_TO_RUN = ["LR", "GBDT"]

MODEL_CONFIG = {
    "LR": {
        "model": LogisticRegression(random_state=42, max_iter=3000),
        "params": {
            "model__C": Real(0.001, 100, prior="log-uniform"),
            "model__penalty": Categorical(["l2"]),
            "model__solver": Categorical(["lbfgs", "saga", "newton-cg"]),
        },
        "n_iter": N_ITER_LR,
    },
    "GBDT": {
        "model": GradientBoostingClassifier(random_state=42),
        "params": {
            "model__n_estimators": Integer(100, 400),
            "model__max_depth": Integer(3, 7),
            "model__learning_rate": Real(0.01, 0.15, prior="log-uniform"),
            "model__min_samples_split": Integer(5, 30),
            "model__min_samples_leaf": Integer(5, 20),
            "model__subsample": Real(0.7, 0.9),
            "model__max_features": Categorical(["sqrt", "log2"]),
        },
        "n_iter": N_ITER_GBDT,
    },
}


def read_csv_flexible(path: Path) -> pd.DataFrame:
    last = None
    for enc in ("utf-8-sig", "utf-8", "gb18030", "gbk", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError as exc:
            last = exc
    raise last or RuntimeError(f"Unable to read {path}")


def calibration_intercept_slope(y_true, y_prob, eps=1e-6):
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(y_prob, dtype=float), eps, 1 - eps)
    lp = logit(p)

    try:
        lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000)
    except (TypeError, ValueError):
        lr = LogisticRegression(penalty="l2", C=1e12, solver="lbfgs", max_iter=2000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lr.fit(lp.reshape(-1, 1), y)
    slope = float(lr.coef_[0, 0])

    def neg_ll(b):
        eta = lp + b[0]
        log_p = -np.logaddexp(0, -eta)
        log_1mp = -np.logaddexp(0, eta)
        return -(y * log_p + (1 - y) * log_1mp).sum()

    res = minimize(neg_ll, x0=[0.0], method="L-BFGS-B")
    return float(res.x[0]), slope


def parse_vars(s):
    if pd.isna(s) or str(s).strip() == "":
        return []
    return [x for x in str(s).split(";") if x]


def classify_features(df, features):
    continuous, categorical = [], []
    for c in features:
        # The archived model matrix contains binary/categorical fields coded with
        # a small number of levels. Continuous selected fields are scaled only.
        if df[c].dtype == object or df[c].nunique(dropna=True) <= 5:
            categorical.append(c)
        else:
            continuous.append(c)
    return continuous, categorical


def make_pipeline(train_df, features, base_model):
    continuous, categorical = classify_features(train_df, features)
    transformers = []
    if continuous:
        transformers.append(("cont", MinMaxScaler(), continuous))
    if categorical:
        transformers.append(("cat", "passthrough", categorical))
    pre = ColumnTransformer(transformers=transformers, remainder="drop")
    pipe = Pipeline([
        ("prep", pre),
        ("model", clone(base_model)),
    ])
    return pipe, continuous, categorical


def make_calibrator(estimator, seed):
    cv = StratifiedKFold(n_splits=CALIBRATION_FOLDS, shuffle=True, random_state=seed)
    try:
        return CalibratedClassifierCV(estimator=clone(estimator), method="sigmoid", cv=cv)
    except TypeError:  # older sklearn compatibility
        return CalibratedClassifierCV(base_estimator=clone(estimator), method="sigmoid", cv=cv)


def youden_threshold(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    j = tpr - fpr
    k = int(np.nanargmax(j))
    return float(thr[k])


def threshold_metrics(y, p, threshold):
    pred = (np.asarray(p) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    npv = tn / (tn + fn) if (tn + fn) else np.nan
    return {
        "Threshold": threshold,
        "Sensitivity": recall_score(y, pred, zero_division=0),
        "Specificity": spec,
        "Precision": precision_score(y, pred, zero_division=0),
        "NPV": npv,
        "F1": f1_score(y, pred, zero_division=0),
        "Accuracy": accuracy_score(y, pred),
        "Balanced_accuracy": balanced_accuracy_score(y, pred),
        "MCC": matthews_corrcoef(y, pred),
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
    }


def evaluate(y, p):
    inter, slope = calibration_intercept_slope(y, p)
    return {
        "AUC": roc_auc_score(y, p),
        "AP": average_precision_score(y, p),
        "Brier": brier_score_loss(y, p),
        "Calibration_intercept": inter,
        "Calibration_slope": slope,
    }


def summarize_metrics(fold_df):
    metrics = [
        "AUC", "AP", "Brier", "Calibration_intercept", "Calibration_slope",
        "Threshold", "Sensitivity", "Specificity", "Precision", "NPV", "F1",
        "Accuracy", "Balanced_accuracy", "MCC",
    ]
    rows = []
    for model, g in fold_df.groupby("Model"):
        for m in metrics:
            x = pd.to_numeric(g[m], errors="coerce").dropna().to_numpy()
            if len(x) == 0:
                continue
            rows.append({
                "Model": model,
                "Metric": m,
                "N_outer_folds": len(x),
                "Mean": float(np.mean(x)),
                "SD": float(np.std(x, ddof=1)) if len(x) > 1 else np.nan,
                "Median": float(np.median(x)),
                "P2_5": float(np.percentile(x, 2.5)),
                "P97_5": float(np.percentile(x, 97.5)),
                "Min": float(np.min(x)),
                "Max": float(np.max(x)),
            })
    return pd.DataFrame(rows)


def validate_inputs(df, assign, sel):
    df.columns = df.columns.astype(str).str.replace("\ufeff", "", regex=False).str.strip()
    if OUTCOME not in df or ID_COL not in df:
        raise ValueError(f"Input must include {ID_COL} and {OUTCOME}")
    df[OUTCOME] = pd.to_numeric(df[OUTCOME], errors="raise").astype(int)

    if len(df) != EXPECTED_N or int(df[OUTCOME].sum()) != EXPECTED_EVENTS:
        raise ValueError(
            f"Input cohort mismatch: n={len(df)}, events={int(df[OUTCOME].sum())}; "
            f"expected {EXPECTED_N}/{EXPECTED_EVENTS}."
        )
    if df[ID_COL].duplicated().any():
        raise ValueError(f"Duplicate {ID_COL} values.")
    if df.isna().sum().sum() != 0:
        raise ValueError("Unexpected missing values: archived analysis-ready input should be complete.")

    required_assign = {ID_COL, "Repeat", "Fold"}
    required_sel = {"Repeat", "Fold", "Final_vars"}
    if not required_assign.issubset(assign.columns):
        raise ValueError(f"Outer assignment file missing {sorted(required_assign - set(assign.columns))}")
    if not required_sel.issubset(sel.columns):
        raise ValueError(f"Feature-selection file missing {sorted(required_sel - set(sel.columns))}")
    if len(sel) != EXPECTED_OUTER_FOLDS:
        raise ValueError(f"Expected 25 outer feature-selection rows; observed {len(sel)}")

    id_set = set(df[ID_COL])
    if not set(assign[ID_COL]).issubset(id_set):
        raise ValueError("Stage-1 assignment contains Study_row_id values absent from input.")

    # Each patient should be in validation once in each of the five repeats.
    counts = assign.groupby(ID_COL).size()
    if len(counts) != len(df) or not (counts == 5).all():
        raise ValueError("Outer assignment integrity failure: each patient must appear five times (once per repeat).")

    return df


def main():
    parser = argparse.ArgumentParser(description="Repeated nested LR/GBDT outer-fold validation.")
    parser.add_argument("--input", required=True, type=Path,
                        help="Restricted complete analysis-ready cohort CSV.")
    parser.add_argument("--stage1-dir", required=True, type=Path,
                        help="Directory containing 00_outer_fold_assignments.csv and 01_nested_outer_fold_feature_selection.csv.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory; defaults to --stage1-dir.")
    parser.add_argument("--n-jobs", type=int, default=N_JOBS)
    args = parser.parse_args()

    out_dir = args.output_dir or args.stage1_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    assign_file = args.stage1_dir / "00_outer_fold_assignments.csv"
    select_file = args.stage1_dir / "01_nested_outer_fold_feature_selection.csv"
    for p in (args.input, assign_file, select_file):
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    df = read_csv_flexible(args.input)
    assign = read_csv_flexible(assign_file)
    sel = read_csv_flexible(select_file)
    df = validate_inputs(df, assign, sel)

    print("=" * 90)
    print("Repeated nested validation of recoverable post-preprocessing pipeline")
    print("=" * 90)
    print("Cohort:", len(df), "events:", int(df[OUTCOME].sum()))
    print("Outer folds:", len(sel), "| Models:", MODELS_TO_RUN)
    print("NOTE: raw imputation/winsorization cannot be repeated from this archived dataset.")

    progress_file = out_dir / "03_nested_LR_GBDT_fold_metrics_PROGRESS.csv"
    pred_file = out_dir / "06_nested_outer_validation_predictions.csv"

    if progress_file.exists():
        old = read_csv_flexible(progress_file)
        fold_rows = old.to_dict("records")
        completed = set(zip(old["Repeat"].astype(int), old["Fold"].astype(int), old["Model"]))
    else:
        fold_rows = []
        completed = set()

    if pred_file.exists():
        old_pred = read_csv_flexible(pred_file)
        pred_rows = old_pred.to_dict("records")
    else:
        pred_rows = []

    for _, sr in sel.sort_values(["Repeat", "Fold"]).iterrows():
        r = int(sr["Repeat"])
        f = int(sr["Fold"])
        features = parse_vars(sr["Final_vars"])
        if len(features) < 3:
            raise ValueError(f"Too few selected features at repeat={r}, fold={f}: {features}")
        missing_features = sorted(set(features) - set(df.columns))
        if missing_features:
            raise ValueError(f"Stage-1 selected fields absent from model input: {missing_features}")

        val_ids = set(assign.loc[(assign["Repeat"] == r) & (assign["Fold"] == f), ID_COL])
        tr = df.loc[~df[ID_COL].isin(val_ids)].copy()
        va = df.loc[df[ID_COL].isin(val_ids)].copy()

        ytr = tr[OUTCOME].to_numpy(dtype=int)
        yva = va[OUTCOME].to_numpy(dtype=int)
        Xtr = tr[features].copy()
        Xva = va[features].copy()

        print("\n" + "=" * 90)
        print(
            f"Repeat {r} Fold {f} | train={len(tr)} events={ytr.sum()} | "
            f"val={len(va)} events={yva.sum()}"
        )
        print(f"Selected {len(features)} features: {features}")

        for model_name in MODELS_TO_RUN:
            key = (r, f, model_name)
            if key in completed:
                print("Skip completed:", key)
                continue

            cfg = MODEL_CONFIG[model_name]
            seed = RANDOM_STATE + r * 100 + f
            base_pipe, continuous, categorical = make_pipeline(Xtr, features, cfg["model"])

            inner_cv = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=seed)
            t0 = time.time()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                search = BayesSearchCV(
                    base_pipe,
                    cfg["params"],
                    n_iter=cfg["n_iter"],
                    cv=inner_cv,
                    scoring="roc_auc",
                    random_state=seed,
                    n_jobs=args.n_jobs,
                    refit=True,
                    verbose=0,
                )
                search.fit(Xtr, ytr)

            best_pipe = search.best_estimator_

            # Threshold derived from cross-fitted CALIBRATED probabilities in OUTER TRAIN only.
            # Hyperparameters remain fixed at the full outer-training search result, matching
            # the manuscript's stated threshold-derivation design.
            threshold_cv = StratifiedKFold(
                n_splits=THRESHOLD_OUTER_FOLDS, shuffle=True, random_state=seed + 10000
            )
            calibrated_template = make_calibrator(best_pipe, seed + 20000)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                oof_prob = cross_val_predict(
                    calibrated_template,
                    Xtr,
                    ytr,
                    cv=threshold_cv,
                    method="predict_proba",
                    n_jobs=1,
                )[:, 1]
            threshold = youden_threshold(ytr, oof_prob)

            # Final outer-training calibrated model -> untouched outer validation.
            final_cal = make_calibrator(best_pipe, seed + 30000)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                final_cal.fit(Xtr, ytr)
            pva = final_cal.predict_proba(Xva)[:, 1]

            perf = evaluate(yva, pva)
            th = threshold_metrics(yva, pva, threshold)
            row = {
                "Repeat": r,
                "Fold": f,
                "Model": model_name,
                "N_train": len(tr),
                "Events_train": int(ytr.sum()),
                "N_validation": len(va),
                "Events_validation": int(yva.sum()),
                "N_selected_features": len(features),
                "Selected_features": ";".join(features),
                "Inner_CV_best_AUC": float(search.best_score_),
                "Continuous_features": ";".join(continuous),
                "Categorical_features": ";".join(categorical),
                "Best_params": json.dumps(
                    {k: (v.item() if hasattr(v, "item") else v) for k, v in search.best_params_.items()},
                    ensure_ascii=False,
                ),
                **perf,
                **th,
                "Runtime_minutes": (time.time() - t0) / 60.0,
            }
            fold_rows.append(row)
            pd.DataFrame(fold_rows).to_csv(progress_file, index=False, encoding="utf-8-sig")

            # Remove any stale rows for this tuple before appending, so resume is safe.
            pred_rows = [x for x in pred_rows
                         if not (int(x["Repeat"]) == r and int(x["Fold"]) == f and x["Model"] == model_name)]
            for sid, yy, pp in zip(va[ID_COL], yva, pva):
                pred_rows.append({
                    ID_COL: sid,
                    "Repeat": r,
                    "Fold": f,
                    "Model": model_name,
                    "True_label": int(yy),
                    "Predicted_probability": float(pp),
                    "Threshold": threshold,
                    "Predicted_class": int(pp >= threshold),
                    "N_selected_features": len(features),
                    "Selected_features": ";".join(features),
                })
            pd.DataFrame(pred_rows).to_csv(pred_file, index=False, encoding="utf-8-sig")

            print(
                f"{model_name}: outer AUC={perf['AUC']:.3f}, AP={perf['AP']:.3f}, "
                f"Brier={perf['Brier']:.3f}, intercept={perf['Calibration_intercept']:+.3f}, "
                f"slope={perf['Calibration_slope']:.3f}, threshold={threshold:.3f}, "
                f"sens={th['Sensitivity']:.3f}, spec={th['Specificity']:.3f}, "
                f"time={row['Runtime_minutes']:.1f} min"
            )

    fold_df = pd.DataFrame(fold_rows).sort_values(["Model", "Repeat", "Fold"])
    fold_file = out_dir / "03_nested_LR_GBDT_fold_metrics.csv"
    fold_df.to_csv(fold_file, index=False, encoding="utf-8-sig")

    # Exact completion check: 25 folds x 2 models = 50 rows.
    expected_rows = EXPECTED_OUTER_FOLDS * len(MODELS_TO_RUN)
    unique_tuples = fold_df[["Repeat", "Fold", "Model"]].drop_duplicates()
    if len(unique_tuples) != expected_rows:
        warnings.warn(
            f"Nested run is not complete: {len(unique_tuples)}/{expected_rows} fold-model tuples present."
        )

    summary = summarize_metrics(fold_df)
    summary_file = out_dir / "04_nested_LR_GBDT_resampling_summary.csv"
    summary.to_csv(summary_file, index=False, encoding="utf-8-sig")

    feature_summary = (
        fold_df[["Repeat", "Fold", "N_selected_features", "Selected_features"]]
        .drop_duplicates()
        .sort_values(["Repeat", "Fold"])
    )
    feature_summary.to_csv(
        out_dir / "05_nested_selected_feature_sets_used_for_modeling.csv",
        index=False,
        encoding="utf-8-sig",
    )

    readme = f"""Repeated nested validation of the recoverable post-preprocessing pipeline
====================================================================

Outer design
------------
- Stage-1 feature selection: 5 repeats x 5 folds = 25 outer validation folds.
- BSA-free LASSO/Boruta feature selection performed within each outer training fold.

Repeated modeling stages inside each outer training fold
--------------------------------------------------------
- training-only MinMax scaling for continuous selected variables;
- inner {INNER_FOLDS}-fold Bayesian hyperparameter tuning;
- {CALIBRATION_FOLDS}-fold Platt calibration within outer training;
- cross-fitted calibrated Youden-threshold derivation within outer training;
- final evaluation in the untouched outer validation fold.

Models
------
- LR: transparent benchmark.
- GBDT: representative leading primary model.

Critical limitation
-------------------
The archived revision input is already imputed and winsorized. Raw pre-imputation values and uncapped original values are unavailable in the revision archive. Therefore imputation models and winsorization limits cannot be re-estimated inside each outer training fold. This analysis must be described as repeated nested validation of the recoverable post-preprocessing development pipeline, not the complete raw-data pipeline.
"""
    (out_dir / "README_stage2.txt").write_text(readme, encoding="utf-8")

    print("\nDONE")
    print("Saved:")
    print(" ", fold_file)
    print(" ", summary_file)
    print(" ", pred_file)


if __name__ == "__main__":
    main()
