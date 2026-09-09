#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bootstrap calibration, ECE/MCE, and Supplementary Figure S3 generator.

This script is designed for the calibrated patient-level probabilities exported
by the current Primary11 main-analysis script.

It reproduces the definitions used in the revised supplement:
- Brier score.
- Calibration intercept (slope fixed at 1) and calibration slope.
- Hosmer-Lemeshow goodness-of-fit P value using 10 equal-sized ordered groups.
- ECE/MCE using approximately equal-frequency (quantile) risk groups.
- 1,000 patient-level bootstrap percentile intervals for ECE/MCE.
- Pointwise bootstrap calibration plot for the eight Primary11 models.

No model is re-fitted here. The script only evaluates locked fixed-test
probabilities.

Expected full Primary11 prediction columns
------------------------------------------
Gradient Boosting Decision Tree Prob
LightGBM Prob
Random Forest Prob
XGBoost Prob
Multilayer Perceptron Prob
Logistic Regression Prob
Decision Tree Prob
Naive Bayes Prob
True Label

For audit convenience, --allow-partial also recognizes the current parsimonious
comparison file columns Primary11_GBDT and Primary11_LR.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import math
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import logit
from scipy.stats import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

EXPECTED_TEST_N = 1011
EXPECTED_TEST_EVENTS = 407
N_BINS = 10

MODEL_COLUMNS = {
    "GBDT": "Gradient Boosting Decision Tree Prob",
    "LightGBM": "LightGBM Prob",
    "RF": "Random Forest Prob",
    "XGBoost": "XGBoost Prob",
    "MLP": "Multilayer Perceptron Prob",
    "LR": "Logistic Regression Prob",
    "DT": "Decision Tree Prob",
    "NB": "Naive Bayes Prob",
}

MODEL_LABELS = {
    "GBDT": "GBDT",
    "LightGBM": "LightGBM",
    "RF": "RF",
    "XGBoost": "XGBoost",
    "MLP": "MLP",
    "LR": "LR",
    "DT": "DT",
    "NB": "NB",
}

PARTIAL_ALIASES = {
    "GBDT": ["Primary11_GBDT"],
    "LR": ["Primary11_LR"],
}


def read_csv_flexible(path: Path) -> pd.DataFrame:
    last = None
    for enc in ("utf-8-sig", "utf-8", "gb18030", "gbk", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError as exc:
            last = exc
    raise last or RuntimeError(f"Unable to read {path}")


def find_label_column(df: pd.DataFrame) -> str:
    for c in ("True Label", "True_Label", "True_label", "Pulmonary_infection"):
        if c in df.columns:
            return c
    raise ValueError("No binary outcome column found. Expected True Label/True_Label/Pulmonary_infection.")


def resolve_model_columns(df: pd.DataFrame, allow_partial: bool) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for key, expected in MODEL_COLUMNS.items():
        if expected in df.columns:
            resolved[key] = expected
            continue
        if allow_partial:
            for alias in PARTIAL_ALIASES.get(key, []):
                if alias in df.columns:
                    resolved[key] = alias
                    break

    if not resolved:
        raise ValueError("No recognized Primary11 probability columns found.")

    missing = [k for k in MODEL_COLUMNS if k not in resolved]
    if missing and not allow_partial:
        raise ValueError(
            "Full eight-model probability file is required. Missing models: " + ", ".join(missing)
        )
    if missing:
        warnings.warn(
            "Partial audit mode: only models found in the input will be processed; "
            "this is not sufficient to regenerate the full Supplementary Figure S3/Table S8."
        )
    return resolved


def validate(df: pd.DataFrame, label_col: str) -> tuple[np.ndarray, pd.DataFrame]:
    y = pd.to_numeric(df[label_col], errors="coerce")
    if y.isna().any() or not set(y.unique()).issubset({0, 1}):
        raise ValueError(f"{label_col} must be complete binary 0/1.")
    y = y.astype(int).to_numpy()

    if len(y) != EXPECTED_TEST_N:
        raise ValueError(f"Expected fixed test n={EXPECTED_TEST_N}; observed n={len(y)}")
    if int(y.sum()) != EXPECTED_TEST_EVENTS:
        raise ValueError(f"Expected fixed test events={EXPECTED_TEST_EVENTS}; observed={int(y.sum())}")
    return y, df


def calibration_intercept_slope(y_true, y_prob, eps=1e-6) -> tuple[float, float]:
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(y_prob, dtype=float), eps, 1 - eps)
    lp = logit(p)

    # calibration slope: logistic(y ~ intercept + slope*logit(p))
    try:
        lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000)
    except (TypeError, ValueError):
        lr = LogisticRegression(penalty="l2", C=1e12, solver="lbfgs", max_iter=2000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lr.fit(lp.reshape(-1, 1), y)
    slope = float(lr.coef_[0, 0])

    # calibration intercept / calibration-in-the-large: slope fixed at 1
    def neg_ll(b):
        eta = lp + b[0]
        log_p = -np.logaddexp(0, -eta)
        log_1mp = -np.logaddexp(0, eta)
        return -float(np.sum(y * log_p + (1 - y) * log_1mp))

    res = minimize(neg_ll, x0=[0.0], method="L-BFGS-B")
    return float(res.x[0]), slope


def quantile_bin_stats(y_true, y_prob, n_bins=N_BINS) -> pd.DataFrame:
    """
    Quantile binning matching sklearn calibration_curve(strategy='quantile')
    for continuous probabilities: percentile cut points followed by searchsorted.
    This is the definition used for the Primary11 ECE/MCE audit.
    """
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_prob, dtype=float)

    edges = np.percentile(p, np.linspace(0, 100, n_bins + 1))
    edges = np.unique(edges)
    if len(edges) < 2:
        raise ValueError("Predictions have no usable variation for calibration bins.")

    ids = np.searchsorted(edges[1:-1], p, side="right")
    rows = []
    for k in range(len(edges) - 1):
        idx = np.flatnonzero(ids == k)
        if len(idx) == 0:
            continue
        mean_p = float(np.mean(p[idx]))
        mean_y = float(np.mean(y[idx]))
        rows.append({
            "Bin": len(rows) + 1,
            "N": int(len(idx)),
            "Mean_predicted_probability": mean_p,
            "Observed_event_rate": mean_y,
            "Absolute_error": abs(mean_p - mean_y),
            "Lower_probability_edge": float(edges[k]),
            "Upper_probability_edge": float(edges[k + 1]),
        })
    return pd.DataFrame(rows)


def ece_mce(y_true, y_prob, n_bins=N_BINS) -> tuple[float, float, pd.DataFrame]:
    bins = quantile_bin_stats(y_true, y_prob, n_bins=n_bins)
    n = len(y_true)
    ece = float(np.sum((bins["N"] / n) * bins["Absolute_error"]))
    mce = float(bins["Absolute_error"].max())
    return ece, mce, bins


def hosmer_lemeshow(y_true, y_prob, n_bins=N_BINS) -> tuple[float, float, int]:
    """
    H-L statistic using 10 equal-sized groups after ordering by predicted risk.
    This grouping reproduces the reported Primary11 H-L values (e.g., GBDT P~0.254).
    """
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=float)
    groups = np.array_split(np.argsort(p), n_bins)
    stat = 0.0
    used = 0
    for idx in groups:
        if len(idx) == 0:
            continue
        used += 1
        n_g = len(idx)
        o1 = float(np.sum(y[idx]))
        e1 = float(np.sum(p[idx]))
        o0 = n_g - o1
        e0 = n_g - e1
        if e1 > 0:
            stat += (o1 - e1) ** 2 / e1
        if e0 > 0:
            stat += (o0 - e0) ** 2 / e0
    df = max(used - 2, 1)
    return float(stat), float(chi2.sf(stat, df)), int(df)


def bootstrap_calibration(y_true, y_prob, n_boot=1000, seed=42, n_bins=N_BINS):
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=float)
    n = len(y)
    rng = np.random.default_rng(seed)

    ece_store = []
    mce_store = []
    # Fixed bin index 1..10 across bootstrap replicates, based on each replicate's
    # own quantile-risk ordering; this yields pointwise uncertainty by risk-group rank.
    obs_store = [[] for _ in range(n_bins)]
    pred_store = [[] for _ in range(n_bins)]

    successful = 0
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        pb = p[idx]
        if np.unique(yb).size < 2:
            continue
        try:
            ece, mce, bins = ece_mce(yb, pb, n_bins=n_bins)
        except Exception:
            continue
        ece_store.append(ece)
        mce_store.append(mce)
        successful += 1

        # Normally 10 quantile groups are available. If ties collapse groups,
        # record the available leading groups and leave the rest missing.
        for j, row in bins.reset_index(drop=True).iterrows():
            if j >= n_bins:
                break
            obs_store[j].append(float(row["Observed_event_rate"]))
            pred_store[j].append(float(row["Mean_predicted_probability"]))

    if successful == 0:
        raise RuntimeError("No successful bootstrap resamples.")

    ece_ci = tuple(np.percentile(ece_store, [2.5, 97.5]).astype(float))
    mce_ci = tuple(np.percentile(mce_store, [2.5, 97.5]).astype(float))

    pointwise = []
    for j in range(n_bins):
        obs = np.asarray(obs_store[j], dtype=float)
        pred = np.asarray(pred_store[j], dtype=float)
        pointwise.append({
            "Bin": j + 1,
            "Bootstrap_n": int(len(obs)),
            "Observed_rate_bootstrap_low": float(np.percentile(obs, 2.5)) if len(obs) else np.nan,
            "Observed_rate_bootstrap_high": float(np.percentile(obs, 97.5)) if len(obs) else np.nan,
            "Mean_predicted_bootstrap_low": float(np.percentile(pred, 2.5)) if len(pred) else np.nan,
            "Mean_predicted_bootstrap_high": float(np.percentile(pred, 97.5)) if len(pred) else np.nan,
        })

    return {
        "ECE_low": ece_ci[0], "ECE_high": ece_ci[1],
        "MCE_low": mce_ci[0], "MCE_high": mce_ci[1],
        "Bootstrap_successful": successful,
        "pointwise": pd.DataFrame(pointwise),
    }


def fmt_p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Primary11 bootstrap calibration and ECE/MCE analysis.")
    parser.add_argument("--predictions", required=True, type=Path,
                        help="Patient-level calibrated fixed-test prediction CSV from the Primary11 main analysis.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/calibration"))
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-partial", action="store_true",
                        help="Audit mode: process only recognized models present in the input.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = read_csv_flexible(args.predictions)
    df.columns = df.columns.astype(str).str.replace("\ufeff", "", regex=False).str.strip()
    label_col = find_label_column(df)
    y, df = validate(df, label_col)
    model_cols = resolve_model_columns(df, allow_partial=args.allow_partial)

    summary_rows = []
    plot_data: dict[str, pd.DataFrame] = {}

    # Model-specific seeds keep reruns deterministic and independent of processing order.
    for model_index, (model_key, col) in enumerate(model_cols.items()):
        p = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError(f"Invalid probability values in {col}")

        brier = float(brier_score_loss(y, p))
        intercept, slope = calibration_intercept_slope(y, p)
        hl_stat, hl_p, hl_df = hosmer_lemeshow(y, p)
        ece, mce, bins = ece_mce(y, p)
        boot = bootstrap_calibration(
            y, p, n_boot=args.n_bootstrap,
            seed=args.seed + 1009 * model_index,
            n_bins=N_BINS,
        )

        point = bins.merge(boot["pointwise"], on="Bin", how="left")
        point.insert(0, "Model", model_key)
        point.to_csv(
            args.output_dir / f"calibration_bins_{model_key}.csv",
            index=False, encoding="utf-8-sig",
        )
        plot_data[model_key] = point

        summary_rows.append({
            "Model": model_key,
            "Probability_column": col,
            "N": len(y),
            "Events": int(y.sum()),
            "Brier": brier,
            "Calibration_intercept": intercept,
            "Calibration_slope": slope,
            "HL_statistic": hl_stat,
            "HL_df": hl_df,
            "HL_P": hl_p,
            "HL_P_display": fmt_p(hl_p),
            "ECE": ece,
            "ECE_95CI_low": boot["ECE_low"],
            "ECE_95CI_high": boot["ECE_high"],
            "MCE": mce,
            "MCE_95CI_low": boot["MCE_low"],
            "MCE_95CI_high": boot["MCE_high"],
            "Bootstrap_successful": boot["Bootstrap_successful"],
        })

        print(
            f"{model_key:8s} Brier={brier:.4f} Int={intercept:+.4f} Slope={slope:.4f} "
            f"HL P={hl_p:.4g} ECE={ece:.4f} MCE={mce:.4f}"
        )

    summary = pd.DataFrame(summary_rows)
    summary_file = args.output_dir / "Table_S8_Primary11_calibration_ECE_MCE.csv"
    summary.to_csv(summary_file, index=False, encoding="utf-8-sig")

    # Supplementary Figure S3: one panel per available model.
    keys = list(plot_data)
    ncols = 4 if len(keys) > 4 else max(1, len(keys))
    nrows = math.ceil(len(keys) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.2 * nrows), squeeze=False)
    axes_flat = axes.ravel()

    summary_by_model = summary.set_index("Model")
    for ax, model_key in zip(axes_flat, keys):
        d = plot_data[model_key]
        x = d["Mean_predicted_probability"].to_numpy(dtype=float)
        yobs = d["Observed_event_rate"].to_numpy(dtype=float)
        lo = d["Observed_rate_bootstrap_low"].to_numpy(dtype=float)
        hi = d["Observed_rate_bootstrap_high"].to_numpy(dtype=float)
        yerr = np.vstack([np.maximum(yobs - lo, 0), np.maximum(hi - yobs, 0)])

        ax.plot([0, 1], [0, 1], "--", linewidth=1.2, label="Perfect calibration")
        ax.errorbar(x, yobs, yerr=yerr, fmt="o-", capsize=3, linewidth=1.3, markersize=4)
        ece = float(summary_by_model.loc[model_key, "ECE"])
        ax.set_title(f"{MODEL_LABELS.get(model_key, model_key)} | ECE={ece:.3f}")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed event rate")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.25)

    for ax in axes_flat[len(keys):]:
        ax.axis("off")

    fig.tight_layout()
    fig_file = args.output_dir / "Figure_S3_Primary11_bootstrap_calibration.png"
    fig.savefig(fig_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    readme = f"""Primary11 bootstrap calibration audit
=====================================
Input: {args.predictions}
Fixed internal test set: n={len(y)}, events={int(y.sum())}
Models processed: {', '.join(keys)}
Bootstrap resamples requested: {args.n_bootstrap}
Base seed: {args.seed}

Definitions
-----------
- Brier score: mean squared probability error.
- Calibration intercept: slope fixed at 1 on logit probability.
- Calibration slope: logistic outcome ~ intercept + slope*logit(probability).
- H-L: 10 equal-sized ordered risk groups; df = groups - 2.
- ECE/MCE: approximately equal-frequency quantile risk groups.
- ECE = weighted mean absolute observed-predicted gap.
- MCE = largest absolute observed-predicted gap.
- Bootstrap intervals: patient-level percentile intervals; no model refitting.
- Figure bands/bars: pointwise bootstrap intervals for observed event rate by risk-group rank.
"""
    (args.output_dir / "README_calibration_method.txt").write_text(readme, encoding="utf-8")

    # Current Primary11 audit checks when the relevant models are present.
    if "GBDT" in summary_by_model.index:
        r = summary_by_model.loc["GBDT"]
        assert abs(float(r["Brier"]) - 0.1620382296112247) < 5e-4
        assert abs(float(r["Calibration_intercept"]) - (-0.0643135425)) < 5e-4
        assert abs(float(r["Calibration_slope"]) - 1.0490782740) < 5e-4
        assert abs(float(r["ECE"]) - 0.0250888134) < 5e-4
        assert abs(float(r["MCE"]) - 0.0668689894) < 5e-4
    if "LR" in summary_by_model.index:
        r = summary_by_model.loc["LR"]
        assert abs(float(r["Brier"]) - 0.1664371188) < 5e-4
        assert abs(float(r["Calibration_intercept"]) - (-0.0594068826)) < 5e-4
        assert abs(float(r["Calibration_slope"]) - 1.0186440027) < 5e-4
        assert abs(float(r["ECE"]) - 0.0617834552) < 5e-4
        assert abs(float(r["MCE"]) - 0.1115609731) < 5e-4

    print(f"Saved: {summary_file}")
    print(f"Saved: {fig_file}")


if __name__ == "__main__":
    main()
