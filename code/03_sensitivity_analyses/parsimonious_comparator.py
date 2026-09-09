# -*- coding: utf-8 -*-
"""
Parsimonious baseline update - STANDALONE v2

This version fixes the FileNotFoundError in the previous script.
It DOES NOT require:
  - 英文列名准备变量筛选(4).csv
  - BSAfree_feature_selection_input_exactsplit.csv
  - the old simple-comparator source dataset

It needs ONE restricted combined held-out-test prediction file:
  03_parsimonious_test_predictions_with_Primary11.csv
Set AIS_ICU_DATA_DIR to the local restricted-data directory, or set
AIS_ICU_PARSIMONIOUS_PREDICTIONS to the exact file path.

That file contains Study_row_id, the true label, current Primary11 LR/GBDT probabilities,
and the four prespecified simple-comparator probabilities. The script then reproduces:
  01_parsimonious_baseline_performance_Primary11.csv
  02_parsimonious_vs_Primary11LR_paired_comparison.csv
  04_parsimonious_Primary11_reviewer_core_summary.csv

All comparisons are on the same fixed 1,011 held-out patients.
"""

from pathlib import Path
import os
import warnings
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
BASE_DIR = Path(
    os.environ.get("AIS_ICU_DATA_DIR", str(REPO_ROOT / "restricted_data"))
).expanduser().resolve()
OUTPUT_ROOT = Path(
    os.environ.get("AIS_ICU_OUTPUT_DIR", str(REPO_ROOT / "outputs"))
).expanduser().resolve()
PREDICTION_FILE_ENV = os.environ.get("AIS_ICU_PARSIMONIOUS_PREDICTIONS", "").strip()

N_BOOT = 1000
BOOT_SEED = 20260903

INPUT_CANDIDATES = [
    "03_parsimonious_test_predictions_with_Primary11.csv",
]

OUTPUT_DIR = OUTPUT_ROOT / "parsimonious_comparator"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SIMPLE_MODELS = [
    "Airway2_MV_Intubation",
    "NLR1",
    "AgeSex2_partial_score_components",
    "NEU_LYM_MV3",
]

EXPECTED_TEST_N = 1011
EXPECTED_TEST_EVENTS = 407
EXPECTED_LR_AUC = 0.8273101518
EXPECTED_GBDT_AUC = 0.8361578014


# -----------------------------------------------------------------------------
# I/O helpers
# -----------------------------------------------------------------------------
def find_input_file():
    if PREDICTION_FILE_ENV:
        p = Path(PREDICTION_FILE_ENV).expanduser().resolve()
        if p.exists():
            return p
        raise FileNotFoundError(f"AIS_ICU_PARSIMONIOUS_PREDICTIONS does not exist: {p}")

    for rel in INPUT_CANDIDATES:
        direct = BASE_DIR / rel
        if direct.exists():
            return direct

    hits = list(BASE_DIR.rglob("03_parsimonious_test_predictions_with_Primary11.csv"))
    if hits:
        return sorted(hits, key=lambda x: x.stat().st_mtime, reverse=True)[0]

    raise FileNotFoundError(
        "Could not locate 03_parsimonious_test_predictions_with_Primary11.csv under "
        f"AIS_ICU_DATA_DIR={BASE_DIR}. Set AIS_ICU_PARSIMONIOUS_PREDICTIONS explicitly."
    )


def read_csv(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = df.columns.astype(str).str.replace("\ufeff", "", regex=False).str.strip()
    return df


def save_csv(df, filename):
    path = OUTPUT_DIR / filename
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  saved: {path}")
    return path


# -----------------------------------------------------------------------------
# Metrics and bootstrap
# -----------------------------------------------------------------------------
def model_metrics(y, p):
    return {
        "AUC": float(roc_auc_score(y, p)),
        "AP": float(average_precision_score(y, p)),
        "Brier": float(brier_score_loss(y, p)),
    }


def bootstrap_model_ci(y, p, n_boot=N_BOOT, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    vals = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        if np.unique(yy).size < 2:
            continue
        pp = p[idx]
        vals.append([
            roc_auc_score(yy, pp),
            average_precision_score(yy, pp),
            brier_score_loss(yy, pp),
        ])

    arr = np.asarray(vals, dtype=float)
    if len(arr) == 0:
        raise RuntimeError("No valid bootstrap resamples were generated.")

    lo = np.percentile(arr, 2.5, axis=0)
    hi = np.percentile(arr, 97.5, axis=0)
    return {
        "AUC_CI_low": float(lo[0]),
        "AUC_CI_high": float(hi[0]),
        "AP_CI_low": float(lo[1]),
        "AP_CI_high": float(hi[1]),
        "Brier_CI_low": float(lo[2]),
        "Brier_CI_high": float(hi[2]),
    }


def paired_bootstrap_delta(y, p_comp, p_ref, n_boot=N_BOOT, seed=BOOT_SEED):
    """
    Delta = simple comparator - Primary11 LR.
    Negative delta AUC/AP means the simple model is worse.
    Positive delta Brier means the simple model has worse probability error.
    """
    rng = np.random.default_rng(seed)
    vals = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        if np.unique(yy).size < 2:
            continue
        pc = p_comp[idx]
        pr = p_ref[idx]
        vals.append([
            roc_auc_score(yy, pc) - roc_auc_score(yy, pr),
            average_precision_score(yy, pc) - average_precision_score(yy, pr),
            brier_score_loss(yy, pc) - brier_score_loss(yy, pr),
        ])

    arr = np.asarray(vals, dtype=float)
    if len(arr) == 0:
        raise RuntimeError("No valid paired bootstrap resamples were generated.")

    lo = np.percentile(arr, 2.5, axis=0)
    hi = np.percentile(arr, 97.5, axis=0)
    return {
        "Delta_AUC_CI_low": float(lo[0]),
        "Delta_AUC_CI_high": float(hi[0]),
        "Delta_AP_CI_low": float(lo[1]),
        "Delta_AP_CI_high": float(hi[1]),
        "Delta_Brier_CI_low": float(lo[2]),
        "Delta_Brier_CI_high": float(hi[2]),
    }


# -----------------------------------------------------------------------------
# Paired DeLong
# -----------------------------------------------------------------------------
def compute_midrank(x):
    x = np.asarray(x)
    order = np.argsort(x)
    z = x[order]
    n = len(x)
    t = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n, dtype=float)
    out[order] = t
    return out


def fast_delong(predictions_sorted_transposed, label_1_count):
    m = int(label_1_count)
    n = predictions_sorted_transposed.shape[1] - m
    positive = predictions_sorted_transposed[:, :m]
    negative = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty((k, m), dtype=float)
    ty = np.empty((k, n), dtype=float)
    tz = np.empty((k, m + n), dtype=float)

    for r in range(k):
        tx[r] = compute_midrank(positive[r])
        ty[r] = compute_midrank(negative[r])
        tz[r] = compute_midrank(predictions_sorted_transposed[r])

    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    cov = sx / m + sy / n
    return aucs, cov


def paired_delong(y, p_ref, p_comp):
    """Reference-minus-comparator orientation."""
    y = np.asarray(y, dtype=int)
    p_ref = np.asarray(p_ref, dtype=float)
    p_comp = np.asarray(p_comp, dtype=float)
    order = np.argsort(-y)
    m = int(y.sum())
    preds = np.vstack([p_ref, p_comp])[:, order]
    aucs, cov = fast_delong(preds, m)
    contrast = np.array([1.0, -1.0])
    var = float(contrast @ cov @ contrast.T)
    diff = float(aucs[0] - aucs[1])
    if var <= 0:
        return diff, np.nan, np.nan
    z = diff / np.sqrt(var)
    p = 2.0 * norm.sf(abs(z))
    return diff, float(z), float(p)


def holm_adjust(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)
    m = len(pvalues)
    order = np.argsort(pvalues)
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        value = (m - rank) * pvalues[idx]
        running = max(running, value)
        adjusted[idx] = min(1.0, running)
    return adjusted


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    print("=" * 88)
    print("Parsimonious baseline update v2: simple comparators vs current Primary11 LR")
    print("=" * 88)

    input_path = find_input_file()
    print(f"Input: {input_path}")

    df = read_csv(input_path)

    required = {
        "Study_row_id",
        "True_Label",
        "Primary11_LR",
        "Primary11_GBDT",
        *SIMPLE_MODELS,
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"输入文件缺少列: {missing}")

    if len(df) != EXPECTED_TEST_N:
        raise ValueError(f"Test样本数应为 {EXPECTED_TEST_N}，实际为 {len(df)}")

    if df["Study_row_id"].duplicated().any():
        raise ValueError("Study_row_id存在重复，停止分析。")

    y = df["True_Label"].astype(int).to_numpy()
    if int(y.sum()) != EXPECTED_TEST_EVENTS:
        raise ValueError(f"Test事件数应为 {EXPECTED_TEST_EVENTS}，实际为 {int(y.sum())}")

    p_ref = df["Primary11_LR"].astype(float).to_numpy()
    p_gbdt = df["Primary11_GBDT"].astype(float).to_numpy()

    auc_lr = roc_auc_score(y, p_ref)
    auc_gbdt = roc_auc_score(y, p_gbdt)

    if abs(auc_lr - EXPECTED_LR_AUC) > 5e-5:
        raise ValueError(f"Primary11 LR AUC核对失败: {auc_lr:.6f}")
    if abs(auc_gbdt - EXPECTED_GBDT_AUC) > 5e-5:
        raise ValueError(f"Primary11 GBDT AUC核对失败: {auc_gbdt:.6f}")

    print("\n固定Test核对通过:")
    print(f"  n={len(y)}, events={int(y.sum())}")
    print(f"  Primary11 LR AUC={auc_lr:.6f}")
    print(f"  Primary11 GBDT AUC={auc_gbdt:.6f}")

    # Patient-level predictions are intentionally not copied into repository-oriented outputs.

    # Performance table
    all_models = {
        "Primary11_LR_current": p_ref,
        "Primary11_GBDT_current": p_gbdt,
        **{name: df[name].astype(float).to_numpy() for name in SIMPLE_MODELS},
    }
    n_predictors = {
        "Primary11_LR_current": 11,
        "Primary11_GBDT_current": 11,
        "Airway2_MV_Intubation": 2,
        "NLR1": 1,
        "AgeSex2_partial_score_components": 2,
        "NEU_LYM_MV3": 3,
    }

    perf_rows = []
    for i, (name, p) in enumerate(all_models.items()):
        mm = model_metrics(y, p)
        ci = bootstrap_model_ci(y, p, seed=BOOT_SEED + i * 100)
        perf_rows.append({
            "Model": name,
            "N_predictors": n_predictors[name],
            **mm,
            **ci,
        })

    perf = pd.DataFrame(perf_rows)
    save_csv(perf, "01_parsimonious_baseline_performance_Primary11.csv")

    # Paired comparisons vs current Primary11 LR
    ref_metrics = model_metrics(y, p_ref)
    comp_rows = []
    for i, name in enumerate(SIMPLE_MODELS):
        p = df[name].astype(float).to_numpy()
        cm = model_metrics(y, p)
        paired_ci = paired_bootstrap_delta(
            y, p, p_ref, seed=BOOT_SEED + 1000 + i * 100
        )
        delong_ref_minus_comp, z, pval = paired_delong(y, p_ref, p)

        comp_rows.append({
            "Comparison": f"{name} vs Primary11_LR",
            "Model": name,
            "Primary11_LR_AUC": ref_metrics["AUC"],
            "Comparator_AUC": cm["AUC"],
            "Delta_AUC_comparator_minus_Primary11LR": cm["AUC"] - ref_metrics["AUC"],
            "Delta_AUC_CI_low": paired_ci["Delta_AUC_CI_low"],
            "Delta_AUC_CI_high": paired_ci["Delta_AUC_CI_high"],
            "Primary11_LR_AP": ref_metrics["AP"],
            "Comparator_AP": cm["AP"],
            "Delta_AP_comparator_minus_Primary11LR": cm["AP"] - ref_metrics["AP"],
            "Delta_AP_CI_low": paired_ci["Delta_AP_CI_low"],
            "Delta_AP_CI_high": paired_ci["Delta_AP_CI_high"],
            "Primary11_LR_Brier": ref_metrics["Brier"],
            "Comparator_Brier": cm["Brier"],
            "Delta_Brier_comparator_minus_Primary11LR": cm["Brier"] - ref_metrics["Brier"],
            "Delta_Brier_CI_low": paired_ci["Delta_Brier_CI_low"],
            "Delta_Brier_CI_high": paired_ci["Delta_Brier_CI_high"],
            "DeLong_delta_AUC_Primary11LR_minus_comparator": delong_ref_minus_comp,
            "DeLong_Z_Primary11LR_minus_comparator": z,
            "DeLong_P_raw": pval,
        })

    comp = pd.DataFrame(comp_rows)
    comp["DeLong_P_Holm"] = holm_adjust(comp["DeLong_P_raw"].to_numpy())
    comp["Significant_after_Holm_0.05"] = np.where(
        comp["DeLong_P_Holm"] < 0.05, "Yes", "No"
    )
    save_csv(comp, "02_parsimonious_vs_Primary11LR_paired_comparison.csv")

    best_idx = comp["Comparator_AUC"].idxmax()
    best = comp.loc[best_idx]
    core = pd.DataFrame([{
        "Reference_model": "Primary11_LR_current",
        "Reference_AUC": ref_metrics["AUC"],
        "Reference_AP": ref_metrics["AP"],
        "Reference_Brier": ref_metrics["Brier"],
        "Best_simple_comparator": best["Model"],
        "Best_simple_AUC": best["Comparator_AUC"],
        "Delta_AUC_simple_minus_Primary11LR": best["Delta_AUC_comparator_minus_Primary11LR"],
        "Delta_AUC_95CI_low": best["Delta_AUC_CI_low"],
        "Delta_AUC_95CI_high": best["Delta_AUC_CI_high"],
        "DeLong_P_raw": best["DeLong_P_raw"],
        "DeLong_P_Holm": best["DeLong_P_Holm"],
        "Interpretation": (
            "The best prespecified simple comparator remained materially less discriminative "
            "than the current 11-predictor LR reference on the same fixed held-out patients."
        ),
    }])
    save_csv(core, "04_parsimonious_Primary11_reviewer_core_summary.csv")

    print("\nKey paired comparisons (comparator minus Primary11 LR):")
    cols = [
        "Model",
        "Comparator_AUC",
        "Delta_AUC_comparator_minus_Primary11LR",
        "Delta_AUC_CI_low",
        "Delta_AUC_CI_high",
        "DeLong_P_Holm",
    ]
    print(comp[cols].to_string(index=False))

    print("\n完成。输出目录:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
