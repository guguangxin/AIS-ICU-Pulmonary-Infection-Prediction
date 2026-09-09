# -*- coding: utf-8 -*-
"""
Reviewer sensitivity analysis: Primary11 vs AgeForced12
=======================================================

Purpose
-------
The revised primary model now uses 11 predictors selected after the prespecified
exclusion of BSA and immunosuppressant use, followed by the training-set
zero-variance rule and LASSO ∩ Boruta feature selection.

This script directly addresses the reviewer concern that Age was not retained:

    Primary11  = revised 11-predictor primary set
    AgeForced12 = Primary11 + Age

Design
------
1. Preserve the locked original Train/Test identities from
   BSAfree_feature_selection_input_exactsplit.csv.
2. Use the already generated Primary11 calibrated test probabilities as the
   reference, so the published/main-analysis reference is not re-fit or altered.
3. Re-fit only the AgeForced12 models using the SAME 8 algorithms, Bayesian
   search spaces, 10-fold stratified tuning, and 10-fold Platt calibration as
   the revised primary analysis.
4. Compare AgeForced12 vs Primary11 on the SAME 1,011 test patients using:
   - AUC, AP, Brier score, calibration intercept, calibration slope
   - 1,000 patient-level bootstrap 95% CIs for AgeForced12 metrics
   - paired bootstrap 95% CIs for ΔAUC, ΔAP, ΔBrier
   - paired DeLong tests for AUC, with Holm adjustment across 8 algorithms
5. This is a sensitivity analysis. It does not use the test set for tuning,
   calibration, threshold selection, or hyperparameter selection.

Expected fixed split
--------------------
Train: n=2357, events=950
Test : n=1011, events=407

Public repository usage
-----------------------
Patient-level inputs are intentionally not included in the repository. Set
AIS_ICU_DATA_DIR to a local restricted-data directory containing the exact-split
analysis-ready file and the locked Primary11 patient-level prediction file.
Outputs are written under AIS_ICU_OUTPUT_DIR (default: <repo>/outputs).
"""

from pathlib import Path
import os
import gc
import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd

from scipy import stats
from scipy.special import logit
from scipy.optimize import minimize

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

warnings.filterwarnings("ignore")


# ============================================================
# 1. Paths and global settings
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DATA_ROOT = Path(
    os.environ.get("AIS_ICU_DATA_DIR", str(REPO_ROOT / "restricted_data"))
).expanduser().resolve()
OUTPUT_ROOT = Path(
    os.environ.get("AIS_ICU_OUTPUT_DIR", str(REPO_ROOT / "outputs"))
).expanduser().resolve()

ROOT = DATA_ROOT
EXACT_SPLIT_FILE = Path(
    os.environ.get(
        "AIS_ICU_EXACT_SPLIT_FILE",
        str(ROOT / "BSAfree_feature_selection_input_exactsplit.csv"),
    )
).expanduser().resolve()
PRIMARY11_OUTPUT_DIR = Path(
    os.environ.get("AIS_ICU_PRIMARY11_OUTPUT_DIR", str(ROOT))
).expanduser().resolve()
OUTPUT_DIR = OUTPUT_ROOT / "age_forced_sensitivity"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TUNING_CV = 10
CALIBRATION_CV = 10
BOOTSTRAP_N = 1000

GLOBAL_N_JOBS = 2
MODEL_N_JOBS = {"MLP": 1, "RF": 2}

# All 8 algorithms are run by default.
# For a quick test only, you may temporarily use ["LR", "GBDT"].
MODELS_TO_RUN = [
    "RF", "GBDT", "LR", "NB", "DT", "LightGBM", "XGBoost", "MLP"
]

OUTCOME = "Pulmonary_infection"
ID_COL = "Study_row_id"
SPLIT_COL = "Primary_split"

PRIMARY11 = [
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
    "CO2",  # manuscript display name = TCO2
]

AGE_FORCED12 = PRIMARY11 + ["Age"]

MODEL_FULL_NAME = {
    "RF": "Random Forest",
    "GBDT": "Gradient Boosting Decision Tree",
    "LR": "Logistic Regression",
    "NB": "Naive Bayes",
    "DT": "Decision Tree",
    "LightGBM": "LightGBM",
    "XGBoost": "XGBoost",
    "MLP": "Multilayer Perceptron",
}

PRIMARY_PROB_COL = {
    k: f"{v} Prob" for k, v in MODEL_FULL_NAME.items()
}

MODEL_N_ITER = {
    "RF": 60,
    "LightGBM": 50,
    "XGBoost": 50,
}
DEFAULT_N_ITER = 40


# ============================================================
# 2. Metric helpers
# ============================================================
def calibration_intercept_slope(y_true, y_proba, eps=1e-6):
    """
    Calibration intercept: fit intercept only with logit(p) as offset.
    Calibration slope: logistic regression of outcome on logit(p).
    """
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(y_proba, dtype=float), eps, 1 - eps)
    lp = logit(p)

    try:
        lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000)
    except (TypeError, ValueError):
        lr = LogisticRegression(
            penalty="l2", C=1e12, solver="lbfgs", max_iter=2000
        )
    lr.fit(lp.reshape(-1, 1), y)
    slope = float(lr.coef_[0, 0])

    def neg_ll(b):
        eta = lp + b[0]
        log_p = -np.logaddexp(0, -eta)
        log_1mp = -np.logaddexp(0, eta)
        return -(y * log_p + (1 - y) * log_1mp).sum()

    res = minimize(neg_ll, x0=[0.0], method="L-BFGS-B")
    intercept = float(res.x[0])
    return intercept, slope


def metric_dict(y_true, p):
    inter, slope = calibration_intercept_slope(y_true, p)
    return {
        "AUC": float(roc_auc_score(y_true, p)),
        "AP": float(average_precision_score(y_true, p)),
        "Brier": float(brier_score_loss(y_true, p)),
        "Calibration_intercept": inter,
        "Calibration_slope": slope,
    }


def percentile_ci(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan, np.nan
    return (
        float(np.percentile(arr, 2.5)),
        float(np.percentile(arr, 97.5)),
    )


def bootstrap_metric_ci(y_true, p, n_boot=BOOTSTRAP_N, seed=RANDOM_STATE):
    """Patient-level bootstrap 95% CIs for probability-based metrics."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y_true)
    p = np.asarray(p)
    n = len(y)

    bag = {
        "AUC": [],
        "AP": [],
        "Brier": [],
        "Calibration_intercept": [],
        "Calibration_slope": [],
    }

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        if np.unique(yb).size < 2:
            continue
        pb = p[idx]
        try:
            mm = metric_dict(yb, pb)
            for k, v in mm.items():
                bag[k].append(v)
        except Exception:
            continue

    out = {}
    for key, vals in bag.items():
        lo, hi = percentile_ci(vals)
        out[f"{key}_95CI_low"] = lo
        out[f"{key}_95CI_high"] = hi
    return out


def paired_bootstrap_delta(y_true, p_new, p_ref,
                           n_boot=BOOTSTRAP_N, seed=RANDOM_STATE):
    """
    Paired bootstrap on the same test patients.
    Delta is always AgeForced12 - Primary11.
    """
    y = np.asarray(y_true)
    p_new = np.asarray(p_new)
    p_ref = np.asarray(p_ref)
    n = len(y)
    rng = np.random.default_rng(seed)

    d_auc, d_ap, d_brier = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        if np.unique(yb).size < 2:
            continue
        pn = p_new[idx]
        pr = p_ref[idx]
        d_auc.append(roc_auc_score(yb, pn) - roc_auc_score(yb, pr))
        d_ap.append(
            average_precision_score(yb, pn)
            - average_precision_score(yb, pr)
        )
        d_brier.append(
            brier_score_loss(yb, pn)
            - brier_score_loss(yb, pr)
        )

    auc_lo, auc_hi = percentile_ci(d_auc)
    ap_lo, ap_hi = percentile_ci(d_ap)
    br_lo, br_hi = percentile_ci(d_brier)

    return {
        "Delta_AUC_boot_mean": float(np.mean(d_auc)),
        "Delta_AUC_95CI_low": auc_lo,
        "Delta_AUC_95CI_high": auc_hi,
        "Delta_AP_boot_mean": float(np.mean(d_ap)),
        "Delta_AP_95CI_low": ap_lo,
        "Delta_AP_95CI_high": ap_hi,
        "Delta_Brier_boot_mean": float(np.mean(d_brier)),
        "Delta_Brier_95CI_low": br_lo,
        "Delta_Brier_95CI_high": br_hi,
    }


# ============================================================
# 3. Paired DeLong test + Holm adjustment
# ============================================================
def _compute_midrank(x):
    j = np.argsort(x)
    z = x[j]
    n = len(x)
    t = np.zeros(n, dtype=float)

    i = 0
    while i < n:
        k = i
        while k < n and z[k] == z[i]:
            k += 1
        t[i:k] = 0.5 * (i + k - 1)
        i = k

    t2 = np.empty(n, dtype=float)
    t2[j] = t + 1
    return t2


def _fast_delong(predictions_sorted_transposed, label_1_count):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    k = predictions_sorted_transposed.shape[0]

    positive = predictions_sorted_transposed[:, :m]
    negative = predictions_sorted_transposed[:, m:]

    tx = np.empty((k, m), dtype=float)
    ty = np.empty((k, n), dtype=float)
    tz = np.empty((k, m + n), dtype=float)

    for r in range(k):
        tx[r] = _compute_midrank(positive[r])
        ty[r] = _compute_midrank(negative[r])
        tz[r] = _compute_midrank(predictions_sorted_transposed[r])

    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    cov = sx / m + sy / n
    return aucs, cov


def paired_delong_age_minus_primary(y_true, p_age, p_primary):
    """Returns AUC_age, AUC_primary, Z for age-primary, and two-sided P."""
    y = np.asarray(y_true)
    p_age = np.asarray(p_age)
    p_primary = np.asarray(p_primary)

    order = np.argsort(y)[::-1]
    pred = np.vstack([p_age[order], p_primary[order]])
    aucs, cov = _fast_delong(pred, int(y.sum()))

    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    se = np.sqrt(max(float(var), 1e-20))
    z = float((aucs[0] - aucs[1]) / se)
    p = float(2 * stats.norm.sf(abs(z)))
    return float(aucs[0]), float(aucs[1]), z, p


def holm_adjust(p_values):
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.zeros(m, dtype=float)
    running = 0.0

    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running = max(running, val)
        adjusted[idx] = min(running, 1.0)
    return adjusted


# ============================================================
# 4. Platt calibration helper
# ============================================================
def make_calibrator(estimator):
    cv = StratifiedKFold(
        n_splits=CALIBRATION_CV,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    try:
        return CalibratedClassifierCV(
            estimator=clone(estimator),
            method="sigmoid",
            cv=cv,
        )
    except TypeError:
        return CalibratedClassifierCV(
            base_estimator=clone(estimator),
            method="sigmoid",
            cv=cv,
        )


# ============================================================
# 5. Model configuration -- copied from revised Primary11 code
# ============================================================
models_config = {
    "RF": {
        "model": RandomForestClassifier(
            random_state=42, n_jobs=1, oob_score=True
        ),
        "params": {
            "n_estimators": Integer(300, 600),
            "max_depth": Integer(4, 10),
            "min_samples_split": Integer(5, 20),
            "min_samples_leaf": Integer(5, 15),
            "max_features": Categorical(["sqrt"]),
            "criterion": Categorical(["gini", "entropy"]),
            "class_weight": Categorical(["balanced", None]),
            "max_samples": Real(0.6, 0.85),
        },
    },
    "GBDT": {
        "model": GradientBoostingClassifier(random_state=42),
        "params": {
            "n_estimators": Integer(100, 400),
            "max_depth": Integer(3, 7),
            "learning_rate": Real(0.01, 0.15, prior="log-uniform"),
            "min_samples_split": Integer(5, 30),
            "min_samples_leaf": Integer(5, 20),
            "subsample": Real(0.7, 0.9),
            "max_features": Categorical(["sqrt", "log2"]),
        },
    },
    "LR": {
        "model": LogisticRegression(random_state=42, max_iter=3000),
        "params": {
            "C": Real(0.001, 100, prior="log-uniform"),
            "penalty": Categorical(["l2"]),
            "solver": Categorical(["lbfgs", "saga", "newton-cg"]),
        },
    },
    "NB": {
        "model": GaussianNB(),
        "params": {
            "var_smoothing": Real(1e-11, 1e-5, prior="log-uniform")
        },
    },
    "DT": {
        "model": DecisionTreeClassifier(random_state=42),
        "params": {
            "max_depth": Integer(3, 15),
            "min_samples_split": Integer(10, 50),
            "min_samples_leaf": Integer(5, 20),
            "criterion": Categorical(["gini", "entropy"]),
            "max_features": Categorical(["sqrt", "log2"]),
            "ccp_alpha": Real(0.001, 0.05),
        },
    },
    "LightGBM": {
        "model": LGBMClassifier(
            random_state=42,
            verbosity=-1,
            n_jobs=1,
            force_col_wise=True,
        ),
        "params": {
            "n_estimators": Integer(100, 400),
            "max_depth": Integer(3, 7),
            "learning_rate": Real(0.01, 0.1, prior="log-uniform"),
            "num_leaves": Integer(10, 40),
            "min_child_samples": Integer(20, 60),
            "reg_alpha": Real(0.05, 2.0, prior="log-uniform"),
            "reg_lambda": Real(0.05, 2.0, prior="log-uniform"),
            "feature_fraction": Real(0.5, 0.85),
            "bagging_fraction": Real(0.5, 0.85),
            "bagging_freq": Integer(1, 7),
        },
    },
    "XGBoost": {
        "model": XGBClassifier(
            random_state=42,
            verbosity=0,
            eval_metric="logloss",
            n_jobs=1,
        ),
        "params": {
            "n_estimators": Integer(100, 400),
            "max_depth": Integer(3, 6),
            "learning_rate": Real(0.01, 0.1, prior="log-uniform"),
            "subsample": Real(0.6, 0.85),
            "colsample_bytree": Real(0.6, 0.85),
            "reg_alpha": Real(0.01, 2.0, prior="log-uniform"),
            "reg_lambda": Real(0.1, 5.0, prior="log-uniform"),
            "min_child_weight": Integer(3, 10),
            "gamma": Real(0.1, 1.0),
        },
    },
    "MLP": {
        "model": MLPClassifier(
            random_state=42,
            max_iter=500,
            hidden_layer_sizes=(64, 32),
            early_stopping=True,
            validation_fraction=0.1,
        ),
        "params": {
            "activation": Categorical(["relu", "tanh"]),
            "alpha": Real(1e-4, 1e-1, prior="log-uniform"),
            "learning_rate_init": Real(
                1e-4, 5e-3, prior="log-uniform"
            ),
            "batch_size": Integer(32, 128),
        },
    },
}


# ============================================================
# 6. Locate restricted inputs
# ============================================================
def locate_exact_split_file():
    if EXACT_SPLIT_FILE.exists():
        return EXACT_SPLIT_FILE

    candidates = list(ROOT.rglob("BSAfree_feature_selection_input_exactsplit.csv"))
    valid = []
    for candidate in candidates:
        try:
            d = pd.read_csv(candidate, encoding="utf-8-sig", nrows=5)
            d.columns = d.columns.astype(str).str.replace("\ufeff", "", regex=False).str.strip()
            required = {ID_COL, SPLIT_COL, OUTCOME, *AGE_FORCED12}
            if required.issubset(d.columns):
                valid.append(candidate)
        except Exception:
            continue

    if not valid:
        raise FileNotFoundError(
            "Could not locate BSAfree_feature_selection_input_exactsplit.csv under "
            f"AIS_ICU_DATA_DIR={ROOT}"
        )
    return sorted(valid, key=lambda x: x.stat().st_mtime, reverse=True)[0]


# ============================================================
# 7. Locate and validate the locked Primary11 probability file
# ============================================================
def locate_primary11_probability_file():
    filename = "测试集逐患者预测概率_校准后_带StudyRowID.csv"

    direct = PRIMARY11_OUTPUT_DIR / filename
    if direct.exists():
        return direct

    # Fallback: recursively search under the revision root.
    candidates = list(ROOT.rglob(filename))
    valid = []
    for p in candidates:
        try:
            d = pd.read_csv(p, encoding="utf-8-sig")
            d.columns = d.columns.astype(str).str.replace(
                "\ufeff", "", regex=False
            ).str.strip()
            required = {ID_COL, "True Label", *PRIMARY_PROB_COL.values()}
            if len(d) == 1011 and required.issubset(d.columns):
                valid.append(p)
        except Exception:
            continue

    if not valid:
        raise FileNotFoundError(
            "找不到新版Primary11逐患者校准后预测概率文件：\n"
            f"{filename}\n\n"
            "请确认主分析输出文件夹位于：\n"
            f"{PRIMARY11_OUTPUT_DIR}"
        )

    # Prefer newest valid file if more than one is present.
    valid = sorted(valid, key=lambda x: x.stat().st_mtime, reverse=True)
    return valid[0]


# ============================================================
# 8. Read exact-split data and Primary11 predictions
# ============================================================
EXACT_SPLIT_FILE = locate_exact_split_file()

print("=" * 88)
print("Age-forced sensitivity: revised Primary11 vs AgeForced12")
print("=" * 88)
print("Exact-split input:", EXACT_SPLIT_FILE)
print("Output:", OUTPUT_DIR)

if not EXACT_SPLIT_FILE.exists():
    raise FileNotFoundError(
        f"找不到：{EXACT_SPLIT_FILE}\n"
        "Set AIS_ICU_DATA_DIR or AIS_ICU_EXACT_SPLIT_FILE to the restricted analysis-ready input."
    )

df = pd.read_csv(EXACT_SPLIT_FILE, encoding="utf-8-sig")
df.columns = df.columns.astype(str).str.replace("\ufeff", "", regex=False).str.strip()

required = {ID_COL, SPLIT_COL, OUTCOME, *AGE_FORCED12}
missing = sorted(required - set(df.columns))
if missing:
    raise ValueError(f"exact-split输入缺少字段：{missing}")
if len(df) != 3368:
    raise ValueError(f"总样本应为3368，实际={len(df)}")

train = df[df[SPLIT_COL].astype(str).str.strip() == "Train"].copy()
test = df[df[SPLIT_COL].astype(str).str.strip() == "Test"].copy()

train = train.reset_index(drop=True)
test = test.reset_index(drop=True)

y_train = train[OUTCOME].astype(int).to_numpy()
y_test = test[OUTCOME].astype(int).to_numpy()
train_ids = train[ID_COL].tolist()
test_ids = test[ID_COL].tolist()

if len(train) != 2357 or int(y_train.sum()) != 950:
    raise ValueError(
        f"Train身份不匹配：n={len(train)}, events={int(y_train.sum())}"
    )
if len(test) != 1011 or int(y_test.sum()) != 407:
    raise ValueError(
        f"Test身份不匹配：n={len(test)}, events={int(y_test.sum())}"
    )

if train[AGE_FORCED12].isna().any().any() or test[AGE_FORCED12].isna().any().any():
    miss_train = train[AGE_FORCED12].isna().sum()
    miss_test = test[AGE_FORCED12].isna().sum()
    raise ValueError(
        "Primary11+Age仍有缺失值，请停止，不要在本敏感性分析中新增临时填补规则。\n"
        f"Train missing: {miss_train[miss_train > 0].to_dict()}\n"
        f"Test missing: {miss_test[miss_test > 0].to_dict()}"
    )

primary_prob_file = locate_primary11_probability_file()
print("Primary11 reference probability file:", primary_prob_file)

primary_pred = pd.read_csv(primary_prob_file, encoding="utf-8-sig")
primary_pred.columns = primary_pred.columns.astype(str).str.replace(
    "\ufeff", "", regex=False
).str.strip()

required_pred = {ID_COL, "True Label", *PRIMARY_PROB_COL.values()}
missing_pred = sorted(required_pred - set(primary_pred.columns))
if missing_pred:
    raise ValueError(f"Primary11概率文件缺少字段：{missing_pred}")
if len(primary_pred) != 1011:
    raise ValueError(f"Primary11概率文件应为1011行，实际={len(primary_pred)}")

# Force exact row alignment by Study_row_id rather than trusting file row order.
primary_pred = primary_pred.set_index(ID_COL)
if set(primary_pred.index.tolist()) != set(test_ids):
    missing_ids = sorted(set(test_ids) - set(primary_pred.index.tolist()))[:20]
    extra_ids = sorted(set(primary_pred.index.tolist()) - set(test_ids))[:20]
    raise ValueError(
        "Primary11概率文件的Study_row_id与固定Test不一致。\n"
        f"Missing IDs (first 20): {missing_ids}\n"
        f"Extra IDs (first 20): {extra_ids}"
    )
primary_pred = primary_pred.loc[test_ids].reset_index()

if not np.array_equal(
    primary_pred["True Label"].astype(int).to_numpy(), y_test
):
    raise ValueError("Primary11概率文件中的True Label与exact-split Test结局不一致")

print(
    f"Locked split verified: Train={len(train)} (events={int(y_train.sum())}) | "
    f"Test={len(test)} (events={int(y_test.sum())})"
)
print("Primary11 predictors:", PRIMARY11)
print("AgeForced12 predictors:", AGE_FORCED12)

# Save design audit.
pd.DataFrame([
    {
        "Total_N": len(df),
        "Train_N": len(train),
        "Train_events": int(y_train.sum()),
        "Test_N": len(test),
        "Test_events": int(y_test.sum()),
        "Primary11_N_predictors": len(PRIMARY11),
        "AgeForced12_N_predictors": len(AGE_FORCED12),
        "Primary_reference_probability_file": str(primary_prob_file),
    }
]).to_csv(
    OUTPUT_DIR / "00_AgeForced12_设计与样本核对.csv",
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 8. Prepare AgeForced12 data exactly as in primary workflow
# ============================================================
X_train_raw = train[AGE_FORCED12].copy()
X_test_raw = test[AGE_FORCED12].copy()

binary_features = []
continuous_features = []
for col in AGE_FORCED12:
    if train[col].nunique(dropna=True) <= 5 or train[col].dtype == object:
        binary_features.append(col)
    else:
        continuous_features.append(col)

print("Categorical/binary:", binary_features)
print("Continuous:", continuous_features)

scaler = MinMaxScaler()
X_train_s = X_train_raw.copy()
X_test_s = X_test_raw.copy()
if continuous_features:
    X_train_s.loc[:, continuous_features] = scaler.fit_transform(
        X_train_raw[continuous_features]
    )
    X_test_s.loc[:, continuous_features] = scaler.transform(
        X_test_raw[continuous_features]
    )

X_train_arr = X_train_s.to_numpy(dtype=float)
X_test_arr = X_test_s.to_numpy(dtype=float)

joblib.dump(
    scaler,
    OUTPUT_DIR / "AgeForced12_标准化器.pkl",
    compress=3,
)


# ============================================================
# 9. Fit AgeForced12 models (checkpoint-resumable)
# ============================================================
age_models = {}
age_raw_models = {}
age_test_probas = {}
age_train_probas = {}
age_cv_auc = {}
age_best_params = {}

performance_rows = []
hyper_rows = []

def checkpoint_path(model_key):
    return CHECKPOINT_DIR / f"ckpt_AgeForced12_{model_key}.pkl"

print("\n" + "=" * 88)
print("Training AgeForced12 models")
print("Same Bayesian spaces + 10-fold tuning + 10-fold Platt calibration as Primary11")
print("=" * 88)

total_t0 = time.time()

for model_key in MODELS_TO_RUN:
    cfg = models_config[model_key]
    cpath = checkpoint_path(model_key)

    print("\n" + "-" * 72)
    print(f"Model: {MODEL_FULL_NAME[model_key]}")

    if cpath.exists():
        try:
            saved = joblib.load(cpath)
            if saved.get("features") != AGE_FORCED12:
                raise ValueError("checkpoint feature set mismatch")
            if saved.get("test_row_ids") != test_ids:
                raise ValueError("checkpoint test identities mismatch")

            age_models[model_key] = saved["calibrated_model"]
            age_raw_models[model_key] = saved["raw_model"]
            age_test_probas[model_key] = np.asarray(saved["p_test"])
            age_train_probas[model_key] = np.asarray(saved["p_train"])
            age_cv_auc[model_key] = float(saved["cv_auc"])
            age_best_params[model_key] = saved["best_params"]
            print(
                f"Loaded checkpoint | CV AUC={age_cv_auc[model_key]:.4f} | "
                f"Test AUC={roc_auc_score(y_test, age_test_probas[model_key]):.4f}"
            )
            continue
        except Exception as e:
            print(f"Checkpoint invalid ({e}); retraining this model.")

    n_iter = MODEL_N_ITER.get(model_key, DEFAULT_N_ITER)
    n_jobs = MODEL_N_JOBS.get(model_key, GLOBAL_N_JOBS)
    t0 = time.time()

    cv = StratifiedKFold(
        n_splits=TUNING_CV,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bayes = BayesSearchCV(
            cfg["model"],
            cfg["params"],
            n_iter=n_iter,
            cv=cv,
            scoring="roc_auc",
            random_state=RANDOM_STATE,
            n_jobs=n_jobs,
            verbose=0,
            refit=True,
        )
        bayes.fit(X_train_arr, y_train)

    raw_model = bayes.best_estimator_

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calibrated_model = make_calibrator(raw_model)
        calibrated_model.fit(X_train_arr, y_train)

    p_test = calibrated_model.predict_proba(X_test_arr)[:, 1]
    p_train = calibrated_model.predict_proba(X_train_arr)[:, 1]

    best_params = {
        k: (v.item() if hasattr(v, "item") else v)
        for k, v in bayes.best_params_.items()
    }

    age_models[model_key] = calibrated_model
    age_raw_models[model_key] = raw_model
    age_test_probas[model_key] = p_test
    age_train_probas[model_key] = p_train
    age_cv_auc[model_key] = float(bayes.best_score_)
    age_best_params[model_key] = best_params

    joblib.dump(
        {
            "features": AGE_FORCED12,
            "test_row_ids": test_ids,
            "calibrated_model": calibrated_model,
            "raw_model": raw_model,
            "p_test": p_test,
            "p_train": p_train,
            "cv_auc": float(bayes.best_score_),
            "best_params": best_params,
        },
        cpath,
        compress=3,
    )

    print(
        f"CV AUC={bayes.best_score_:.4f} | "
        f"Test AUC={roc_auc_score(y_test, p_test):.4f} | "
        f"{(time.time() - t0)/60:.1f} min"
    )
    print("Checkpoint saved:", cpath.name)

    del bayes
    gc.collect()

print(f"\nAll AgeForced12 models ready. Total={(time.time()-total_t0)/60:.1f} min")


# ============================================================
# 10. Probability-based performance table
# ============================================================
print("\nCalculating performance and bootstrap CIs...")

for model_key in MODELS_TO_RUN:
    p_primary = primary_pred[PRIMARY_PROB_COL[model_key]].to_numpy(dtype=float)
    p_age = age_test_probas[model_key]

    m_primary = metric_dict(y_test, p_primary)
    m_age = metric_dict(y_test, p_age)
    age_ci = bootstrap_metric_ci(
        y_test,
        p_age,
        n_boot=BOOTSTRAP_N,
        seed=RANDOM_STATE,
    )

    performance_rows.append({
        "Predictor_set": "Primary11",
        "N_predictors": 11,
        "Model": model_key,
        "Model_full_name": MODEL_FULL_NAME[model_key],
        "Train_CV_AUC": np.nan,
        **m_primary,
    })

    performance_rows.append({
        "Predictor_set": "AgeForced12",
        "N_predictors": 12,
        "Model": model_key,
        "Model_full_name": MODEL_FULL_NAME[model_key],
        "Train_CV_AUC": age_cv_auc[model_key],
        **m_age,
        **age_ci,
    })

    hyper_rows.append({
        "Predictor_set": "AgeForced12",
        "Model": model_key,
        "Model_full_name": MODEL_FULL_NAME[model_key],
        "Train_CV_AUC": age_cv_auc[model_key],
        "Best_parameters": json.dumps(
            age_best_params[model_key],
            ensure_ascii=False,
        ),
    })

perf_df = pd.DataFrame(performance_rows)
perf_df.to_csv(
    OUTPUT_DIR / "01_Primary11_vs_AgeForced12_八模型性能.csv",
    index=False,
    encoding="utf-8-sig",
)

pd.DataFrame(hyper_rows).to_csv(
    OUTPUT_DIR / "03_AgeForced12_最优超参数.csv",
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 11. Paired comparison: point delta + bootstrap CI + DeLong/Holm
# ============================================================
comparison_rows = []

for model_key in MODELS_TO_RUN:
    p_primary = primary_pred[PRIMARY_PROB_COL[model_key]].to_numpy(dtype=float)
    p_age = age_test_probas[model_key]

    m_primary = metric_dict(y_test, p_primary)
    m_age = metric_dict(y_test, p_age)

    boot = paired_bootstrap_delta(
        y_test,
        p_age,
        p_primary,
        n_boot=BOOTSTRAP_N,
        seed=RANDOM_STATE,
    )

    auc_age_d, auc_primary_d, z, p_delong = paired_delong_age_minus_primary(
        y_test, p_age, p_primary
    )

    comparison_rows.append({
        "Model": model_key,
        "Model_full_name": MODEL_FULL_NAME[model_key],
        "Primary11_AUC": m_primary["AUC"],
        "AgeForced12_AUC": m_age["AUC"],
        "Delta_AUC_Age12minusPrimary11": m_age["AUC"] - m_primary["AUC"],
        "Delta_AUC_95CI_low": boot["Delta_AUC_95CI_low"],
        "Delta_AUC_95CI_high": boot["Delta_AUC_95CI_high"],
        "DeLong_Z_Age12minusPrimary11": z,
        "DeLong_P": p_delong,
        "Primary11_AP": m_primary["AP"],
        "AgeForced12_AP": m_age["AP"],
        "Delta_AP_Age12minusPrimary11": m_age["AP"] - m_primary["AP"],
        "Delta_AP_95CI_low": boot["Delta_AP_95CI_low"],
        "Delta_AP_95CI_high": boot["Delta_AP_95CI_high"],
        "Primary11_Brier": m_primary["Brier"],
        "AgeForced12_Brier": m_age["Brier"],
        "Delta_Brier_Age12minusPrimary11": m_age["Brier"] - m_primary["Brier"],
        "Delta_Brier_95CI_low": boot["Delta_Brier_95CI_low"],
        "Delta_Brier_95CI_high": boot["Delta_Brier_95CI_high"],
        "Primary11_Calibration_intercept": m_primary["Calibration_intercept"],
        "AgeForced12_Calibration_intercept": m_age["Calibration_intercept"],
        "Primary11_Calibration_slope": m_primary["Calibration_slope"],
        "AgeForced12_Calibration_slope": m_age["Calibration_slope"],
        "DeLong_AUC_Age12_check": auc_age_d,
        "DeLong_AUC_Primary11_check": auc_primary_d,
    })

comparison_df = pd.DataFrame(comparison_rows)
comparison_df["DeLong_P_Holm"] = holm_adjust(comparison_df["DeLong_P"].values)
comparison_df["DeLong_significant_after_Holm"] = (
    comparison_df["DeLong_P_Holm"] < 0.05
)

comparison_df.to_csv(
    OUTPUT_DIR / "02_AgeForced12_vs_Primary11_配对比较.csv",
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 12. Patient-level probability output
# ============================================================
pred_out = pd.DataFrame({
    ID_COL: test_ids,
    "True_Label": y_test,
})

for model_key in MODELS_TO_RUN:
    pred_out[f"Primary11__{model_key}"] = primary_pred[
        PRIMARY_PROB_COL[model_key]
    ].to_numpy(dtype=float)
    pred_out[f"AgeForced12__{model_key}"] = age_test_probas[model_key]

pred_out.to_csv(
    OUTPUT_DIR / "04_测试集_Primary11与AgeForced12_逐患者校准后概率.csv",
    index=False,
    encoding="utf-8-sig",
)

joblib.dump(
    age_models,
    OUTPUT_DIR / "AgeForced12_calibrated_models.pkl",
    compress=3,
)
joblib.dump(
    age_raw_models,
    OUTPUT_DIR / "AgeForced12_raw_models_uncalibrated.pkl",
    compress=3,
)


# ============================================================
# 13. Compact reviewer-oriented summary
# ============================================================
summary_cols = [
    "Model_full_name",
    "Primary11_AUC",
    "AgeForced12_AUC",
    "Delta_AUC_Age12minusPrimary11",
    "Delta_AUC_95CI_low",
    "Delta_AUC_95CI_high",
    "DeLong_P",
    "DeLong_P_Holm",
    "Primary11_AP",
    "AgeForced12_AP",
    "Delta_AP_Age12minusPrimary11",
    "Delta_AP_95CI_low",
    "Delta_AP_95CI_high",
    "Primary11_Brier",
    "AgeForced12_Brier",
    "Delta_Brier_Age12minusPrimary11",
    "Delta_Brier_95CI_low",
    "Delta_Brier_95CI_high",
]
comparison_df[summary_cols].to_csv(
    OUTPUT_DIR / "05_AgeForced12_审稿回复核心比较.csv",
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 14. README
# ============================================================
readme = f"""
AgeForced12 sensitivity analysis
================================

Reviewer issue addressed
-------------------------
Age was available but was not retained by the revised feature-selection rule.
This analysis evaluates the effect of forcing Age into the revised 11-predictor
primary model.

Locked sample identities
------------------------
Train: n={len(train)}, events={int(y_train.sum())}
Test : n={len(test)}, events={int(y_test.sum())}

Primary11
---------
{PRIMARY11}

AgeForced12
-----------
{AGE_FORCED12}

Methods
-------
- Primary11 reference probabilities are read directly from the completed revised
  main analysis; the Primary11 model is NOT re-fit in this script.
- AgeForced12 uses the same eight algorithms and exactly the same Bayesian search
  spaces as the revised Primary11 code.
- Hyperparameter tuning: 10-fold stratified BayesSearchCV, ROC-AUC scoring.
- Calibration: 10-fold sigmoid / Platt calibration confined to Train.
- Evaluation: the same locked 1,011-patient internal Test partition.
- Probability metrics: AUC, AP, Brier, calibration intercept, calibration slope.
- AgeForced12 metric uncertainty: 1,000 patient-level bootstrap resamples.
- Paired ΔAUC, ΔAP, and ΔBrier: AgeForced12 minus Primary11, with 1,000 paired
  patient-level bootstrap resamples.
- Paired DeLong tests compare AUCs on the same Test patients; Holm correction is
  applied across the eight algorithms.

Important interpretation
------------------------
This is a reviewer-requested sensitivity analysis. Age was forced into the model
rather than selected by the primary feature-selection rule. Any small performance
change should be interpreted as sensitivity to forced inclusion, not as evidence
that age is causally unimportant or clinically irrelevant.

Primary11 probability source
----------------------------
{primary_prob_file}

Files to send back for manuscript revision
------------------------------------------
1) 01_Primary11_vs_AgeForced12_八模型性能.csv
2) 02_AgeForced12_vs_Primary11_配对比较.csv
3) 05_AgeForced12_审稿回复核心比较.csv
"""

(OUTPUT_DIR / "README_AgeForced12敏感性分析.txt").write_text(
    readme.strip() + "\n",
    encoding="utf-8",
)


# ============================================================
# 15. Console summary
# ============================================================
print("\n" + "=" * 88)
print("AgeForced12 sensitivity analysis COMPLETE")
print("=" * 88)

show_cols = [
    "Model",
    "Primary11_AUC",
    "AgeForced12_AUC",
    "Delta_AUC_Age12minusPrimary11",
    "Delta_AUC_95CI_low",
    "Delta_AUC_95CI_high",
    "DeLong_P_Holm",
    "Primary11_AP",
    "AgeForced12_AP",
    "Primary11_Brier",
    "AgeForced12_Brier",
]
print(comparison_df[show_cols].to_string(index=False))

print("\nAll outputs saved to:")
print(OUTPUT_DIR)
print("\nPlease send me these three files:")
print("1) 01_Primary11_vs_AgeForced12_八模型性能.csv")
print("2) 02_AgeForced12_vs_Primary11_配对比较.csv")
print("3) 05_AgeForced12_审稿回复核心比较.csv")
