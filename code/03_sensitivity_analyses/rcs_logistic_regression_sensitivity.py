# -*- coding: utf-8 -*-
"""
Reviewer-specific restricted cubic spline (RCS) sensitivity analysis
for the current 11-predictor logistic-regression model.

Purpose
-------
Compare the locked primary linear LR predictions with a new LR model that allows
non-linear effects for the six continuous laboratory predictors using 4-knot
restricted cubic splines. CCI is retained as a linear score and binary predictors
remain binary.

Design safeguards
-----------------
- Uses the original fixed Primary_split; NEVER re-splits the data.
- Knots are estimated from Train only, then frozen and applied to Test.
- Hyperparameters are tuned on Train only by 10-fold CV.
- Platt calibration is fitted on Train only by 10-fold CV.
- Primary linear-LR and GBDT predictions are read from the locked primary-analysis
  prediction file and aligned by Study_row_id.
- Test is used only for final evaluation/comparison.

Expected input files
--------------------
1) E:\\新建文件夹\\第三次修稿\\BSAfree_feature_selection_input_exactsplit.csv
2) E:\\新建文件夹\\第三次修稿\\主分析11变量_八模型完整重跑_SHAP\\测试集逐患者预测概率_校准后_带StudyRowID.csv

Main outputs
------------
- RCS_LR_vs_PrimaryLR_summary.csv
- RCS_LR_test_predictions_with_ID.csv
- RCS_knots_train_only.csv
- RCS_design_columns.csv
- RCS_LR_calibration_plot.png
- RCS_analysis_design.json
"""

import os
import json
import warnings
import numpy as np
import pandas as pd

from scipy import stats
from scipy.optimize import minimize
from scipy.stats import chi2

from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)

import matplotlib.pyplot as plt

# Optional: use the same Bayesian tuning family as the main analysis when available.
try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Categorical
    HAVE_SKOPT = True
except Exception:
    HAVE_SKOPT = False


# ============================================================
# 0. PATHS / CONSTANTS
# ============================================================
BASE_DIR = r"E:\新建文件夹\第三次修稿"
DATA_FILE = os.path.join(BASE_DIR, "BSAfree_feature_selection_input_exactsplit.csv")
PRIMARY_PRED_FILE = os.path.join(
    BASE_DIR,
    "主分析11变量_八模型完整重跑_SHAP",
    "测试集逐患者预测概率_校准后_带StudyRowID.csv",
)
OUTPUT_DIR = os.path.join(BASE_DIR, "RCS_LR_nonlinearity_sensitivity")
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTCOME = "Pulmonary_infection"
ID_COL = "Study_row_id"
SPLIT_COL = "Primary_split"

# Current primary 11 predictors
PRIMARY_FEATURES = [
    "NEUT_abs",                    # NEU
    "Intubation_tracheotomy",      # intubation/tracheotomy
    "Mechanical_ventilation",      # MV
    "LDH",
    "LYMPH_abs",                   # LYM
    "BUN",
    "CCI",
    "FIB",
    "Surgery",
    "Diuretics",
    "CO2",                         # TCO2
]

# RCS is applied to the six continuous laboratory predictors.
# CCI is intentionally retained as a linear score because it is a low-range,
# discrete Charlson score; binary predictors are unchanged.
RCS_VARIABLES = [
    "NEUT_abs",
    "LDH",
    "LYMPH_abs",
    "BUN",
    "FIB",
    "CO2",
]

LINEAR_CONTINUOUS = ["CCI"]
BINARY_VARIABLES = [
    "Intubation_tracheotomy",
    "Mechanical_ventilation",
    "Surgery",
    "Diuretics",
]

# Harrell-style 4-knot locations: 5th, 35th, 65th, 95th percentiles.
KNOT_QUANTILES = [0.05, 0.35, 0.65, 0.95]
N_BAYES_ITER = 40
CV_FOLDS = 10
CALIBRATION_CV = 10
N_BOOT = 1000
RANDOM_SEED = 42

# Exact names produced by the main script.
PRIMARY_LR_PROB_COL = "Logistic Regression Prob"
PRIMARY_GBDT_PROB_COL = "Gradient Boosting Decision Tree Prob"
PRIMARY_TRUE_COL = "True Label"


# ============================================================
# 1. HELPERS
# ============================================================
def save_csv(df, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"Saved: {path}")
    return path


def calibration_intercept_slope(y_true, y_proba, eps=1e-6):
    """Calibration intercept (slope fixed to 1) and calibration slope."""
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(y_proba, dtype=float), eps, 1 - eps)
    lp = np.log(p / (1 - p))

    try:
        lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=3000)
    except (TypeError, ValueError):
        lr = LogisticRegression(penalty="l2", C=1e12, solver="lbfgs", max_iter=3000)
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


def ece_mce_equal_frequency(y_true, y_prob, n_bins=10):
    """ECE/MCE using approximately equal-frequency risk groups."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=float)
    order = np.argsort(p)
    bins = np.array_split(order, n_bins)

    abs_errors = []
    weights = []
    rows = []
    n = len(y)

    for i, idx in enumerate(bins, start=1):
        if len(idx) == 0:
            continue
        obs = float(y[idx].mean())
        pred = float(p[idx].mean())
        err = abs(obs - pred)
        abs_errors.append(err)
        weights.append(len(idx) / n)
        rows.append({
            "Bin": i,
            "N": int(len(idx)),
            "Mean_predicted": pred,
            "Observed_rate": obs,
            "Abs_error": err,
        })

    ece = float(np.sum(np.asarray(weights) * np.asarray(abs_errors)))
    mce = float(np.max(abs_errors))
    return ece, mce, pd.DataFrame(rows)


def hosmer_lemeshow_test(y_true, y_proba, n_bins=10):
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_proba, dtype=float)
    order = np.argsort(p)
    ys = y[order]
    ps = p[order]
    bins = np.array_split(np.arange(len(y)), n_bins)

    hl = 0.0
    for b in bins:
        if len(b) == 0:
            continue
        op = ys[b].sum()
        ep = ps[b].sum()
        on = len(b) - op
        en = len(b) - ep
        if ep > 0:
            hl += (op - ep) ** 2 / ep
        if en > 0:
            hl += (on - en) ** 2 / en

    pval = 1 - chi2.cdf(hl, n_bins - 2)
    return float(hl), float(pval)


# ---------- DeLong test: copied to match the main-analysis implementation ----------
def _compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=np.float64)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N, dtype=np.float64)
    T2[J] = T + 1
    return T2


def _fast_delong(pst, m):
    n = pst.shape[1] - m
    k = pst.shape[0]
    pos = pst[:, :m]
    neg = pst[:, m:]
    tx = np.empty([k, m], dtype=np.float64)
    ty = np.empty([k, n], dtype=np.float64)
    tz = np.empty([k, m + n], dtype=np.float64)
    for r in range(k):
        tx[r] = _compute_midrank(pos[r])
        ty[r] = _compute_midrank(neg[r])
        tz[r] = _compute_midrank(pst[r])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    return aucs, cov


def delong_test(y_true, p1, p2):
    y = np.asarray(y_true, dtype=int)
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    order = np.argsort(y)[::-1]
    pst = np.vstack([p1[order], p2[order]])
    aucs, cov = _fast_delong(pst, int(y.sum()))
    se = np.sqrt(max(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1], 1e-20))
    z = (aucs[0] - aucs[1]) / se
    p = 2 * stats.norm.sf(abs(z))
    return float(aucs[0]), float(aucs[1]), float(z), float(p)


def metric_dict(y, p):
    ci, cs = calibration_intercept_slope(y, p)
    ece, mce, bins = ece_mce_equal_frequency(y, p, n_bins=10)
    hl, hl_p = hosmer_lemeshow_test(y, p, n_bins=10)
    return {
        "AUC": float(roc_auc_score(y, p)),
        "AP": float(average_precision_score(y, p)),
        "Brier": float(brier_score_loss(y, p)),
        "Calibration_intercept": ci,
        "Calibration_slope": cs,
        "ECE": ece,
        "MCE": mce,
        "HL_statistic": hl,
        "HL_P": hl_p,
    }, bins


def paired_bootstrap_differences(y, p_new, p_ref, n_boot=1000, seed=42):
    """
    Returns paired bootstrap distributions for:
      dAUC   = new - ref
      dAP    = new - ref
      dBrier = new - ref (negative means better Brier for new model)
    """
    y = np.asarray(y, dtype=int)
    p_new = np.asarray(p_new, dtype=float)
    p_ref = np.asarray(p_ref, dtype=float)
    rng = np.random.RandomState(seed)
    n = len(y)

    d_auc = []
    d_ap = []
    d_brier = []

    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        yb = y[idx]
        if len(np.unique(yb)) < 2:
            continue
        pn = p_new[idx]
        pr = p_ref[idx]
        d_auc.append(roc_auc_score(yb, pn) - roc_auc_score(yb, pr))
        d_ap.append(average_precision_score(yb, pn) - average_precision_score(yb, pr))
        d_brier.append(brier_score_loss(yb, pn) - brier_score_loss(yb, pr))

    def summarize(vals):
        vals = np.asarray(vals, dtype=float)
        return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

    return {
        "Delta_AUC_CI": summarize(d_auc),
        "Delta_AP_CI": summarize(d_ap),
        "Delta_Brier_CI": summarize(d_brier),
    }


def _positive_cube(x):
    x = np.asarray(x, dtype=float)
    return np.maximum(x, 0.0) ** 3


def rcs_basis_4knots(x, knots):
    """
    Harrell-style restricted cubic spline basis for 4 ordered knots.

    Returns 3 columns:
      x (linear component), nonlinear_1, nonlinear_2

    Scaling of the nonlinear terms is divided by (k4-k1)^2 for numerical stability;
    this does not change the represented function space.
    """
    x = np.asarray(x, dtype=float)
    k1, k2, k3, k4 = [float(v) for v in knots]
    if not (k1 < k2 < k3 < k4):
        raise ValueError(f"RCS knots must be strictly increasing, got {knots}")

    denom_tail = (k4 - k3)
    scale = (k4 - k1) ** 2

    def h(kj):
        return (
            _positive_cube(x - kj)
            - _positive_cube(x - k3) * (k4 - kj) / denom_tail
            + _positive_cube(x - k4) * (k3 - kj) / denom_tail
        ) / scale

    return np.column_stack([x, h(k1), h(k2)])


def get_train_knots(series, var_name):
    vals = np.asarray(series, dtype=float)
    knots = np.quantile(vals, KNOT_QUANTILES)
    if len(np.unique(knots)) < 4:
        raise ValueError(
            f"{var_name}: 4-knot quantiles are not unique: {knots}. "
            "This variable is not suitable for the prespecified 4-knot RCS rule."
        )
    return knots.astype(float)


def build_rcs_design(df, knots_by_var):
    """Create RCS-expanded design matrix in a fixed column order."""
    out = pd.DataFrame(index=df.index)

    for var in RCS_VARIABLES:
        basis = rcs_basis_4knots(df[var].astype(float).values, knots_by_var[var])
        out[f"{var}__linear"] = basis[:, 0]
        out[f"{var}__rcs1"] = basis[:, 1]
        out[f"{var}__rcs2"] = basis[:, 2]

    for var in LINEAR_CONTINUOUS:
        out[var] = df[var].astype(float).values

    for var in BINARY_VARIABLES:
        out[var] = df[var].astype(float).values

    return out


def calibrate_model(base_estimator, X, y, cv=10):
    """Compatibility wrapper for sklearn versions using estimator/base_estimator."""
    try:
        model = CalibratedClassifierCV(
            estimator=base_estimator,
            method="sigmoid",
            cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_SEED),
            n_jobs=1,
        )
    except TypeError:
        model = CalibratedClassifierCV(
            base_estimator=base_estimator,
            method="sigmoid",
            cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_SEED),
        )
    model.fit(X, y)
    return model


def calibration_group_points(y, p, n_bins=10):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    groups = np.array_split(order, n_bins)
    rows = []
    for g in groups:
        if len(g) == 0:
            continue
        rows.append((float(p[g].mean()), float(y[g].mean()), len(g)))
    return rows


# ============================================================
# 2. LOAD / VERIFY LOCKED DATA
# ============================================================
print("=" * 88)
print("RCS sensitivity analysis for the current 11-predictor logistic model")
print("=" * 88)

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"Data file not found:\n{DATA_FILE}")
if not os.path.exists(PRIMARY_PRED_FILE):
    raise FileNotFoundError(
        "Locked primary prediction file not found. Expected:\n"
        f"{PRIMARY_PRED_FILE}"
    )

data = pd.read_csv(DATA_FILE, encoding="utf-8-sig")
data.columns = data.columns.str.replace("\ufeff", "", regex=False).str.strip()

required_cols = [ID_COL, SPLIT_COL, OUTCOME] + PRIMARY_FEATURES
missing = [c for c in required_cols if c not in data.columns]
if missing:
    raise ValueError(f"Missing columns in analysis-ready data: {missing}")

if len(data) != 3368:
    raise ValueError(f"Expected N=3368, got {len(data)}")
if data[ID_COL].duplicated().any():
    raise ValueError(f"Duplicate {ID_COL} values detected")

if data[OUTCOME].dtype == object:
    data[OUTCOME] = data[OUTCOME].map({
        "No": 0, "Yes": 1, "no": 0, "yes": 1,
        "0": 0, "1": 1, 0: 0, 1: 1,
    })
data[OUTCOME] = data[OUTCOME].astype(int)

for col in PRIMARY_FEATURES:
    data[col] = pd.to_numeric(data[col], errors="coerce")

if data[PRIMARY_FEATURES].isna().any().any():
    miss = data[PRIMARY_FEATURES].isna().sum()
    raise ValueError(f"Primary predictors still contain missing values: {miss[miss > 0].to_dict()}")

train_df = data[
    data[SPLIT_COL].astype(str).str.strip().str.lower() == "train"
].copy().reset_index(drop=True)

test_df = data[
    data[SPLIT_COL].astype(str).str.strip().str.lower() == "test"
].copy().reset_index(drop=True)

if len(train_df) != 2357 or int(train_df[OUTCOME].sum()) != 950:
    raise ValueError(
        f"Locked Train mismatch: N={len(train_df)}, events={int(train_df[OUTCOME].sum())}"
    )
if len(test_df) != 1011 or int(test_df[OUTCOME].sum()) != 407:
    raise ValueError(
        f"Locked Test mismatch: N={len(test_df)}, events={int(test_df[OUTCOME].sum())}"
    )

print(f"Locked Train: N={len(train_df)}, events={int(train_df[OUTCOME].sum())}")
print(f"Locked Test : N={len(test_df)}, events={int(test_df[OUTCOME].sum())}")

# ============================================================
# 3. TRAIN-ONLY KNOTS AND RCS DESIGN
# ============================================================
knots_by_var = {}
knot_rows = []

for var in RCS_VARIABLES:
    knots = get_train_knots(train_df[var], var)
    knots_by_var[var] = knots
    knot_rows.append({
        "Variable": var,
        "Knot1_q05": knots[0],
        "Knot2_q35": knots[1],
        "Knot3_q65": knots[2],
        "Knot4_q95": knots[3],
    })
    print(f"{var:12s} knots = {np.round(knots, 6)}")

save_csv(pd.DataFrame(knot_rows), "RCS_knots_train_only.csv")

X_train_raw = build_rcs_design(train_df, knots_by_var)
X_test_raw = build_rcs_design(test_df, knots_by_var)
y_train = train_df[OUTCOME].values.astype(int)
y_test = test_df[OUTCOME].values.astype(int)

if list(X_train_raw.columns) != list(X_test_raw.columns):
    raise RuntimeError("Train/Test RCS design columns differ")

# Scale all non-binary design terms using Train only.
scale_cols = [c for c in X_train_raw.columns if c not in BINARY_VARIABLES]
scaler = MinMaxScaler()
X_train = X_train_raw.copy()
X_test = X_test_raw.copy()
X_train.loc[:, scale_cols] = scaler.fit_transform(X_train_raw[scale_cols])
X_test.loc[:, scale_cols] = scaler.transform(X_test_raw[scale_cols])

save_csv(pd.DataFrame({"Design_column": X_train.columns}), "RCS_design_columns.csv")

print(f"RCS design dimension: {X_train.shape[1]} columns")
print("RCS variables:", RCS_VARIABLES)
print("Linear CCI retained as:", LINEAR_CONTINUOUS)
print("Binary variables:", BINARY_VARIABLES)

# ============================================================
# 4. TRAIN-ONLY HYPERPARAMETER TUNING
# ============================================================
base_lr = LogisticRegression(random_state=RANDOM_SEED, max_iter=5000)
cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

print("\nTuning RCS-LR using Train only...")

if HAVE_SKOPT:
    print("Using BayesSearchCV (same tuning family as the main analysis).")
    search = BayesSearchCV(
        estimator=base_lr,
        search_spaces={
            "C": Real(0.001, 100, prior="log-uniform"),
            "penalty": Categorical(["l2"]),
            "solver": Categorical(["lbfgs", "saga", "newton-cg"]),
        },
        n_iter=N_BAYES_ITER,
        cv=cv,
        scoring="roc_auc",
        random_state=RANDOM_SEED,
        n_jobs=2,
        verbose=0,
        refit=True,
    )
else:
    print("scikit-optimize not available; using deterministic 10-fold GridSearchCV fallback.")
    search = GridSearchCV(
        estimator=base_lr,
        param_grid={
            "C": np.logspace(-3, 2, 30),
            "penalty": ["l2"],
            "solver": ["lbfgs", "newton-cg"],
        },
        cv=cv,
        scoring="roc_auc",
        n_jobs=2,
        verbose=0,
        refit=True,
    )

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    search.fit(X_train.values, y_train)

best_raw_lr = search.best_estimator_
print("Best Train-CV AUC:", round(float(search.best_score_), 6))
print("Best hyperparameters:", search.best_params_)

# ============================================================
# 5. TRAIN-ONLY PLATT CALIBRATION
# ============================================================
print("\nFitting 10-fold Platt calibration on Train only...")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    rcs_calibrated = calibrate_model(
        best_raw_lr,
        X_train.values,
        y_train,
        cv=CALIBRATION_CV,
    )

p_rcs = rcs_calibrated.predict_proba(X_test.values)[:, 1]

# ============================================================
# 6. LOAD LOCKED PRIMARY LR / GBDT TEST PREDICTIONS
# ============================================================
primary = pd.read_csv(PRIMARY_PRED_FILE, encoding="utf-8-sig")
primary.columns = primary.columns.str.replace("\ufeff", "", regex=False).str.strip()

required_primary = [ID_COL, PRIMARY_LR_PROB_COL, PRIMARY_TRUE_COL]
missing_primary = [c for c in required_primary if c not in primary.columns]
if missing_primary:
    raise ValueError(
        f"Missing columns in locked primary prediction file: {missing_primary}\n"
        f"Available columns: {list(primary.columns)}"
    )

# Align locked predictions to test_df by Study_row_id.
locked_cols = [ID_COL, PRIMARY_LR_PROB_COL, PRIMARY_TRUE_COL]
if PRIMARY_GBDT_PROB_COL in primary.columns:
    locked_cols.append(PRIMARY_GBDT_PROB_COL)

aligned = test_df[[ID_COL, OUTCOME]].merge(
    primary[locked_cols],
    on=ID_COL,
    how="left",
    validate="one_to_one",
)

if aligned[PRIMARY_LR_PROB_COL].isna().any():
    raise ValueError("Could not align all locked primary LR predictions by Study_row_id")

if not np.array_equal(aligned[OUTCOME].values.astype(int), aligned[PRIMARY_TRUE_COL].values.astype(int)):
    raise ValueError("Outcome labels differ between analysis-ready data and locked primary predictions")

p_primary_lr = aligned[PRIMARY_LR_PROB_COL].values.astype(float)
p_primary_gbdt = None
if PRIMARY_GBDT_PROB_COL in aligned.columns:
    p_primary_gbdt = aligned[PRIMARY_GBDT_PROB_COL].values.astype(float)

# Verify locked primary LR AUC is the expected current result.
locked_auc = roc_auc_score(y_test, p_primary_lr)
print(f"\nLocked primary LR AUC = {locked_auc:.6f}")
if abs(locked_auc - 0.8273) > 0.002:
    print("WARNING: locked primary LR AUC differs from the manuscript value 0.8273 by >0.002")

# ============================================================
# 7. TEST METRICS
# ============================================================
metrics_primary, bins_primary = metric_dict(y_test, p_primary_lr)
metrics_rcs, bins_rcs = metric_dict(y_test, p_rcs)

summary_rows = []
for label, m in [
    ("Primary linear LR", metrics_primary),
    ("RCS-LR", metrics_rcs),
]:
    row = {"Model": label}
    row.update(m)
    summary_rows.append(row)

if p_primary_gbdt is not None:
    metrics_gbdt, bins_gbdt = metric_dict(y_test, p_primary_gbdt)
    row = {"Model": "Primary GBDT (locked context)"}
    row.update(metrics_gbdt)
    summary_rows.append(row)
else:
    metrics_gbdt = None
    bins_gbdt = None

summary_df = pd.DataFrame(summary_rows)
save_csv(summary_df, "RCS_LR_test_metric_summary.csv")

# Save calibration bins
bins_primary = bins_primary.copy(); bins_primary.insert(0, "Model", "Primary linear LR")
bins_rcs = bins_rcs.copy(); bins_rcs.insert(0, "Model", "RCS-LR")
cal_bins = [bins_primary, bins_rcs]
if bins_gbdt is not None:
    bins_gbdt = bins_gbdt.copy(); bins_gbdt.insert(0, "Model", "Primary GBDT (locked context)")
    cal_bins.append(bins_gbdt)
save_csv(pd.concat(cal_bins, ignore_index=True), "RCS_LR_calibration_bins.csv")

# ============================================================
# 8. PAIRED PRIMARY-LR vs RCS-LR COMPARISON
# ============================================================
boot = paired_bootstrap_differences(
    y_test,
    p_new=p_rcs,
    p_ref=p_primary_lr,
    n_boot=N_BOOT,
    seed=RANDOM_SEED,
)

auc_ref, auc_new, z, p_delong = delong_test(y_test, p_primary_lr, p_rcs)

d_auc = metrics_rcs["AUC"] - metrics_primary["AUC"]
d_ap = metrics_rcs["AP"] - metrics_primary["AP"]
d_brier = metrics_rcs["Brier"] - metrics_primary["Brier"]

comparison = pd.DataFrame([{
    "Comparison": "RCS-LR minus Primary linear LR",
    "Primary_LR_AUC": metrics_primary["AUC"],
    "RCS_LR_AUC": metrics_rcs["AUC"],
    "Delta_AUC": d_auc,
    "Delta_AUC_CI_low": boot["Delta_AUC_CI"][0],
    "Delta_AUC_CI_high": boot["Delta_AUC_CI"][1],
    "DeLong_Z_primary_minus_RCS": z,
    "DeLong_P": p_delong,
    "Primary_LR_AP": metrics_primary["AP"],
    "RCS_LR_AP": metrics_rcs["AP"],
    "Delta_AP": d_ap,
    "Delta_AP_CI_low": boot["Delta_AP_CI"][0],
    "Delta_AP_CI_high": boot["Delta_AP_CI"][1],
    "Primary_LR_Brier": metrics_primary["Brier"],
    "RCS_LR_Brier": metrics_rcs["Brier"],
    "Delta_Brier": d_brier,
    "Delta_Brier_CI_low": boot["Delta_Brier_CI"][0],
    "Delta_Brier_CI_high": boot["Delta_Brier_CI"][1],
    "Primary_LR_Calibration_intercept": metrics_primary["Calibration_intercept"],
    "RCS_LR_Calibration_intercept": metrics_rcs["Calibration_intercept"],
    "Primary_LR_Calibration_slope": metrics_primary["Calibration_slope"],
    "RCS_LR_Calibration_slope": metrics_rcs["Calibration_slope"],
    "Primary_LR_ECE": metrics_primary["ECE"],
    "RCS_LR_ECE": metrics_rcs["ECE"],
    "Primary_LR_MCE": metrics_primary["MCE"],
    "RCS_LR_MCE": metrics_rcs["MCE"],
    "Primary_LR_HL_P": metrics_primary["HL_P"],
    "RCS_LR_HL_P": metrics_rcs["HL_P"],
}])

save_csv(comparison, "RCS_LR_vs_PrimaryLR_summary.csv")

# ============================================================
# 9. SAVE PATIENT-LEVEL TEST PREDICTIONS
# ============================================================
pred_out = pd.DataFrame({
    ID_COL: test_df[ID_COL].values,
    "True Label": y_test,
    "Primary linear LR Prob": p_primary_lr,
    "RCS-LR Prob": p_rcs,
})
if p_primary_gbdt is not None:
    pred_out["Primary GBDT Prob"] = p_primary_gbdt
save_csv(pred_out, "RCS_LR_test_predictions_with_ID.csv")

# ============================================================
# 10. SIMPLE CALIBRATION PLOT
# ============================================================
plt.figure(figsize=(8, 7), facecolor="white")
plt.plot([0, 1], [0, 1], "--", linewidth=1.2, label="Ideal")

for label, prob in [
    ("Primary linear LR", p_primary_lr),
    ("RCS-LR", p_rcs),
]:
    pts = calibration_group_points(y_test, prob, n_bins=10)
    x = [v[0] for v in pts]
    yv = [v[1] for v in pts]
    plt.plot(x, yv, marker="o", linewidth=1.8, label=label)

if p_primary_gbdt is not None:
    pts = calibration_group_points(y_test, p_primary_gbdt, n_bins=10)
    x = [v[0] for v in pts]
    yv = [v[1] for v in pts]
    plt.plot(x, yv, marker="o", linewidth=1.8, label="Primary GBDT")

plt.xlabel("Mean predicted probability")
plt.ylabel("Observed event rate")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, "RCS_LR_calibration_plot.png")
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {plot_path}")

# ============================================================
# 11. ANALYSIS DESIGN JSON
# ============================================================
design = {
    "analysis": "Reviewer-specific RCS nonlinearity sensitivity analysis",
    "fixed_original_split": True,
    "train_n": int(len(train_df)),
    "train_events": int(y_train.sum()),
    "test_n": int(len(test_df)),
    "test_events": int(y_test.sum()),
    "primary_features": PRIMARY_FEATURES,
    "rcs_variables": RCS_VARIABLES,
    "cci_treatment": "retained as a linear score",
    "binary_variables": BINARY_VARIABLES,
    "number_of_knots": 4,
    "knot_quantiles": KNOT_QUANTILES,
    "knots_estimated_from": "fixed Train only",
    "rcs_basis": "restricted cubic spline / natural cubic tail restriction, 2 nonlinear terms plus 1 linear term per 4-knot variable",
    "scaling": "MinMax scaling fit on Train only for non-binary RCS design columns",
    "hyperparameter_search": "BayesSearchCV 10-fold Train-only if skopt available; deterministic GridSearchCV fallback otherwise",
    "bayes_iterations": N_BAYES_ITER,
    "calibration": "Platt sigmoid, 10-fold Train-only",
    "test_role": "final fixed internal comparison only",
    "bootstrap_resamples": N_BOOT,
    "best_train_cv_auc": float(search.best_score_),
    "best_params": {k: (float(v) if isinstance(v, np.floating) else v) for k, v in dict(search.best_params_).items()},
    "skopt_available": HAVE_SKOPT,
}

with open(os.path.join(OUTPUT_DIR, "RCS_analysis_design.json"), "w", encoding="utf-8") as f:
    json.dump(design, f, ensure_ascii=False, indent=2)

# ============================================================
# 12. FINAL CONSOLE SUMMARY
# ============================================================
print("\n" + "=" * 88)
print("FINAL FIXED-TEST SUMMARY")
print("=" * 88)
print(f"Primary linear LR AUC : {metrics_primary['AUC']:.4f}")
print(f"RCS-LR AUC            : {metrics_rcs['AUC']:.4f}")
print(
    f"Delta AUC (RCS-LR - primary LR): {d_auc:+.4f} "
    f"({boot['Delta_AUC_CI'][0]:+.4f} to {boot['Delta_AUC_CI'][1]:+.4f})"
)
print(f"DeLong P               : {p_delong:.4f}")
print()
print(f"Primary linear LR AP   : {metrics_primary['AP']:.4f}")
print(f"RCS-LR AP              : {metrics_rcs['AP']:.4f}")
print(
    f"Delta AP               : {d_ap:+.4f} "
    f"({boot['Delta_AP_CI'][0]:+.4f} to {boot['Delta_AP_CI'][1]:+.4f})"
)
print()
print(f"Primary linear LR Brier: {metrics_primary['Brier']:.4f}")
print(f"RCS-LR Brier           : {metrics_rcs['Brier']:.4f}")
print(
    f"Delta Brier            : {d_brier:+.4f} "
    f"({boot['Delta_Brier_CI'][0]:+.4f} to {boot['Delta_Brier_CI'][1]:+.4f})"
)
print()
print(
    "Primary linear LR calibration: "
    f"intercept={metrics_primary['Calibration_intercept']:+.4f}, "
    f"slope={metrics_primary['Calibration_slope']:.4f}, "
    f"ECE={metrics_primary['ECE']:.4f}, MCE={metrics_primary['MCE']:.4f}, "
    f"HL P={metrics_primary['HL_P']:.4g}"
)
print(
    "RCS-LR calibration           : "
    f"intercept={metrics_rcs['Calibration_intercept']:+.4f}, "
    f"slope={metrics_rcs['Calibration_slope']:.4f}, "
    f"ECE={metrics_rcs['ECE']:.4f}, MCE={metrics_rcs['MCE']:.4f}, "
    f"HL P={metrics_rcs['HL_P']:.4g}"
)

if metrics_gbdt is not None:
    print(
        "Primary GBDT context         : "
        f"AUC={metrics_gbdt['AUC']:.4f}, Brier={metrics_gbdt['Brier']:.4f}, "
        f"intercept={metrics_gbdt['Calibration_intercept']:+.4f}, "
        f"slope={metrics_gbdt['Calibration_slope']:.4f}, "
        f"ECE={metrics_gbdt['ECE']:.4f}, MCE={metrics_gbdt['MCE']:.4f}, "
        f"HL P={metrics_gbdt['HL_P']:.4g}"
    )

print("\nInterpretation rule:")
print("- If RCS materially improves calibration/Brier without loss of discrimination, the linear LR form was likely too restrictive.")
print("- If RCS does not materially improve fixed-test calibration or discrimination, the reviewer concern is empirically reduced, but the calibration trade-off versus GBDT should still be stated.")
print(f"\nOutput directory:\n{OUTPUT_DIR}")
print("=" * 88)
