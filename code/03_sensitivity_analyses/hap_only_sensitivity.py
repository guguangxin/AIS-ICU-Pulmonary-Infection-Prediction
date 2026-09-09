# -*- coding: utf-8 -*-
r"""
第三次修稿：HAP-only敏感性分析（新版11变量主分析）
====================================================

研究目的
--------
A. 排除全部334例VAP，仅分析：
   1023例 non-VAP HAP vs 2011例无肺部感染。
B. 在完全相同的HAP-only测试患者中比较：
   1) 新版主分析11变量模型；
   2) 在新版11变量基础上去除 Mechanical_ventilation 与
      Intubation_tracheotomy 后的9变量模型。

新版主分析11变量
----------------
NEUT_abs, Intubation_tracheotomy, Mechanical_ventilation, LDH,
LYMPH_abs, BUN, CCI, FIB, Surgery, Diuretics, CO2(TCO2)

去除两项气道支持后的9变量
--------------------------
NEUT_abs, LDH, LYMPH_abs, BUN, CCI, FIB, Surgery, Diuretics, CO2(TCO2)

关键原则
--------
1. 不重新随机划分患者。Primary_split直接读取主分析锁定的
   BSAfree_feature_selection_input_exactsplit.csv，并按Study_row_id与
   HAP/VAP逐患者复核文件合并。
2. 合并后强制核对原始主分析：Train=2357/events=950；
   Test=1011/events=407。
3. 再排除334例VAP，剩余HAP-only队列必须为：
   Train=2104/events=697；Test=930/events=326。
4. VAP仅由病例复核后的Pneumonia_subtype定义，绝不由MV或插管变量推断。
5. 两套特征使用相同8个算法、与主分析一致的Bayesian搜索空间和
   10-fold Platt calibration。
6. 重点报告AUC、AP、Brier、calibration intercept/slope、ECE/MCE，
   并给出1000次患者级bootstrap 95% percentile CI。
7. 11变量与去除MV/插管后的9变量，在同一HAP-only测试患者中进行
   paired DeLong检验（8模型Holm校正），并计算配对bootstrap的
   ΔAUC、ΔAP、ΔBrier 95%CI。

Public repository usage
-----------------------
Patient-level inputs are not included. Set AIS_ICU_DATA_DIR to a local restricted-data
directory containing the exact-split file and the adjudicated Pneumonia_subtype file.
Outputs are written under AIS_ICU_OUTPUT_DIR (default: <repo>/outputs).
"""

import os
import sys
import gc
import json
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

from scipy import stats
from scipy.special import logit
from scipy.optimize import minimize

from sklearn.base import clone
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical


# ============================================================
# 1. 路径
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
PROJECT_DIR = Path(
    os.environ.get("AIS_ICU_DATA_DIR", str(REPO_ROOT / "restricted_data"))
).expanduser().resolve()
OUTPUT_ROOT = Path(
    os.environ.get("AIS_ICU_OUTPUT_DIR", str(REPO_ROOT / "outputs"))
).expanduser().resolve()
DATA_DIR = OUTPUT_ROOT / "hap_only_sensitivity"

OUT_11 = DATA_DIR / "01_HAP_only_11predictors"
OUT_9 = DATA_DIR / "02_HAP_only_9predictors_no_MV_no_intubation"
CKPT_11 = OUT_11 / "checkpoints"
CKPT_9 = OUT_9 / "checkpoints"

for p in [DATA_DIR, OUT_11, OUT_9, CKPT_11, CKPT_9]:
    p.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. 全局参数
# ============================================================
RANDOM_STATE = 42
TEST_SIZE = 0.30
TUNING_CV = 10
CALIBRATION_CV = 10
BOOTSTRAP_N = 1000

GLOBAL_N_JOBS = 2
MODEL_N_JOBS = {"MLP": 1, "RF": 2}

OUTCOME = "Pulmonary_infection"
SUBTYPE = "Pneumonia_subtype"

FEATURES_11 = [
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
    "CO2",  # manuscript / R output: TCO2
]

FEATURES_9_NO_AIRWAY = [
    "NEUT_abs",
    "LDH",
    "LYMPH_abs",
    "BUN",
    "CCI",
    "FIB",
    "Surgery",
    "Diuretics",
    "CO2",  # manuscript / R output: TCO2
]

EXPECTED = {
    "total": 3368,
    "no_infection": 2011,
    "hap": 1023,
    "vap": 334,
    "primary_train_n": 2357,
    "primary_train_events": 950,
    "primary_test_n": 1011,
    "primary_test_events": 407,
    "hap_train_n": 2104,
    "hap_train_events": 697,
    "hap_test_n": 930,
    "hap_test_events": 326,
}

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

MODEL_N_ITER = {
    "RF": 60,
    "LightGBM": 50,
    "XGBoost": 50,
}
DEFAULT_N_ITER = 40


# ============================================================
# 3. 基础工具
# ============================================================
def save_csv(df, path):
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  已保存: {path}")


def normalize_primary_outcome(s):
    if pd.api.types.is_numeric_dtype(s):
        out = pd.to_numeric(s, errors="coerce")
    else:
        mp = {
            "No": 0, "no": 0, "NO": 0, "0": 0, "0.0": 0,
            "Yes": 1, "yes": 1, "YES": 1, "1": 1, "1.0": 1,
        }
        out = s.astype(str).str.strip().map(mp)

    if out.isna().any():
        bad = s[out.isna()].astype(str).value_counts().head(20).to_dict()
        raise ValueError(f"{OUTCOME}存在无法识别的取值: {bad}")

    out = out.astype(int)
    if not set(out.unique()).issubset({0, 1}):
        raise ValueError(f"{OUTCOME}必须为0/1。")
    return out


def normalize_subtype(s):
    """
    输出统一编码：
    0 = 无感染
    1 = HAP
    2 = VAP
    """
    if pd.api.types.is_numeric_dtype(s):
        out = pd.to_numeric(s, errors="coerce")
    else:
        x = s.astype(str).str.strip().str.lower()
        mp = {
            "0": 0, "0.0": 0,
            "no infection": 0,
            "no pulmonary infection": 0,
            "none": 0,
            "non-infection": 0,

            "1": 1, "1.0": 1,
            "hap": 1,
            "non-vap hap": 1,
            "non-vap": 1,
            "hospital-acquired pneumonia": 1,

            "2": 2, "2.0": 2,
            "vap": 2,
            "ventilator-associated pneumonia": 2,
        }
        out = x.map(mp)

    if out.isna().any():
        return None

    out = out.astype(int)
    if not set(out.unique()).issubset({0, 1, 2}):
        return None
    return out


def _clean_columns(df):
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )
    return df


def find_exact_split_csv():
    """寻找锁定Primary_split的主分析输入文件。"""
    preferred = PROJECT_DIR / "BSAfree_feature_selection_input_exactsplit.csv"
    candidates = []
    if preferred.exists():
        candidates.append(preferred)
    candidates.extend(
        p for p in PROJECT_DIR.rglob("BSAfree_feature_selection_input_exactsplit.csv")
        if p not in candidates
    )

    for path in candidates:
        try:
            df = _clean_columns(pd.read_csv(path, encoding="utf-8-sig"))
            needed = {"Study_row_id", "Primary_split", OUTCOME}
            if not needed.issubset(df.columns):
                continue
            if len(df) != EXPECTED["total"]:
                continue
            if df["Study_row_id"].duplicated().any():
                continue
            split_vals = set(df["Primary_split"].astype(str).str.strip().unique())
            if split_vals != {"Train", "Test"}:
                continue
            return path
        except Exception:
            continue

    raise FileNotFoundError(
        "没有找到有效的 BSAfree_feature_selection_input_exactsplit.csv。\n"
        f"请把文件放到：{PROJECT_DIR} 或其子目录。"
    )


def find_valid_input_csv():
    """
    自动寻找已经完成HAP/VAP逐患者标记的CSV。
    接受条件：
    - 3368行；
    - Study_row_id唯一；
    - 有Pulmonary_infection和Pneumonia_subtype；
    - 新版11个预测变量齐全；
    - subtype计数0=2011, 1=1023, 2=334；
    - subtype与Pulmonary_infection逻辑一致。
    """
    preferred_names = [
        "英文列名准备变量筛选_HAPVAP复核_待填写.csv",
        "英文列名准备变量筛选_HAPVAP复核.csv",
    ]
    candidates = []
    for name in preferred_names:
        p = PROJECT_DIR / name
        if p.exists():
            candidates.append(p)
    # 递归搜索，但排除本脚本新生成的结果CSV
    for p in PROJECT_DIR.rglob("*.csv"):
        if p in candidates:
            continue
        if DATA_DIR in p.parents:
            continue
        candidates.append(p)

    candidates = sorted(
        candidates,
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    valid = []
    diagnostics = []
    required = ["Study_row_id", OUTCOME, SUBTYPE, *FEATURES_11]

    for path in candidates:
        try:
            df = _clean_columns(pd.read_csv(path, encoding="utf-8-sig"))
            missing = [c for c in required if c not in df.columns]
            if missing:
                continue
            if len(df) != EXPECTED["total"]:
                diagnostics.append((path.name, f"行数={len(df)}，不是3368"))
                continue
            if df["Study_row_id"].duplicated().any():
                diagnostics.append((path.name, "Study_row_id重复"))
                continue

            subtype = normalize_subtype(df[SUBTYPE])
            if subtype is None:
                diagnostics.append((path.name, "Pneumonia_subtype不完整或有无法识别值"))
                continue
            primary = normalize_primary_outcome(df[OUTCOME])

            counts = subtype.value_counts().to_dict()
            if (
                int(counts.get(0, 0)),
                int(counts.get(1, 0)),
                int(counts.get(2, 0)),
            ) != (
                EXPECTED["no_infection"],
                EXPECTED["hap"],
                EXPECTED["vap"],
            ):
                diagnostics.append((path.name, f"亚型计数不符: {counts}"))
                continue

            bad = (
                ((subtype == 0) & (primary != 0))
                | ((subtype.isin([1, 2])) & (primary != 1))
            )
            if int(bad.sum()) != 0:
                diagnostics.append((path.name, f"亚型与结局矛盾={int(bad.sum())}行"))
                continue

            valid.append(path)
        except Exception as e:
            diagnostics.append((path.name, f"读取失败: {e}"))

    if not valid:
        print("\n未找到有效的HAP/VAP逐患者复核CSV。")
        for name, status in diagnostics[:30]:
            print(f"  - {name}: {status}")
        raise FileNotFoundError(
            f"请把已填好Pneumonia_subtype的3368例CSV放到 {PROJECT_DIR} 或其子目录。"
        )

    if len(valid) > 1:
        print("\n检测到多个有效HAP/VAP文件，将使用最近修改的一个：")
    print(f"  {valid[0]}")
    return valid[0]


# ============================================================
# 4. 校准指标
# ============================================================
def calibration_intercept_slope(y_true, y_prob, eps=1e-6):
    """
    Calibration intercept:
       logit(y) = logit(p) + intercept，固定斜率为1
    Calibration slope:
       logit(y) = a + slope * logit(p)
    """
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(y_prob, dtype=float), eps, 1 - eps)
    lp = logit(p)

    # slope
    try:
        lr = LogisticRegression(
            penalty=None, solver="lbfgs", max_iter=3000
        )
        lr.fit(lp.reshape(-1, 1), y)
    except Exception:
        lr = LogisticRegression(
            penalty="l2", C=1e12,
            solver="lbfgs", max_iter=3000
        )
        lr.fit(lp.reshape(-1, 1), y)

    slope = float(lr.coef_[0, 0])

    # intercept with slope fixed at 1
    def neg_ll(b):
        eta = lp + b[0]
        log_p = -np.logaddexp(0, -eta)
        log_1mp = -np.logaddexp(0, eta)
        return -(y * log_p + (1 - y) * log_1mp).sum()

    res = minimize(
        neg_ll,
        x0=np.array([0.0]),
        method="L-BFGS-B"
    )
    intercept = float(res.x[0])

    return intercept, slope


def calibration_error(y_true, y_prob, n_bins=10):
    """
    Equal-sized bins:
    ECE = sum(bin fraction * abs(observed - predicted))
    MCE = max abs(observed - predicted)
    """
    y = np.asarray(y_true)
    p = np.asarray(y_prob)

    order = np.argsort(p)
    bins = np.array_split(order, n_bins)

    ece = 0.0
    mce = 0.0
    rows = []

    for i, idx in enumerate(bins, 1):
        if len(idx) == 0:
            continue

        mean_p = float(np.mean(p[idx]))
        mean_y = float(np.mean(y[idx]))
        err = abs(mean_p - mean_y)

        ece += len(idx) / len(y) * err
        mce = max(mce, err)

        rows.append({
            "Bin": i,
            "N": len(idx),
            "Mean_predicted_probability": mean_p,
            "Observed_event_rate": mean_y,
            "Absolute_error": err,
        })

    return float(ece), float(mce), pd.DataFrame(rows)


def point_metrics(y_true, y_prob):
    intercept, slope = calibration_intercept_slope(
        y_true, y_prob
    )
    ece, mce, _ = calibration_error(
        y_true, y_prob, n_bins=10
    )

    return {
        "AUC": float(roc_auc_score(y_true, y_prob)),
        "AP": float(average_precision_score(y_true, y_prob)),
        "Brier": float(brier_score_loss(y_true, y_prob)),
        "Calibration_intercept": intercept,
        "Calibration_slope": slope,
        "ECE": ece,
        "MCE": mce,
    }


# ============================================================
# 5. Bootstrap CI
# ============================================================
def bootstrap_metric_ci(
    y_true,
    y_prob,
    n_boot=1000,
    random_state=42
):
    y = np.asarray(y_true)
    p = np.asarray(y_prob)

    rng = np.random.default_rng(random_state)
    store = {
        "AUC": [],
        "AP": [],
        "Brier": [],
        "Calibration_intercept": [],
        "Calibration_slope": [],
        "ECE": [],
        "MCE": [],
    }

    n = len(y)

    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        pb = p[idx]

        if np.unique(yb).size < 2:
            continue

        try:
            m = point_metrics(yb, pb)
            for key in store:
                store[key].append(m[key])
        except Exception:
            continue

        if (b + 1) % 200 == 0:
            print(f"      bootstrap {b+1}/{n_boot}")

    result = {}
    for key, vals in store.items():
        arr = np.asarray(vals, dtype=float)
        if len(arr) == 0:
            lo, hi = np.nan, np.nan
        else:
            lo = float(np.percentile(arr, 2.5))
            hi = float(np.percentile(arr, 97.5))
        result[key] = (lo, hi)

    return result


# ============================================================
# 6. DeLong
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
        tz[r] = _compute_midrank(
            predictions_sorted_transposed[r]
        )

    aucs = (
        tz[:, :m].sum(axis=1) / m / n
        - float(m + 1) / 2.0 / n
    )

    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m

    sx = np.cov(v01)
    sy = np.cov(v10)
    cov = sx / m + sy / n

    return aucs, cov


def paired_delong(y_true, p9, p7):
    y = np.asarray(y_true)
    p9 = np.asarray(p9)
    p7 = np.asarray(p7)

    order = np.argsort(y)[::-1]
    pred = np.vstack([p9[order], p7[order]])

    aucs, cov = _fast_delong(
        pred,
        int(y.sum())
    )

    var = (
        cov[0, 0]
        + cov[1, 1]
        - 2 * cov[0, 1]
    )
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


def paired_bootstrap_difference(
    y_true,
    p11,
    p9,
    n_boot=1000,
    random_state=2026
):
    """
    返回去除MV/插管后的9变量 - 新版11变量的
    ΔAUC / ΔAP / ΔBrier 95% percentile CI。
    """
    y = np.asarray(y_true)
    p11 = np.asarray(p11)
    p9 = np.asarray(p9)

    rng = np.random.default_rng(random_state)
    da, dap, db = [], [], []
    n = len(y)

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        if np.unique(yb).size < 2:
            continue

        q11 = p11[idx]
        q9 = p9[idx]
        da.append(roc_auc_score(yb, q9) - roc_auc_score(yb, q11))
        dap.append(average_precision_score(yb, q9) - average_precision_score(yb, q11))
        db.append(brier_score_loss(yb, q9) - brier_score_loss(yb, q11))

    def ci(vals):
        arr = np.asarray(vals, dtype=float)
        return (
            float(np.percentile(arr, 2.5)),
            float(np.percentile(arr, 97.5)),
        )

    return {
        "Delta_AUC_9minus11": ci(da),
        "Delta_AP_9minus11": ci(dap),
        "Delta_Brier_9minus11": ci(db),
    }


# ============================================================
# 7. 数据预处理
# ============================================================
def split_feature_types(X_train):
    binary = []
    continuous = []

    for col in X_train.columns:
        nunique = X_train[col].nunique(dropna=True)
        if nunique <= 5 or X_train[col].dtype == object:
            binary.append(col)
        else:
            continuous.append(col)

    return binary, continuous


def transform_train_test(X_train_raw, X_test_raw):
    """
    为保持与原主分析一致：
    连续变量MinMaxScaler，仅在当前HAP-only训练集拟合；
    二分类变量不缩放。
    当前分析数据应为analysis-ready数据，因此若发现缺失直接停止，
    不在敏感性分析中偷偷新增一套缺失值处理规则。
    """
    if X_train_raw.isna().any().any() or X_test_raw.isna().any().any():
        miss = pd.concat(
            [
                X_train_raw.isna().sum().rename("Train_missing"),
                X_test_raw.isna().sum().rename("Test_missing"),
            ],
            axis=1
        )
        miss = miss[(miss > 0).any(axis=1)]
        raise ValueError(
            "敏感性分析变量发现缺失值，请先核查：\n"
            + miss.to_string()
        )

    binary, continuous = split_feature_types(
        X_train_raw
    )

    Xtr = X_train_raw.copy()
    Xte = X_test_raw.copy()

    scaler = MinMaxScaler()

    if continuous:
        Xtr.loc[:, continuous] = scaler.fit_transform(
            X_train_raw[continuous]
        )
        Xte.loc[:, continuous] = scaler.transform(
            X_test_raw[continuous]
        )

    return Xtr, Xte, scaler, binary, continuous


# ============================================================
# 8. 模型配置：与原主分析保持一致
# ============================================================
def model_config():
    return {
        "RF": {
            "model": RandomForestClassifier(
                random_state=42,
                n_jobs=1,
                oob_score=True
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
            }
        },

        "GBDT": {
            "model": GradientBoostingClassifier(
                random_state=42
            ),
            "params": {
                "n_estimators": Integer(100, 400),
                "max_depth": Integer(3, 7),
                "learning_rate": Real(
                    0.01, 0.15,
                    prior="log-uniform"
                ),
                "min_samples_split": Integer(5, 30),
                "min_samples_leaf": Integer(5, 20),
                "subsample": Real(0.7, 0.9),
                "max_features": Categorical(
                    ["sqrt", "log2"]
                ),
            }
        },

        "LR": {
            "model": LogisticRegression(
                random_state=42,
                max_iter=3000
            ),
            "params": {
                "C": Real(
                    0.001, 100,
                    prior="log-uniform"
                ),
                "penalty": Categorical(["l2"]),
                "solver": Categorical(
                    ["lbfgs", "saga", "newton-cg"]
                ),
            }
        },

        "NB": {
            "model": GaussianNB(),
            "params": {
                "var_smoothing": Real(
                    1e-11, 1e-5,
                    prior="log-uniform"
                )
            }
        },

        "DT": {
            "model": DecisionTreeClassifier(
                random_state=42
            ),
            "params": {
                "max_depth": Integer(3, 15),
                "min_samples_split": Integer(10, 50),
                "min_samples_leaf": Integer(5, 20),
                "criterion": Categorical(
                    ["gini", "entropy"]
                ),
                "max_features": Categorical(
                    ["sqrt", "log2"]
                ),
                "ccp_alpha": Real(0.001, 0.05),
            }
        },

        "LightGBM": {
            "model": LGBMClassifier(
                random_state=42,
                verbosity=-1,
                n_jobs=1,
                force_col_wise=True
            ),
            "params": {
                "n_estimators": Integer(100, 400),
                "max_depth": Integer(3, 7),
                "learning_rate": Real(
                    0.01, 0.1,
                    prior="log-uniform"
                ),
                "num_leaves": Integer(10, 40),
                "min_child_samples": Integer(20, 60),
                "reg_alpha": Real(
                    0.05, 2.0,
                    prior="log-uniform"
                ),
                "reg_lambda": Real(
                    0.05, 2.0,
                    prior="log-uniform"
                ),
                "feature_fraction": Real(0.5, 0.85),
                "bagging_fraction": Real(0.5, 0.85),
                "bagging_freq": Integer(1, 7),
            }
        },

        "XGBoost": {
            "model": XGBClassifier(
                random_state=42,
                verbosity=0,
                eval_metric="logloss",
                n_jobs=1
            ),
            "params": {
                "n_estimators": Integer(100, 400),
                "max_depth": Integer(3, 6),
                "learning_rate": Real(
                    0.01, 0.1,
                    prior="log-uniform"
                ),
                "subsample": Real(0.6, 0.85),
                "colsample_bytree": Real(0.6, 0.85),
                "reg_alpha": Real(
                    0.01, 2.0,
                    prior="log-uniform"
                ),
                "reg_lambda": Real(
                    0.1, 5.0,
                    prior="log-uniform"
                ),
                "min_child_weight": Integer(3, 10),
                "gamma": Real(0.1, 1.0),
            }
        },

        "MLP": {
            "model": MLPClassifier(
                random_state=42,
                max_iter=500,
                hidden_layer_sizes=(64, 32),
                early_stopping=True,
                validation_fraction=0.1
            ),
            "params": {
                "activation": Categorical(
                    ["relu", "tanh"]
                ),
                "alpha": Real(
                    1e-4, 1e-1,
                    prior="log-uniform"
                ),
                "learning_rate_init": Real(
                    1e-4, 5e-3,
                    prior="log-uniform"
                ),
                "batch_size": Integer(32, 128),
            }
        },
    }


def calibrate_model(estimator, X, y):
    try:
        model = CalibratedClassifierCV(
            estimator=estimator,
            method="sigmoid",
            cv=CALIBRATION_CV,
            n_jobs=1
        )
    except TypeError:
        model = CalibratedClassifierCV(
            base_estimator=estimator,
            method="sigmoid",
            cv=CALIBRATION_CV,
            n_jobs=1
        )

    model.fit(X, y)
    return model


# ============================================================
# 9. 读取并核查病例数据
# ============================================================
def prepare_data():
    subtype_csv = find_valid_input_csv()
    split_csv = find_exact_split_csv()

    print(f"\nHAP/VAP逐患者文件：\n{subtype_csv}")
    print(f"\n锁定主分析Train/Test身份文件：\n{split_csv}")

    df = _clean_columns(pd.read_csv(subtype_csv, encoding="utf-8-sig"))
    split_df = _clean_columns(pd.read_csv(split_csv, encoding="utf-8-sig"))

    df[OUTCOME] = normalize_primary_outcome(df[OUTCOME])
    df[SUBTYPE] = normalize_subtype(df[SUBTYPE])
    split_df[OUTCOME] = normalize_primary_outcome(split_df[OUTCOME])

    if df["Study_row_id"].duplicated().any() or split_df["Study_row_id"].duplicated().any():
        raise ValueError("Study_row_id存在重复，请停止核查。")

    # 只从锁定split文件取Primary_split和主结局作交叉核对。
    split_small = split_df[["Study_row_id", "Primary_split", OUTCOME]].rename(
        columns={OUTCOME: "Outcome_in_exactsplit"}
    )
    df = df.merge(split_small, on="Study_row_id", how="left", validate="one_to_one")

    if df["Primary_split"].isna().any():
        raise RuntimeError("存在病例无法匹配到锁定Primary_split。")
    if not np.array_equal(df[OUTCOME].values, df["Outcome_in_exactsplit"].values):
        mismatch = int((df[OUTCOME] != df["Outcome_in_exactsplit"]).sum())
        raise RuntimeError(f"HAP/VAP文件与exactsplit主结局不一致：{mismatch}行。")
    df = df.drop(columns=["Outcome_in_exactsplit"])

    subtype_summary = pd.DataFrame({
        "Subtype": ["No pulmonary infection", "Non-VAP HAP", "VAP"],
        "Code": [0, 1, 2],
        "N": [
            int((df[SUBTYPE] == 0).sum()),
            int((df[SUBTYPE] == 1).sum()),
            int((df[SUBTYPE] == 2).sum()),
        ],
    })
    subtype_summary["Percent_of_total"] = subtype_summary["N"] / len(df) * 100
    save_csv(subtype_summary, DATA_DIR / "00_重新复核HAP_VAP亚型总计.csv")

    # 原始主分析锁定分组核对
    full_check = (
        df.groupby("Primary_split")[OUTCOME]
        .agg(N="size", Events="sum")
        .reset_index()
    )
    full_check["Event_rate"] = full_check["Events"] / full_check["N"] * 100
    save_csv(full_check, DATA_DIR / "00_锁定原主分析TrainTest身份核查.csv")

    train_row = full_check[full_check["Primary_split"] == "Train"].iloc[0]
    test_row = full_check[full_check["Primary_split"] == "Test"].iloc[0]
    if not (
        int(train_row["N"]) == EXPECTED["primary_train_n"]
        and int(train_row["Events"]) == EXPECTED["primary_train_events"]
        and int(test_row["N"]) == EXPECTED["primary_test_n"]
        and int(test_row["Events"]) == EXPECTED["primary_test_events"]
    ):
        raise RuntimeError(
            "锁定主分析分组不符合预期。应为Train=2357/events=950；"
            "Test=1011/events=407。"
        )

    print("\n锁定原主分析分组核对通过：")
    print(full_check.to_string(index=False))

    # 排除全部VAP，保留锁定身份
    hap = df[df[SUBTYPE] != 2].copy()
    hap["HAP_outcome"] = (hap[SUBTYPE] == 1).astype(int)

    if len(hap) != EXPECTED["no_infection"] + EXPECTED["hap"]:
        raise RuntimeError(f"HAP-only队列应为3034人，实际{len(hap)}。")
    if int(hap["HAP_outcome"].sum()) != EXPECTED["hap"]:
        raise RuntimeError(f"HAP事件应为1023，实际{int(hap['HAP_outcome'].sum())}。")

    audit_rows = []
    for split in ["Train", "Test"]:
        full_s = df[df["Primary_split"] == split]
        hap_s = hap[hap["Primary_split"] == split]
        audit_rows.append({
            "Split": split,
            "Original_N": len(full_s),
            "Original_infection_events": int(full_s[OUTCOME].sum()),
            "VAP_excluded": int((full_s[SUBTYPE] == 2).sum()),
            "HAP_only_N": len(hap_s),
            "HAP_events": int(hap_s["HAP_outcome"].sum()),
            "No_infection": int((hap_s["HAP_outcome"] == 0).sum()),
            "HAP_event_rate_percent": float(hap_s["HAP_outcome"].mean() * 100),
        })
    audit = pd.DataFrame(audit_rows)
    save_csv(audit, DATA_DIR / "00_HAP_only队列划分与VAP排除核查.csv")

    train_h = audit[audit["Split"] == "Train"].iloc[0]
    test_h = audit[audit["Split"] == "Test"].iloc[0]
    if not (
        int(train_h["HAP_only_N"]) == EXPECTED["hap_train_n"]
        and int(train_h["HAP_events"]) == EXPECTED["hap_train_events"]
        and int(test_h["HAP_only_N"]) == EXPECTED["hap_test_n"]
        and int(test_h["HAP_events"]) == EXPECTED["hap_test_events"]
    ):
        raise RuntimeError(
            "HAP-only锁定分组与既往核对值不一致。应为"
            "Train=2104/events=697；Test=930/events=326。"
        )

    save_csv(
        df[["Study_row_id", OUTCOME, SUBTYPE, "Primary_split"]],
        DATA_DIR / "00_患者亚型与锁定TrainTest身份_追溯表.csv",
    )

    print("\nHAP-only队列核对通过：")
    print(audit.to_string(index=False))
    return df, hap, subtype_csv, split_csv


# ============================================================
# 10. 跑一套特征模型
# ============================================================
def run_feature_set(
    hap,
    features,
    output_dir,
    checkpoint_dir,
    label
):
    train_df = hap[
        hap["Primary_split"] == "Train"
    ].copy()

    test_df = hap[
        hap["Primary_split"] == "Test"
    ].copy()

    X_train_raw = train_df[features].copy()
    X_test_raw = test_df[features].copy()

    y_train = train_df["HAP_outcome"].astype(int)
    y_test = test_df["HAP_outcome"].astype(int)

    (
        X_train,
        X_test,
        scaler,
        binary_features,
        continuous_features
    ) = transform_train_test(
        X_train_raw,
        X_test_raw
    )

    Xtr = X_train.values
    Xte = X_test.values

    joblib.dump(
        scaler,
        output_dir / "MinMaxScaler.pkl",
        compress=3
    )

    setup = {
        "Analysis": label,
        "Features": features,
        "Train_N": len(y_train),
        "Train_HAP_events": int(y_train.sum()),
        "Test_N": len(y_test),
        "Test_HAP_events": int(y_test.sum()),
        "Binary_features": binary_features,
        "Continuous_scaled_features": continuous_features,
        "Random_state": RANDOM_STATE,
        "Test_size_original_primary": TEST_SIZE,
        "Primary_split_preserved": True,
        "VAP_excluded_total": EXPECTED["vap"],
        "Bayesian_tuning_CV": TUNING_CV,
        "Platt_calibration_CV": CALIBRATION_CV,
        "Bootstrap_N": BOOTSTRAP_N,
    }

    with open(
        output_dir / "分析设计与参数.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            setup,
            f,
            ensure_ascii=False,
            indent=2
        )

    configs = model_config()

    results = {}
    predictions = {}
    params_rows = []

    for model_key, cfg in configs.items():
        full_name = MODEL_FULL_NAME[model_key]
        ckpt_file = checkpoint_dir / f"{model_key}.pkl"

        print("\n" + "=" * 78)
        print(f"{label}")
        print(f"Model: {full_name}")
        print("=" * 78)

        # 断点续跑
        if ckpt_file.exists():
            try:
                saved = joblib.load(ckpt_file)
                if (
                    saved.get("features") == features
                    and saved.get("test_row_ids")
                    == test_df["Study_row_id"].tolist()
                ):
                    print("  从断点读取成功。")
                    results[model_key] = saved["result"]
                    predictions[model_key] = np.asarray(
                        saved["test_probability"]
                    )
                    params_rows.append({
                        "Model": full_name,
                        "Best_parameters": json.dumps(
                            saved["best_params"],
                            ensure_ascii=False,
                            default=str
                        )
                    })
                    continue
            except Exception as e:
                print(f"  旧断点不可用，重新训练: {e}")

        start = time.time()

        n_iter = MODEL_N_ITER.get(
            model_key,
            DEFAULT_N_ITER
        )

        n_jobs = MODEL_N_JOBS.get(
            model_key,
            GLOBAL_N_JOBS
        )

        print(
            f"  Bayesian tuning: "
            f"n_iter={n_iter}, CV={TUNING_CV}"
        )

        search = BayesSearchCV(
            estimator=cfg["model"],
            search_spaces=cfg["params"],
            n_iter=n_iter,
            cv=StratifiedKFold(
                n_splits=TUNING_CV,
                shuffle=True,
                random_state=RANDOM_STATE
            ),
            scoring="roc_auc",
            random_state=RANDOM_STATE,
            n_jobs=n_jobs,
            verbose=0,
            refit=True
        )

        search.fit(
            Xtr,
            y_train.values
        )

        best_estimator = search.best_estimator_
        best_params = dict(search.best_params_)

        print(
            f"  Training CV AUC = "
            f"{search.best_score_:.4f}"
        )

        print(
            f"  Platt calibration: "
            f"{CALIBRATION_CV}-fold"
        )

        calibrated = calibrate_model(
            clone(best_estimator),
            Xtr,
            y_train.values
        )

        test_prob = calibrated.predict_proba(
            Xte
        )[:, 1]

        metrics = point_metrics(
            y_test.values,
            test_prob
        )

        print(
            "  Test: "
            f"AUC={metrics['AUC']:.4f}, "
            f"AP={metrics['AP']:.4f}, "
            f"Brier={metrics['Brier']:.4f}, "
            f"Cal.Int={metrics['Calibration_intercept']:+.3f}, "
            f"Cal.Slope={metrics['Calibration_slope']:.3f}"
        )

        print(
            f"  Bootstrap 95%CI: "
            f"{BOOTSTRAP_N} iterations"
        )

        ci = bootstrap_metric_ci(
            y_test.values,
            test_prob,
            n_boot=BOOTSTRAP_N,
            random_state=RANDOM_STATE
        )

        result = {
            "Model": full_name,
            "Training_CV_AUC": float(
                search.best_score_
            ),
            **metrics,
        }

        for key, (lo, hi) in ci.items():
            result[f"{key}_95CI_low"] = lo
            result[f"{key}_95CI_high"] = hi

        results[model_key] = result
        predictions[model_key] = test_prob

        _, _, bin_df = calibration_error(
            y_test.values,
            test_prob,
            n_bins=10
        )

        save_csv(
            bin_df,
            output_dir
            / f"Calibration_bins_{model_key}.csv"
        )

        params_rows.append({
            "Model": full_name,
            "Best_parameters": json.dumps(
                best_params,
                ensure_ascii=False,
                default=str
            )
        })

        joblib.dump(
            {
                "features": features,
                "test_row_ids":
                    test_df["Study_row_id"].tolist(),
                "result": result,
                "test_probability": test_prob,
                "best_params": best_params,
                "model": calibrated,
                "scaler": scaler,
            },
            ckpt_file,
            compress=3
        )

        print(
            f"  完成，用时 "
            f"{(time.time()-start)/60:.1f} min"
        )

        del search, calibrated, best_estimator
        gc.collect()

    # 模型性能表
    perf = pd.DataFrame(
        list(results.values())
    ).sort_values(
        "AUC",
        ascending=False
    ).reset_index(drop=True)

    save_csv(
        perf,
        output_dir / "模型性能_HAP_only.csv"
    )

    # 参数表
    params_df = pd.DataFrame(params_rows)
    params_df = params_df.drop_duplicates(
        subset=["Model"],
        keep="last"
    )
    save_csv(
        params_df,
        output_dir / "最佳超参数.csv"
    )

    # 测试集逐患者预测概率
    pred_df = test_df[
        [
            "Study_row_id",
            SUBTYPE,
            "HAP_outcome"
        ]
    ].copy().reset_index(drop=True)

    for key, prob in predictions.items():
        pred_df[
            f"{MODEL_FULL_NAME[key]}_Probability"
        ] = prob

    save_csv(
        pred_df,
        output_dir / "测试集逐患者预测概率.csv"
    )

    return {
        "results": results,
        "predictions": predictions,
        "y_test": y_test.reset_index(drop=True),
        "test_row_ids": test_df[
            "Study_row_id"
        ].reset_index(drop=True),
    }


# ============================================================
# 11. 11变量 vs 去MV/插管9变量配对比较
# ============================================================
def compare_11_vs_9(res11, res9):
    if not np.array_equal(res11["y_test"].values, res9["y_test"].values):
        raise RuntimeError("11变量和去气道支持9变量的测试集标签不一致。")
    if not np.array_equal(res11["test_row_ids"].values, res9["test_row_ids"].values):
        raise RuntimeError("11变量和去气道支持9变量不是同一批测试患者。")

    y = res11["y_test"].values
    model_keys = [k for k in res11["predictions"] if k in res9["predictions"]]
    rows, raw_p = [], []

    print("\n" + "=" * 78)
    print("HAP-only：11变量 vs 去除MV/插管后的9变量（同一测试患者）")
    print("=" * 78)

    for key in model_keys:
        p11 = res11["predictions"][key]
        p9 = res9["predictions"][key]

        # paired_delong返回第一个预测 - 第二个预测的Z方向
        auc11, auc9, z, p = paired_delong(y, p11, p9)
        diffs = paired_bootstrap_difference(
            y, p11, p9, n_boot=BOOTSTRAP_N, random_state=2026
        )

        m11 = res11["results"][key]
        m9 = res9["results"][key]
        row = {
            "Model": MODEL_FULL_NAME[key],
            "AUC_11pred": auc11,
            "AUC_9pred_no_airway": auc9,
            "Delta_AUC_9minus11": auc9 - auc11,
            "Delta_AUC_95CI_low": diffs["Delta_AUC_9minus11"][0],
            "Delta_AUC_95CI_high": diffs["Delta_AUC_9minus11"][1],
            "DeLong_Z_11minus9": z,
            "DeLong_P": p,
            "AP_11pred": m11["AP"],
            "AP_9pred_no_airway": m9["AP"],
            "Delta_AP_9minus11": m9["AP"] - m11["AP"],
            "Delta_AP_95CI_low": diffs["Delta_AP_9minus11"][0],
            "Delta_AP_95CI_high": diffs["Delta_AP_9minus11"][1],
            "Brier_11pred": m11["Brier"],
            "Brier_9pred_no_airway": m9["Brier"],
            "Delta_Brier_9minus11": m9["Brier"] - m11["Brier"],
            "Delta_Brier_95CI_low": diffs["Delta_Brier_9minus11"][0],
            "Delta_Brier_95CI_high": diffs["Delta_Brier_9minus11"][1],
            "Calibration_intercept_11pred": m11["Calibration_intercept"],
            "Calibration_intercept_9pred_no_airway": m9["Calibration_intercept"],
            "Calibration_slope_11pred": m11["Calibration_slope"],
            "Calibration_slope_9pred_no_airway": m9["Calibration_slope"],
            "ECE_11pred": m11["ECE"],
            "ECE_9pred_no_airway": m9["ECE"],
            "MCE_11pred": m11["MCE"],
            "MCE_9pred_no_airway": m9["MCE"],
        }
        rows.append(row)
        raw_p.append(p)

        print(
            f"  {MODEL_FULL_NAME[key]:35s} "
            f"AUC11={auc11:.3f}, AUC9(no airway)={auc9:.3f}, "
            f"Δ(9-11)={auc9-auc11:+.3f}, P={p:.4f}"
        )

    adjusted = holm_adjust(raw_p)
    for row, adj in zip(rows, adjusted):
        row["DeLong_P_Holm"] = float(adj)

    df = pd.DataFrame(rows).sort_values("AUC_11pred", ascending=False).reset_index(drop=True)
    save_csv(
        df,
        DATA_DIR / "03_HAP_only_11变量_vs_去MV插管9变量_配对比较.csv",
    )
    return df


# ============================================================
# 12. 主程序
# ============================================================
def main():
    print("\n" + "=" * 88)
    print("HAP-only sensitivity analysis: new 11-predictor primary specification")
    print("=" * 88)
    print(f"项目目录：{PROJECT_DIR}")
    print(f"输出目录：{DATA_DIR}")

    full_df, hap_df, subtype_csv, split_csv = prepare_data()

    print("\n开始分析A：HAP-only + 新版主分析11变量")
    res11 = run_feature_set(
        hap=hap_df,
        features=FEATURES_11,
        output_dir=OUT_11,
        checkpoint_dir=CKPT_11,
        label="HAP-only sensitivity: new primary 11 predictors",
    )

    print("\n开始分析B：HAP-only + 去除MV和插管后的9变量")
    res9 = run_feature_set(
        hap=hap_df,
        features=FEATURES_9_NO_AIRWAY,
        output_dir=OUT_9,
        checkpoint_dir=CKPT_9,
        label="HAP-only sensitivity: 9 predictors without MV/intubation",
    )

    comparison = compare_11_vs_9(res11, res9)

    best11 = pd.DataFrame(list(res11["results"].values())).sort_values(
        "AUC", ascending=False
    ).iloc[0]
    best9 = pd.DataFrame(list(res9["results"].values())).sort_values(
        "AUC", ascending=False
    ).iloc[0]

    summary = pd.DataFrame([
        {
            "Analysis": "HAP-only, new primary 11 predictors",
            "Best_model_by_test_AUC": best11["Model"],
            "AUC": best11["AUC"],
            "AP": best11["AP"],
            "Brier": best11["Brier"],
            "Calibration_intercept": best11["Calibration_intercept"],
            "Calibration_slope": best11["Calibration_slope"],
        },
        {
            "Analysis": "HAP-only, 9 predictors without MV/intubation",
            "Best_model_by_test_AUC": best9["Model"],
            "AUC": best9["AUC"],
            "AP": best9["AP"],
            "Brier": best9["Brier"],
            "Calibration_intercept": best9["Calibration_intercept"],
            "Calibration_slope": best9["Calibration_slope"],
        },
    ])
    save_csv(summary, DATA_DIR / "04_HAP_only敏感性分析_核心结果汇总_11vs9.csv")

    readme = f"""
HAP-only sensitivity analysis -- new 11-predictor primary specification

HAP/VAP subtype file:
{subtype_csv}

Locked primary Train/Test file:
{split_csv}

Re-audited subtype counts:
- No pulmonary infection: 2011
- Non-VAP HAP: 1023
- VAP: 334
- Total: 3368

Locked primary split before VAP exclusion:
- Train: 2357, infection events: 950
- Test: 1011, infection events: 407

HAP-only split after excluding all 334 VAP cases:
- Train: 2104, HAP events: 697
- Test: 930, HAP events: 326

Analysis A -- 11 predictors:
{FEATURES_11}

Analysis B -- 9 predictors after removing MV/intubation:
{FEATURES_9_NO_AIRWAY}

Methods:
- Same eight algorithms and Bayesian tuning spaces as the revised primary analysis.
- 10-fold stratified Bayesian hyperparameter tuning using ROC AUC.
- Train-only MinMax scaling for continuous variables within the HAP-only analysis.
- 10-fold Platt calibration within the HAP-only training set.
- Test AUC, AP, Brier, calibration intercept/slope, ECE and MCE with 1000 patient-level bootstrap percentile 95% CIs.
- 11 vs 9 predictor versions compared on the same 930 HAP-only test patients using paired DeLong tests with Holm adjustment across eight algorithms.
- Paired bootstrap 95% CIs for Delta AUC, Delta AP and Delta Brier are defined as 9-predictor(no-airway) minus 11-predictor.
"""
    (DATA_DIR / "README_HAP_only敏感性分析_11vs9.txt").write_text(
        readme.strip() + "\n", encoding="utf-8"
    )

    print("\n" + "=" * 88)
    print("全部完成")
    print("=" * 88)
    cols = [
        "Model",
        "AUC_11pred",
        "AUC_9pred_no_airway",
        "Delta_AUC_9minus11",
        "Delta_AUC_95CI_low",
        "Delta_AUC_95CI_high",
        "DeLong_P",
        "DeLong_P_Holm",
        "AP_11pred",
        "AP_9pred_no_airway",
        "Brier_11pred",
        "Brier_9pred_no_airway",
    ]
    print(comparison[cols].to_string(index=False))

    print(f"\n所有结果：{DATA_DIR}")
    print("\n请把这三个文件发给我：")
    print(r"1) 01_HAP_only_11predictors\模型性能_HAP_only.csv")
    print(r"2) 02_HAP_only_9predictors_no_MV_no_intubation\模型性能_HAP_only.csv")
    print("3) 03_HAP_only_11变量_vs_去MV插管9变量_配对比较.csv")


if __name__ == "__main__":
    main()
