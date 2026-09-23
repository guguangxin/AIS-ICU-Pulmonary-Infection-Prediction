# -*- coding: utf-8 -*-
"""
Reviewer 3 post hoc predictor-count sensitivity analysis
========================================================

Purpose
-------
Use the *saved R/glmnet LASSO output* from the current feature-selection analysis
as the sole source of predictor ordering, then fit cumulative logistic-regression
models with Top 1, Top 2, ..., Top 11 predictors on the original fixed Train/Test split.

This script does NOT rerun LASSO or Boruta.

Required local files
--------------------
1) E:\\新建文件夹\\第三次修稿\\BSAfree_feature_selection_input_exactsplit.csv
2) R feature-selection result files, normally under:
   E:\\新建文件夹\\第三次修稿\\主分析_36候选变量重筛\\
   - 肺部感染_主分析重筛_LASSO系数.csv
   - 06_主分析重筛_LASSO_Boruta_intersection.csv

Optional audit file
-------------------
Current locked Primary11 test probabilities:
   测试集逐患者预测概率_校准后_带StudyRowID.csv
If found, the script checks that the Top11 refit is close to the locked Primary11 LR AUC.

Analysis principles
-------------------
- Predictor order comes directly from the saved R output. The R script already sorted
  the non-zero lambda.1se coefficients by abs(coefficient), decreasing.
- No test outcome is used to define the order or choose a predictor count.
- Same fixed Train/Test identities as the current primary analysis:
  Train n=2357, events=950; Test n=1011, events=407.
- For each cumulative LR model:
  * continuous-variable MinMax scaling fit on Train only;
  * 10-fold Bayesian hyperparameter tuning using ROC AUC;
  * 10-fold Platt (sigmoid) calibration on Train;
  * evaluation on the same fixed Test set.
- The curve is descriptive/post hoc. It must NOT be used to select an "optimal" k.
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from scipy.special import logit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

from skopt import BayesSearchCV
from skopt.space import Real, Categorical

warnings.filterwarnings("ignore")


# =============================================================================
# 1. Paths and fixed study constants
# =============================================================================
PROJECT_DIR = Path(r"E:\新建文件夹\第三次修稿")
SELECTION_DIR = PROJECT_DIR / "主分析_36候选变量重筛"

DATA_FILE = PROJECT_DIR / "BSAfree_feature_selection_input_exactsplit.csv"
LASSO_FILE = SELECTION_DIR / "肺部感染_主分析重筛_LASSO系数.csv"
INTERSECTION_FILE = SELECTION_DIR / "06_主分析重筛_LASSO_Boruta_intersection.csv"

OUTPUT_DIR = PROJECT_DIR / "LASSO排序_Top1到Top11_LR_AUC曲线"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTCOME = "Pulmonary_infection"
ID_COL = "Study_row_id"
SPLIT_COL = "Primary_split"

EXPECTED_TOTAL_N = 3368
EXPECTED_TRAIN_N = 2357
EXPECTED_TRAIN_EVENTS = 950
EXPECTED_TEST_N = 1011
EXPECTED_TEST_EVENTS = 407
EXPECTED_PRIMARY11_LR_AUC = 0.8273101518

RANDOM_STATE = 42
BAYES_CV = 10
CALIBRATION_CV = 10
BAYES_N_ITER = 40
N_BOOT = 1000
BOOT_SEED = 20260921

# Current primary 11 predictors in the Python modeling dataset.
CURRENT11_RAW = [
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

# R feature-selection display names -> Python analysis-ready column names.
R_TO_RAW = {
    "NEU": "NEUT_abs",
    "Intubation": "Intubation_tracheotomy",
    "MV": "Mechanical_ventilation",
    "LDH": "LDH",
    "LYM": "LYMPH_abs",
    "BUN": "BUN",
    "CCI": "CCI",
    "FIB": "FIB",
    "Surgery": "Surgery",
    "Diuretics": "Diuretics",
    "TCO2": "CO2",
}

RAW_TO_DISPLAY = {v: k for k, v in R_TO_RAW.items()}

# Terms saved by model.matrix/glmnet may have dummy suffixes for factor variables.
TERM_TO_RVAR = {
    "Intubation1": "Intubation",
    "MV1": "MV",
    "Surgery1": "Surgery",
    "Diuretics1": "Diuretics",
    "NEU": "NEU",
    "LYM": "LYM",
    "LDH": "LDH",
    "BUN": "BUN",
    "CCI": "CCI",
    "FIB": "FIB",
    "TCO2": "TCO2",
}

LR_SEARCH_SPACE = {
    "C": Real(0.001, 100, prior="log-uniform"),
    "penalty": Categorical(["l2"]),
    "solver": Categorical(["lbfgs", "saga", "newton-cg"]),
}


# =============================================================================
# 2. Basic helpers
# =============================================================================
def read_csv(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = (
        df.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )
    return df


def save_csv(df, filename):
    path = OUTPUT_DIR / filename
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  已保存: {path}")
    return path


def normalize_binary_outcome(s):
    if pd.api.types.is_numeric_dtype(s):
        x = pd.to_numeric(s, errors="coerce")
    else:
        mp = {
            "0": 0, "0.0": 0, "No": 0, "no": 0, "NO": 0,
            "1": 1, "1.0": 1, "Yes": 1, "yes": 1, "YES": 1,
        }
        x = s.astype(str).str.strip().map(mp)

    if x.isna().any():
        bad = s[x.isna()].astype(str).value_counts().head(10).to_dict()
        raise ValueError(f"{OUTCOME}存在无法识别的取值: {bad}")

    x = x.astype(int)
    if not set(x.unique()).issubset({0, 1}):
        raise ValueError(f"{OUTCOME}必须为0/1。")
    return x


def find_required_file(preferred, filename):
    if preferred.exists():
        return preferred

    hits = list(PROJECT_DIR.rglob(filename))
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        hits.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        print(f"⚠️ 找到多个 {filename}，使用最近修改者：{hits[0]}")
        return hits[0]

    # Also allow the script and CSVs to be put into the same folder.
    local = Path(__file__).resolve().parent / filename
    if local.exists():
        return local

    raise FileNotFoundError(
        f"找不到文件：{filename}\n"
        f"首选路径：{preferred}\n"
        f"也已在 {PROJECT_DIR} 子目录和脚本所在目录搜索。"
    )


# =============================================================================
# 3. Read and lock the R-LASSO order
# =============================================================================
def load_r_lasso_order():
    lasso_path = find_required_file(LASSO_FILE, LASSO_FILE.name)
    inter_path = find_required_file(INTERSECTION_FILE, INTERSECTION_FILE.name)

    print("\nR特征筛选结果：")
    print(f"  LASSO系数: {lasso_path}")
    print(f"  LASSO∩Boruta交集: {inter_path}")

    coef = read_csv(lasso_path)
    inter = read_csv(inter_path)

    if not {"term", "coefficient"}.issubset(coef.columns):
        raise ValueError("LASSO系数CSV必须包含 term 和 coefficient 两列。")
    if "Variable" not in inter.columns:
        raise ValueError("交集CSV必须包含 Variable 列。")

    coef = coef[["term", "coefficient"]].copy()
    coef["coefficient"] = pd.to_numeric(coef["coefficient"], errors="coerce")
    if coef["coefficient"].isna().any():
        raise ValueError("LASSO coefficient 中存在非数值。")

    # The R code saved this table in descending abs(coefficient) order.
    # Verify rather than silently reordering.
    abs_beta = coef["coefficient"].abs().to_numpy()
    if np.any(abs_beta[:-1] < abs_beta[1:] - 1e-12):
        raise ValueError(
            "LASSO系数CSV当前行序并非按 |coefficient| 递减。"
            "请不要继续；先核对是否为本次主分析R输出。"
        )

    unknown_terms = sorted(set(coef["term"]) - set(TERM_TO_RVAR))
    if unknown_terms:
        raise ValueError(
            "LASSO系数CSV出现无法映射的term: " + ", ".join(unknown_terms)
        )

    coef["R_variable"] = coef["term"].map(TERM_TO_RVAR)
    coef["Raw_column"] = coef["R_variable"].map(R_TO_RAW)

    if len(coef) != 11 or coef["R_variable"].nunique() != 11:
        raise ValueError(
            f"预期lambda.1se非零LASSO变量为11个；当前行数={len(coef)}, "
            f"唯一变量数={coef['R_variable'].nunique()}。"
        )

    inter_vars = inter["Variable"].astype(str).str.strip().tolist()
    if set(inter_vars) != set(coef["R_variable"]):
        print("\nLASSO变量：", sorted(coef["R_variable"].tolist()))
        print("交集变量：", sorted(inter_vars))
        raise ValueError("LASSO非零变量与最终LASSO∩Boruta交集不完全一致。")

    if set(coef["Raw_column"]) != set(CURRENT11_RAW):
        raise ValueError("R-LASSO映射后的11变量与当前Primary11不一致。")

    coef.insert(0, "LASSO_order", np.arange(1, 12))
    coef["Abs_coefficient"] = coef["coefficient"].abs()

    save_csv(coef, "00_锁定R_LASSO_lambda1se排序.csv")

    print("\n✅ 已锁定R-LASSO顺序（直接使用原CSV行序）：")
    print(
        coef[["LASSO_order", "R_variable", "coefficient"]]
        .to_string(index=False)
    )

    return coef["Raw_column"].tolist(), coef


# =============================================================================
# 4. Calibration and metrics
# =============================================================================
def calibration_intercept_slope(y_true, y_prob, eps=1e-6):
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(y_prob, dtype=float), eps, 1 - eps)
    lp = logit(p)

    # Calibration slope: logistic(y ~ intercept + slope*logit(p))
    try:
        m = LogisticRegression(penalty=None, solver="lbfgs", max_iter=3000)
        m.fit(lp.reshape(-1, 1), y)
    except Exception:
        m = LogisticRegression(
            penalty="l2", C=1e12, solver="lbfgs", max_iter=3000
        )
        m.fit(lp.reshape(-1, 1), y)
    slope = float(m.coef_[0, 0])

    # Calibration intercept with slope fixed at 1.
    def neg_ll(b):
        eta = lp + b[0]
        log_p = -np.logaddexp(0, -eta)
        log_1mp = -np.logaddexp(0, eta)
        return -(y * log_p + (1 - y) * log_1mp).sum()

    res = minimize(neg_ll, x0=np.array([0.0]), method="L-BFGS-B")
    intercept = float(res.x[0])
    return intercept, slope


def calibration_error(y_true, y_prob, n_bins=10):
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=float)
    order = np.argsort(p)
    bins = np.array_split(order, n_bins)

    ece = 0.0
    mce = 0.0
    for idx in bins:
        if len(idx) == 0:
            continue
        mean_p = float(np.mean(p[idx]))
        mean_y = float(np.mean(y[idx]))
        err = abs(mean_p - mean_y)
        ece += len(idx) / len(y) * err
        mce = max(mce, err)
    return float(ece), float(mce)


def point_metrics(y, p):
    ci, cs = calibration_intercept_slope(y, p)
    ece, mce = calibration_error(y, p, n_bins=10)
    return {
        "AUC": float(roc_auc_score(y, p)),
        "AP": float(average_precision_score(y, p)),
        "Brier": float(brier_score_loss(y, p)),
        "Calibration_intercept": ci,
        "Calibration_slope": cs,
        "ECE": ece,
        "MCE": mce,
    }


def bootstrap_auc_ci(y, p, n_boot=N_BOOT, seed=BOOT_SEED):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    rng = np.random.default_rng(seed)
    vals = []
    n = len(y)

    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        if np.unique(yy).size < 2:
            continue
        vals.append(roc_auc_score(yy, p[idx]))

    arr = np.asarray(vals, dtype=float)
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def paired_bootstrap_delta_auc(y, p_k, p_11, n_boot=N_BOOT, seed=BOOT_SEED):
    """Delta AUC = AUC(Top-k) - AUC(Top11)."""
    y = np.asarray(y, dtype=int)
    p_k = np.asarray(p_k, dtype=float)
    p_11 = np.asarray(p_11, dtype=float)

    rng = np.random.default_rng(seed)
    vals = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        if np.unique(yy).size < 2:
            continue
        vals.append(
            roc_auc_score(yy, p_k[idx]) - roc_auc_score(yy, p_11[idx])
        )

    arr = np.asarray(vals, dtype=float)
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def calibrate_model(estimator, X, y):
    try:
        cal = CalibratedClassifierCV(
            estimator=estimator,
            method="sigmoid",
            cv=CALIBRATION_CV,
            n_jobs=1,
        )
    except TypeError:
        cal = CalibratedClassifierCV(
            base_estimator=estimator,
            method="sigmoid",
            cv=CALIBRATION_CV,
            n_jobs=1,
        )
    cal.fit(X, y)
    return cal


# =============================================================================
# 5. Prepare fixed Train/Test data
# =============================================================================
def load_modeling_data():
    data_path = find_required_file(DATA_FILE, DATA_FILE.name)
    print(f"\n主分析数据：{data_path}")

    df = read_csv(data_path)
    required = {ID_COL, SPLIT_COL, OUTCOME, *CURRENT11_RAW}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"主分析输入文件缺列: {missing}")

    if len(df) != EXPECTED_TOTAL_N:
        raise ValueError(f"总样本量应为{EXPECTED_TOTAL_N}，实际={len(df)}")
    if df[ID_COL].duplicated().any():
        raise ValueError("Study_row_id存在重复。")

    df[OUTCOME] = normalize_binary_outcome(df[OUTCOME])
    df[SPLIT_COL] = df[SPLIT_COL].astype(str).str.strip()

    if set(df[SPLIT_COL].unique()) != {"Train", "Test"}:
        raise ValueError(
            f"Primary_split异常: {df[SPLIT_COL].value_counts().to_dict()}"
        )

    for c in CURRENT11_RAW:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if df[CURRENT11_RAW].isna().any().any():
        miss = df[CURRENT11_RAW].isna().sum()
        miss = miss[miss > 0].to_dict()
        raise ValueError(
            "当前11变量在analysis-ready文件中仍有缺失，请停止核查: " + str(miss)
        )

    train = df[df[SPLIT_COL] == "Train"].copy().reset_index(drop=True)
    test = df[df[SPLIT_COL] == "Test"].copy().reset_index(drop=True)

    if len(train) != EXPECTED_TRAIN_N or int(train[OUTCOME].sum()) != EXPECTED_TRAIN_EVENTS:
        raise ValueError(
            f"Train应为n={EXPECTED_TRAIN_N}/events={EXPECTED_TRAIN_EVENTS}；"
            f"实际n={len(train)}/events={int(train[OUTCOME].sum())}。"
        )
    if len(test) != EXPECTED_TEST_N or int(test[OUTCOME].sum()) != EXPECTED_TEST_EVENTS:
        raise ValueError(
            f"Test应为n={EXPECTED_TEST_N}/events={EXPECTED_TEST_EVENTS}；"
            f"实际n={len(test)}/events={int(test[OUTCOME].sum())}。"
        )

    print(
        f"✅ 固定分组核对通过：Train={len(train)} (events={int(train[OUTCOME].sum())}) | "
        f"Test={len(test)} (events={int(test[OUTCOME].sum())})"
    )
    return df, train, test, data_path


# =============================================================================
# 6. Fit one cumulative LR
# =============================================================================
def fit_cumulative_lr(train, test, features, k):
    X_train_raw = train[features].copy().reset_index(drop=True)
    X_test_raw = test[features].copy().reset_index(drop=True)
    y_train = train[OUTCOME].astype(int).reset_index(drop=True)
    y_test = test[OUTCOME].astype(int).reset_index(drop=True)

    # Exactly follow the current primary script's feature-type rule.
    binary_features = []
    continuous_features = []
    for col in features:
        if train[col].nunique(dropna=True) <= 5 or train[col].dtype == object:
            binary_features.append(col)
        else:
            continuous_features.append(col)

    scaler = MinMaxScaler()
    X_train = X_train_raw.copy()
    X_test = X_test_raw.copy()
    if continuous_features:
        X_train.loc[:, continuous_features] = scaler.fit_transform(
            X_train_raw[continuous_features]
        )
        X_test.loc[:, continuous_features] = scaler.transform(
            X_test_raw[continuous_features]
        )

    base_lr = LogisticRegression(random_state=RANDOM_STATE, max_iter=3000)
    cv = StratifiedKFold(
        n_splits=BAYES_CV,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    search = BayesSearchCV(
        estimator=base_lr,
        search_spaces=LR_SEARCH_SPACE,
        n_iter=BAYES_N_ITER,
        cv=cv,
        scoring="roc_auc",
        random_state=RANDOM_STATE,
        n_jobs=2,
        verbose=0,
        refit=True,
    )
    search.fit(X_train.values, y_train.values)

    calibrated = calibrate_model(
        search.best_estimator_,
        X_train.values,
        y_train.values,
    )
    p_test = calibrated.predict_proba(X_test.values)[:, 1]

    metrics = point_metrics(y_test.values, p_test)
    auc_lo, auc_hi = bootstrap_auc_ci(
        y_test.values,
        p_test,
        n_boot=N_BOOT,
        seed=BOOT_SEED + k * 100,
    )

    out = {
        **metrics,
        "AUC_95CI_low": auc_lo,
        "AUC_95CI_high": auc_hi,
        "Development_CV_AUC": float(search.best_score_),
        "Best_C": float(search.best_params_["C"]),
        "Best_penalty": str(search.best_params_["penalty"]),
        "Best_solver": str(search.best_params_["solver"]),
        "Binary_features": ";".join(RAW_TO_DISPLAY.get(x, x) for x in binary_features),
        "Continuous_scaled_features": ";".join(
            RAW_TO_DISPLAY.get(x, x) for x in continuous_features
        ),
        "p_test": p_test,
    }
    return out


# =============================================================================
# 7. Optional audit against locked current Primary11 predictions
# =============================================================================
def find_locked_primary_prediction_file():
    exact_candidates = [
        PROJECT_DIR / "主分析11变量_八模型完整重跑_SHAP" / "测试集逐患者预测概率_校准后_带StudyRowID.csv",
        PROJECT_DIR / "主分析11变量_八模型完整重跑_SHAP" / "测试集预测概率_校准后.csv",
    ]
    for p in exact_candidates:
        if p.exists():
            return p

    hits = list(PROJECT_DIR.rglob("测试集逐患者预测概率_校准后_带StudyRowID.csv"))
    if hits:
        hits.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return hits[0]
    return None


def detect_lr_prob_col(df):
    preferred = [
        "Logistic Regression Prob",
        "Logistic Regression Probability",
        "Primary11_LR",
        "LR Prob",
    ]
    for c in preferred:
        if c in df.columns:
            return c

    candidates = [
        c for c in df.columns
        if "logistic" in c.lower() and ("prob" in c.lower() or "proba" in c.lower())
    ]
    return candidates[0] if candidates else None


def audit_locked_primary(test, p_top11):
    y = test[OUTCOME].astype(int).to_numpy()
    top11_auc = float(roc_auc_score(y, p_top11))

    rows = [{
        "Source": "Top11 cumulative refit in this post hoc analysis",
        "AUC": top11_auc,
        "Expected_locked_primary_AUC": EXPECTED_PRIMARY11_LR_AUC,
        "Difference_from_expected": top11_auc - EXPECTED_PRIMARY11_LR_AUC,
    }]

    pred_path = find_locked_primary_prediction_file()
    locked_auc = np.nan
    if pred_path is not None:
        try:
            d = read_csv(pred_path)
            col = detect_lr_prob_col(d)
            if col is not None:
                if ID_COL in d.columns:
                    m = test[[ID_COL, OUTCOME]].merge(
                        d[[ID_COL, col]],
                        on=ID_COL,
                        how="inner",
                        validate="one_to_one",
                    )
                    if len(m) == EXPECTED_TEST_N:
                        locked_auc = float(
                            roc_auc_score(
                                m[OUTCOME].astype(int),
                                pd.to_numeric(m[col], errors="raise"),
                            )
                        )
                elif len(d) == EXPECTED_TEST_N:
                    locked_auc = float(
                        roc_auc_score(y, pd.to_numeric(d[col], errors="raise"))
                    )
        except Exception as e:
            print(f"⚠️ 读取锁定Primary11预测概率失败，仅跳过该审计：{e}")

    if np.isfinite(locked_auc):
        rows.append({
            "Source": "Locked current Primary11 LR predictions",
            "AUC": locked_auc,
            "Expected_locked_primary_AUC": EXPECTED_PRIMARY11_LR_AUC,
            "Difference_from_expected": locked_auc - EXPECTED_PRIMARY11_LR_AUC,
        })

    audit = pd.DataFrame(rows)
    save_csv(audit, "04_Top11与锁定Primary11_LR_AUC核对.csv")

    print("\nTop11核对：")
    print(f"  本次Top11 refit AUC = {top11_auc:.10f}")
    print(f"  论文锁定Primary11 = {EXPECTED_PRIMARY11_LR_AUC:.10f}")
    if np.isfinite(locked_auc):
        print(f"  锁定预测文件AUC     = {locked_auc:.10f}")

    if abs(top11_auc - EXPECTED_PRIMARY11_LR_AUC) > 0.002:
        print(
            "\n⚠️ 警告：本次Top11 AUC与锁定Primary11相差>0.002。"
            "请先把结果发给我，不要直接写进论文。"
        )

    return locked_auc


# =============================================================================
# 8. Figure
# =============================================================================
def make_auc_curve(result_df, locked_auc=np.nan):
    x = result_df["N_predictors"].to_numpy(dtype=float)
    y = result_df["AUC"].to_numpy(dtype=float)
    lo = result_df["AUC_95CI_low"].to_numpy(dtype=float)
    hi = result_df["AUC_95CI_high"].to_numpy(dtype=float)

    yerr = np.vstack([y - lo, hi - y])

    fig, ax = plt.subplots(figsize=(10.5, 7.2))
    ax.errorbar(
        x, y, yerr=yerr,
        fmt="o-",
        linewidth=2,
        markersize=6,
        capsize=3,
        label="Cumulative LR",
    )

    if np.isfinite(locked_auc):
        ax.axhline(
            locked_auc,
            linestyle="--",
            linewidth=1.5,
            label=f"Locked Primary11 LR AUC={locked_auc:.3f}",
        )

    ax.set_xticks(np.arange(1, 12))
    ax.set_xlim(0.6, 11.4)
    ax.set_xlabel("Number of predictors entered in R-LASSO coefficient order")
    ax.set_ylabel("Fixed internal test AUC")
    ax.grid(alpha=0.25)

    for _, row in result_df.iterrows():
        k = int(row["N_predictors"])
        lab = str(row["Added_predictor"])
        offset = 11 if k % 2 else -19
        ax.annotate(
            lab,
            (k, row["AUC"]),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
            rotation=35,
        )

    ax.legend(loc="lower right")
    fig.tight_layout()

    png = OUTPUT_DIR / "Figure_LASSO_order_predictor_count_test_AUC.png"
    pdf = OUTPUT_DIR / "Figure_LASSO_order_predictor_count_test_AUC.pdf"
    fig.savefig(png, dpi=400, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"  已保存: {png}")
    print(f"  已保存: {pdf}")


# =============================================================================
# 9. Main
# =============================================================================
def main():
    print("=" * 96)
    print("Reviewer 3: R-LASSO order, cumulative Top1-to-Top11 LR test-AUC curve")
    print("=" * 96)

    lasso_order, lasso_table = load_r_lasso_order()
    _, train, test, data_path = load_modeling_data()

    print("\n开始按R-LASSO锁定顺序累计建模。")
    print("重要：测试集只用于描述性评价，不用来选择k。")

    pred_df = test[[ID_COL, OUTCOME]].copy().reset_index(drop=True)
    pred_df = pred_df.rename(columns={OUTCOME: "True_Label"})

    rows = []
    all_probs = {}

    for k in range(1, 12):
        features = lasso_order[:k]
        display_features = [RAW_TO_DISPLAY.get(x, x) for x in features]

        print("\n" + "-" * 96)
        print(f"Top{k}: " + " + ".join(display_features))

        fit = fit_cumulative_lr(train, test, features, k)
        p_test = fit.pop("p_test")
        all_probs[k] = p_test
        pred_df[f"LR_Top{k}_Prob"] = p_test

        row = {
            "N_predictors": k,
            "Added_predictor": display_features[-1],
            "Predictor_set": " + ".join(display_features),
            **fit,
            "Analysis_status": (
                "Post hoc descriptive cumulative curve; predictor count not selected from test performance"
            ),
        }
        rows.append(row)

        print(
            f"  Development CV AUC={fit['Development_CV_AUC']:.4f} | "
            f"Test AUC={fit['AUC']:.4f} "
            f"({fit['AUC_95CI_low']:.4f}-{fit['AUC_95CI_high']:.4f}) | "
            f"AP={fit['AP']:.4f} | Brier={fit['Brier']:.4f}"
        )

    results = pd.DataFrame(rows)
    y_test = test[OUTCOME].astype(int).to_numpy()
    p11 = all_probs[11]
    auc11 = float(roc_auc_score(y_test, p11))

    delta = []
    dlo = []
    dhi = []
    for k in range(1, 12):
        pk = all_probs[k]
        d = float(roc_auc_score(y_test, pk) - auc11)
        delta.append(d)
        if k == 11:
            dlo.append(0.0)
            dhi.append(0.0)
        else:
            lo, hi = paired_bootstrap_delta_auc(
                y_test,
                pk,
                p11,
                n_boot=N_BOOT,
                seed=BOOT_SEED + 5000 + k * 100,
            )
            dlo.append(lo)
            dhi.append(hi)

    results["Delta_AUC_TopK_minus_Top11"] = delta
    results["Delta_AUC_95CI_low"] = dlo
    results["Delta_AUC_95CI_high"] = dhi

    save_csv(results, "01_Top1到Top11_LR_固定Test性能.csv")
    save_csv(pred_df, "02_Top1到Top11_LR_固定Test逐患者概率.csv")

    locked_auc = audit_locked_primary(test, p11)
    make_auc_curve(results, locked_auc=locked_auc)

    core = results[[
        "N_predictors",
        "Added_predictor",
        "Predictor_set",
        "AUC",
        "AUC_95CI_low",
        "AUC_95CI_high",
        "Delta_AUC_TopK_minus_Top11",
        "Delta_AUC_95CI_low",
        "Delta_AUC_95CI_high",
        "AP",
        "Brier",
        "Calibration_intercept",
        "Calibration_slope",
        "ECE",
        "MCE",
    ]].copy()
    core["Note"] = (
        "Post hoc descriptive cumulative analysis using the order saved by the original R/glmnet "
        "lambda.1se output; no predictor count was selected using fixed-test performance."
    )
    save_csv(core, "03_Reviewer核心表_R_LASSO排序_predictor_count_curve.csv")

    design = {
        "Feature_order_source": str(find_required_file(LASSO_FILE, LASSO_FILE.name)),
        "Intersection_source": str(find_required_file(INTERSECTION_FILE, INTERSECTION_FILE.name)),
        "Modeling_data_source": str(data_path),
        "Order_definition": "Existing R output row order; original R script sorted nonzero lambda.1se coefficients by absolute coefficient magnitude, decreasing",
        "No_LASSO_refit_in_this_script": True,
        "No_Boruta_refit_in_this_script": True,
        "Fixed_original_split": True,
        "Train_N": EXPECTED_TRAIN_N,
        "Train_events": EXPECTED_TRAIN_EVENTS,
        "Test_N": EXPECTED_TEST_N,
        "Test_events": EXPECTED_TEST_EVENTS,
        "Bayesian_tuning_CV": BAYES_CV,
        "Bayesian_iterations_per_k": BAYES_N_ITER,
        "Platt_calibration_CV": CALIBRATION_CV,
        "Bootstrap_N": N_BOOT,
        "Interpretation": "Post hoc descriptive only; do not choose optimal predictor count from test curve",
    }
    with open(
        OUTPUT_DIR / "00_分析设计_R_LASSO排序_predictor_count.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(design, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 96)
    print("全部完成")
    print("=" * 96)
    print(f"输出目录：{OUTPUT_DIR}")
    print("\n请把下面3个文件发给我：")
    print("1) 01_Top1到Top11_LR_固定Test性能.csv")
    print("2) 03_Reviewer核心表_R_LASSO排序_predictor_count_curve.csv")
    print("3) Figure_LASSO_order_predictor_count_test_AUC.png")
    print("\n另外如04_Top11与锁定Primary11_LR_AUC核对.csv出现明显差异，也一并发给我。")


if __name__ == "__main__":
    main()
