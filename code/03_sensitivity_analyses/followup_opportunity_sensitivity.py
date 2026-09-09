# -*- coding: utf-8 -*-
r"""
第三次修稿：不等随访机会 / ICU退出竞争事件敏感性分析
当前主分析11变量 · 固定原始Test集 · 直接使用主分析校准后预测概率
======================================================================

目的
----
回应审稿人关于48小时landmark之后观察机会不等、ICU退出/院内死亡结束观察机会、
以及缺少精确事件时间而无法进行有效Cox/Fine-Gray分析的问题。

本脚本不重新训练模型，而是直接读取“当前11变量主分析”已经锁定的1011例Test集
患者级校准后预测概率，因此分层分析与正文主分析的GBDT/LR结果完全同源，避免
重新调参或重新拟合导致轻微数值漂移。

分析内容
--------
A. 全3368例描述：
   Remaining_ICU_days = ICU_LOS - 2
   按结局报告median/IQR、mean/SD、range、院内死亡。

B. 预定义剩余ICU观察机会分层：
   0-1 days
   2-3 days
   >=4 days
   报告各层N、events、observed event proportion。

C. 在固定原始Test集(n=1011, events=407)中，对当前11变量主分析的：
   - GBDT
   - Logistic Regression
   直接使用主分析校准后概率，报告：
   AUC、AP、Brier、calibration intercept、calibration slope及1000次bootstrap 95%CI。
   同时报告限制分析：Remaining_ICU_days >=2 和 >=4。

重要解释
--------
- 这不是Cox/Fine-Gray分析。
- analysis-ready数据没有可靠的感染发生时间、精确ICU退出时间和死亡时间，不能
  构造有效的time-to-event / competing-risk数据结构。
- ICU_LOS是最终总ICU住院时长，可能被肺部感染本身延长；因此按Remaining ICU LOS
  分层是post hoc descriptive sensitivity analysis，不是对随访时间的因果调整。
- 本分析的estimand仍应表述为：48小时landmark后、ICU退出前“被记录/判定为HAP/VAP”
  的概率，而不是固定7天/14天发病风险。

Public repository usage
-----------------------
Patient-level inputs are not included. Set AIS_ICU_DATA_DIR (or legacy
REVISION_BASE_DIR) to a local restricted-data directory. Outputs are written under
AIS_ICU_OUTPUT_DIR (default: <repo>/outputs).
"""

import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import logit, expit
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss


# ============================================================
# 1. 参数
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
BASE_DIR = Path(
    os.environ.get(
        "AIS_ICU_DATA_DIR",
        os.environ.get("REVISION_BASE_DIR", str(REPO_ROOT / "restricted_data")),
    )
).expanduser().resolve()
OUTPUT_ROOT = Path(
    os.environ.get("AIS_ICU_OUTPUT_DIR", str(REPO_ROOT / "outputs"))
).expanduser().resolve()
OUTPUT_DIR = OUTPUT_ROOT / "followup_opportunity_sensitivity"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BOOTSTRAP_N = int(os.environ.get("FOLLOWUP_BOOTSTRAP_N", "1000"))
BOOTSTRAP_SEED = 20260903

OUTCOME = "Pulmonary_infection"
LOS_COL = "ICU_LOS"
DEATH_COL = "In_hospital_death"
ID_COL = "Study_row_id"
SPLIT_COL = "Primary_split"

EXPECTED = {
    "total_n": 3368,
    "total_events": 1357,
    "train_n": 2357,
    "train_events": 950,
    "test_n": 1011,
    "test_events": 407,
}

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
    "CO2",
]

PRED_COLS = {
    "GBDT": "Gradient Boosting Decision Tree Prob",
    "LR": "Logistic Regression Prob",
}
MODEL_FULL_NAME = {
    "GBDT": "Gradient Boosting Decision Tree",
    "LR": "Logistic Regression",
}


# ============================================================
# 2. 文件寻找与基础工具
# ============================================================
def clean_columns(df):
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )
    return df


def read_csv(path, nrows=None):
    return clean_columns(pd.read_csv(path, encoding="utf-8-sig", nrows=nrows))


def save_csv(df, filename):
    path = OUTPUT_DIR / filename
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  已保存: {path}")
    return path


def find_exact_or_recursive(filename):
    direct = BASE_DIR / filename
    if direct.exists():
        return direct
    hits = list(BASE_DIR.rglob(filename))
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        hits = sorted(hits, key=lambda p: p.stat().st_mtime, reverse=True)
        print(f"  注意：发现多个 {filename}，使用最新修改者：{hits[0]}")
        return hits[0]
    raise FileNotFoundError(f"在 {BASE_DIR} 下未找到：{filename}")


def normalize_id(s, name=ID_COL):
    x = pd.to_numeric(s, errors="coerce")
    if x.isna().any():
        bad = s[x.isna()].astype(str).value_counts().head(10).to_dict()
        raise ValueError(f"{name}存在无法识别值: {bad}")
    if not np.allclose(x.values, np.round(x.values)):
        raise ValueError(f"{name}存在非整数ID。")
    return x.astype(int)


def normalize_binary(s, name):
    if pd.api.types.is_numeric_dtype(s):
        x = pd.to_numeric(s, errors="coerce")
    else:
        mp = {
            "0": 0, "0.0": 0, "no": 0, "No": 0, "NO": 0,
            "1": 1, "1.0": 1, "yes": 1, "Yes": 1, "YES": 1,
        }
        x = s.astype(str).str.strip().map(mp)
    if x.isna().any():
        bad = s[x.isna()].astype(str).value_counts().head(10).to_dict()
        raise ValueError(f"{name}存在无法识别值: {bad}")
    x = x.astype(int)
    if not set(x.unique()).issubset({0, 1}):
        raise ValueError(f"{name}必须为0/1。")
    return x


def find_audit_csv():
    preferred_names = [
        "英文列名准备变量筛选_HAPVAP复核_待填写.csv",
        "英文列名准备变量筛选_HAPVAP复核_待填写(1).csv",
        "英文列名准备变量筛选(4).csv",
    ]
    required = {ID_COL, OUTCOME, LOS_COL, DEATH_COL}

    candidates = []
    for name in preferred_names:
        p = BASE_DIR / name
        if p.exists():
            candidates.append(p)
    candidates += [p for p in BASE_DIR.rglob("*.csv") if p not in candidates]

    valid = []
    for p in candidates:
        try:
            d = read_csv(p, nrows=5)
            if not required.issubset(d.columns):
                continue
            full = read_csv(p)
            if len(full) != EXPECTED["total_n"]:
                continue
            full[ID_COL] = normalize_id(full[ID_COL])
            if full[ID_COL].duplicated().any():
                continue
            yy = normalize_binary(full[OUTCOME], OUTCOME)
            if int(yy.sum()) != EXPECTED["total_events"]:
                continue
            valid.append(p)
        except Exception:
            continue

    if not valid:
        raise FileNotFoundError(
            "未找到同时含Study_row_id、ICU_LOS、In_hospital_death、"
            "Pulmonary_infection且n=3368的审计/源数据CSV。"
        )

    # 优先HAP/VAP复核文件，再按修改时间
    def score(p):
        name = p.name
        priority = 0
        if "HAPVAP" in name or "复核" in name:
            priority += 10
        if "待填写" in name:
            priority += 2
        return (priority, p.stat().st_mtime)

    valid.sort(key=score, reverse=True)
    if len(valid) > 1:
        print(f"  审计数据候选共{len(valid)}个，使用：{valid[0]}")
    return valid[0]


# ============================================================
# 3. 数据装配与严格核对
# ============================================================
def load_and_validate():
    split_path = find_exact_or_recursive("BSAfree_feature_selection_input_exactsplit.csv")
    pred_path = find_exact_or_recursive("测试集逐患者预测概率_校准后_带StudyRowID.csv")
    audit_path = find_audit_csv()

    print("\n输入文件：")
    print(f"  固定主分析身份/11变量: {split_path}")
    print(f"  主分析Test预测概率   : {pred_path}")
    print(f"  ICU LOS/死亡审计数据 : {audit_path}")

    split_df = read_csv(split_path)
    pred_df = read_csv(pred_path)
    audit_df = read_csv(audit_path)

    req_split = {ID_COL, SPLIT_COL, OUTCOME, *PRIMARY11}
    miss = sorted(req_split - set(split_df.columns))
    if miss:
        raise ValueError(f"固定主分析文件缺列: {miss}")

    req_pred = {ID_COL, "True Label", *PRED_COLS.values()}
    miss = sorted(req_pred - set(pred_df.columns))
    if miss:
        raise ValueError(f"主分析预测概率文件缺列: {miss}")

    req_audit = {ID_COL, OUTCOME, LOS_COL, DEATH_COL}
    miss = sorted(req_audit - set(audit_df.columns))
    if miss:
        raise ValueError(f"审计数据缺列: {miss}")

    if len(split_df) != EXPECTED["total_n"]:
        raise ValueError(f"固定主分析文件n应为3368，实际={len(split_df)}")
    if len(audit_df) != EXPECTED["total_n"]:
        raise ValueError(f"审计数据n应为3368，实际={len(audit_df)}")
    if len(pred_df) != EXPECTED["test_n"]:
        raise ValueError(f"Test预测概率n应为1011，实际={len(pred_df)}")

    for d in (split_df, pred_df, audit_df):
        d[ID_COL] = normalize_id(d[ID_COL])
        if d[ID_COL].duplicated().any():
            raise ValueError("发现重复Study_row_id。")

    split_df[OUTCOME] = normalize_binary(split_df[OUTCOME], OUTCOME)
    audit_df[OUTCOME] = normalize_binary(audit_df[OUTCOME], OUTCOME)
    audit_df[DEATH_COL] = normalize_binary(audit_df[DEATH_COL], DEATH_COL)
    pred_df["True Label"] = normalize_binary(pred_df["True Label"], "True Label")

    split_df[SPLIT_COL] = split_df[SPLIT_COL].astype(str).str.strip()
    if set(split_df[SPLIT_COL].unique()) != {"Train", "Test"}:
        raise ValueError(f"Primary_split异常: {split_df[SPLIT_COL].value_counts().to_dict()}")

    sc = split_df.groupby(SPLIT_COL)[OUTCOME].agg(N="size", Events="sum")
    expected_sc = {
        "Train": (EXPECTED["train_n"], EXPECTED["train_events"]),
        "Test": (EXPECTED["test_n"], EXPECTED["test_events"]),
    }
    for group, (n_exp, e_exp) in expected_sc.items():
        n_obs = int(sc.loc[group, "N"])
        e_obs = int(sc.loc[group, "Events"])
        if (n_obs, e_obs) != (n_exp, e_exp):
            raise ValueError(
                f"{group}固定分组不匹配: observed=({n_obs},{e_obs}), "
                f"expected=({n_exp},{e_exp})"
            )

    merged = split_df[[ID_COL, SPLIT_COL, OUTCOME]].merge(
        audit_df[[ID_COL, OUTCOME, LOS_COL, DEATH_COL]],
        on=ID_COL,
        how="left",
        suffixes=("_primary", "_audit"),
        validate="one_to_one",
    )
    if merged[[f"{OUTCOME}_audit", LOS_COL, DEATH_COL]].isna().any().any():
        raise ValueError("固定主分析数据与审计数据合并后出现缺失。")
    if not (merged[f"{OUTCOME}_primary"] == merged[f"{OUTCOME}_audit"]).all():
        raise ValueError("固定主分析与审计数据的Pulmonary_infection不一致。")

    merged = merged.rename(columns={f"{OUTCOME}_primary": OUTCOME})
    merged = merged.drop(columns=[f"{OUTCOME}_audit"])
    merged[LOS_COL] = pd.to_numeric(merged[LOS_COL], errors="coerce")
    if merged[LOS_COL].isna().any():
        raise ValueError("ICU_LOS存在缺失或非数值。")

    merged["Remaining_ICU_days"] = merged[LOS_COL].astype(float) - 2.0
    if (merged["Remaining_ICU_days"] < -1e-12).any():
        bad = merged.loc[merged["Remaining_ICU_days"] < 0, [ID_COL, LOS_COL]].head()
        raise ValueError("发现ICU_LOS<2，与48小时landmark纳入条件矛盾：\n" + bad.to_string(index=False))

    merged["Followup_stratum"] = pd.cut(
        merged["Remaining_ICU_days"],
        bins=[-0.001, 1, 3, np.inf],
        labels=["0-1 days", "2-3 days", ">=4 days"],
        include_lowest=True,
    )

    test_meta = merged.loc[merged[SPLIT_COL] == "Test"].copy()
    test = test_meta.merge(pred_df, on=ID_COL, how="inner", validate="one_to_one")
    if len(test) != EXPECTED["test_n"]:
        raise ValueError(f"Test元数据与预测概率合并后n={len(test)}，应为1011。")
    if not (test[OUTCOME].values == test["True Label"].values).all():
        raise ValueError("主分析预测文件True Label与固定Test结局不一致。")

    for col in PRED_COLS.values():
        test[col] = pd.to_numeric(test[col], errors="coerce")
        if test[col].isna().any() or ((test[col] < 0) | (test[col] > 1)).any():
            raise ValueError(f"{col}存在缺失或非[0,1]概率。")

    print(
        f"\n✅ 固定主分析核对通过：Train={EXPECTED['train_n']} "
        f"(events={EXPECTED['train_events']}) | Test={EXPECTED['test_n']} "
        f"(events={EXPECTED['test_events']})"
    )
    print(f"✅ Test患者Study_row_id与主分析概率一一匹配：n={len(test)}")

    return merged, test, split_path, pred_path, audit_path


# ============================================================
# 4. 描述性随访机会分析
# ============================================================
def fmt_median_iqr(x):
    x = pd.Series(x).dropna().astype(float)
    return f"{x.median():.1f} ({x.quantile(.25):.1f}-{x.quantile(.75):.1f})"


def descriptive_followup(full_df):
    rows = []
    for value, label in [
        (None, "Overall"),
        (0, "No pulmonary infection"),
        (1, "Pulmonary infection"),
    ]:
        d = full_df if value is None else full_df[full_df[OUTCOME] == value]
        x = d["Remaining_ICU_days"].astype(float)
        death_n = int(d[DEATH_COL].sum())
        rows.append({
            "Group": label,
            "N": int(len(d)),
            "Remaining_ICU_days_median_IQR": fmt_median_iqr(x),
            "Remaining_ICU_days_median": float(x.median()),
            "Remaining_ICU_days_Q1": float(x.quantile(.25)),
            "Remaining_ICU_days_Q3": float(x.quantile(.75)),
            "Remaining_ICU_days_mean": float(x.mean()),
            "Remaining_ICU_days_SD": float(x.std(ddof=1)),
            "Remaining_ICU_days_min": float(x.min()),
            "Remaining_ICU_days_max": float(x.max()),
            "In_hospital_death_n": death_n,
            "In_hospital_death_percent": death_n / len(d) * 100,
        })
    desc = pd.DataFrame(rows)
    save_csv(desc, "00_剩余ICU观察时间与院内死亡_按结局描述.csv")

    no = full_df.loc[full_df[OUTCOME] == 0, "Remaining_ICU_days"]
    yes = full_df.loc[full_df[OUTCOME] == 1, "Remaining_ICU_days"]
    u = stats.mannwhitneyu(no, yes, alternative="two-sided")
    death_tab = pd.crosstab(full_df[OUTCOME], full_df[DEATH_COL])
    chi2, p_chi, _, _ = stats.chi2_contingency(death_tab)
    tests = pd.DataFrame([
        {
            "Comparison": "Remaining ICU days: infection vs no infection",
            "Test": "Mann-Whitney U",
            "Statistic": float(u.statistic),
            "P_value": float(u.pvalue),
        },
        {
            "Comparison": "In-hospital death: infection vs no infection",
            "Test": "Chi-square",
            "Statistic": float(chi2),
            "P_value": float(p_chi),
        },
    ])
    save_csv(tests, "00_剩余ICU观察时间_组间检验.csv")

    strata = (
        full_df.groupby("Followup_stratum", observed=True)[OUTCOME]
        .agg(N="size", Events="sum", Event_rate="mean")
        .reset_index()
    )
    strata["Event_rate_percent"] = strata["Event_rate"] * 100
    strata = strata.drop(columns=["Event_rate"])
    save_csv(strata, "01_剩余ICU观察时间分层_总体结局率.csv")

    return desc, tests, strata


# ============================================================
# 5. 模型指标
# ============================================================
def calibration_intercept_slope(y_true, y_prob, eps=1e-6):
    """
    Fast Newton-Raphson implementation.

    Returns the same two quantities used in the manuscript workflow:
    - calibration intercept / calibration-in-the-large with slope fixed at 1
    - calibration slope from logistic(y ~ intercept + slope*logit(p))

    A custom 1D/2D Newton solver is used here because this function is called
    thousands of times during bootstrap and is much faster than constructing a
    sklearn model for every resample.
    """
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(y_prob, dtype=float), eps, 1 - eps)
    x = logit(p)

    # --- calibration-in-the-large: eta = x + a ---
    a = 0.0
    for _ in range(50):
        mu = expit(x + a)
        g = float(np.sum(y - mu))
        h = float(np.sum(mu * (1.0 - mu)))
        if h <= 1e-12:
            break
        step = g / h
        a += step
        if abs(step) < 1e-10:
            break
    intercept = float(a)

    # --- calibration slope: eta = a2 + b*x ---
    beta = np.array([0.0, 1.0], dtype=float)
    for _ in range(50):
        eta = beta[0] + beta[1] * x
        mu = expit(eta)
        w = mu * (1.0 - mu)
        g0 = float(np.sum(y - mu))
        g1 = float(np.sum((y - mu) * x))
        h00 = float(np.sum(w))
        h01 = float(np.sum(w * x))
        h11 = float(np.sum(w * x * x))
        H = np.array([[h00, h01], [h01, h11]], dtype=float)
        g = np.array([g0, g1], dtype=float)
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(H) @ g
        beta += step
        if float(np.max(np.abs(step))) < 1e-10:
            break
    slope = float(beta[1])
    return intercept, slope

def calculate_metrics(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    if len(y) < 10 or np.unique(y).size < 2:
        return {
            "AUC": np.nan,
            "AP": np.nan,
            "Brier": np.nan,
            "Calibration_intercept": np.nan,
            "Calibration_slope": np.nan,
        }
    ci, cs = calibration_intercept_slope(y, p)
    return {
        "AUC": float(roc_auc_score(y, p)),
        "AP": float(average_precision_score(y, p)),
        "Brier": float(brier_score_loss(y, p)),
        "Calibration_intercept": ci,
        "Calibration_slope": cs,
    }


def bootstrap_ci(y, p, n_boot=BOOTSTRAP_N, seed=BOOTSTRAP_SEED):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    rng = np.random.default_rng(seed)
    store = {k: [] for k in [
        "AUC", "AP", "Brier", "Calibration_intercept", "Calibration_slope"
    ]}
    n = len(y)

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb, pb = y[idx], p[idx]
        if np.unique(yb).size < 2:
            continue
        try:
            mm = calculate_metrics(yb, pb)
            for k in store:
                if np.isfinite(mm[k]):
                    store[k].append(mm[k])
        except Exception:
            continue

    out = {}
    for k, vals in store.items():
        arr = np.asarray(vals, dtype=float)
        if len(arr) == 0:
            out[k] = (np.nan, np.nan)
        else:
            out[k] = (
                float(np.percentile(arr, 2.5)),
                float(np.percentile(arr, 97.5)),
            )
    return out


def evaluate_test_strata(test_df):
    # 先保存Test层级计数
    test_strata = (
        test_df.groupby("Followup_stratum", observed=True)[OUTCOME]
        .agg(N="size", Events="sum", Event_rate="mean")
        .reset_index()
    )
    test_strata["Event_rate_percent"] = test_strata["Event_rate"] * 100
    test_strata = test_strata.drop(columns="Event_rate")
    save_csv(test_strata, "02_固定主分析Test_剩余ICU观察时间分层计数.csv")

    keep = [
        ID_COL, OUTCOME, LOS_COL, "Remaining_ICU_days", "Followup_stratum", DEATH_COL,
        PRED_COLS["GBDT"], PRED_COLS["LR"],
    ]
    save_csv(
        test_df[keep].sort_values(ID_COL),
        "03_固定Test逐患者_随访分层与主分析11变量概率.csv"
    )

    subsets = [
        ("All test patients", np.ones(len(test_df), dtype=bool)),
        ("Remaining ICU 0-1 days", test_df["Followup_stratum"].astype(str).eq("0-1 days").values),
        ("Remaining ICU 2-3 days", test_df["Followup_stratum"].astype(str).eq("2-3 days").values),
        ("Remaining ICU >=4 days", test_df["Followup_stratum"].astype(str).eq(">=4 days").values),
        ("Restriction: Remaining ICU >=2 days", (test_df["Remaining_ICU_days"] >= 2).values),
        ("Restriction: Remaining ICU >=4 days", (test_df["Remaining_ICU_days"] >= 4).values),
    ]

    rows = []
    for subset_idx, (subset_name, mask) in enumerate(subsets):
        yy = test_df.loc[mask, OUTCOME].astype(int).values
        for model_idx, (key, prob_col) in enumerate(PRED_COLS.items()):
            pp = test_df.loc[mask, prob_col].astype(float).values
            mm = calculate_metrics(yy, pp)
            ci = bootstrap_ci(
                yy, pp,
                n_boot=BOOTSTRAP_N,
                seed=BOOTSTRAP_SEED + subset_idx * 100 + model_idx,
            )
            row = {
                "Subset": subset_name,
                "Model": MODEL_FULL_NAME[key],
                "N": int(len(yy)),
                "Events": int(yy.sum()),
                "Event_rate_percent": float(yy.mean() * 100),
                **mm,
            }
            for metric, (lo, hi) in ci.items():
                row[f"{metric}_95CI_low"] = lo
                row[f"{metric}_95CI_high"] = hi
            rows.append(row)
            print(
                f"  {subset_name:38s} | {key:4s} | n={len(yy):4d} e={int(yy.sum()):3d} | "
                f"AUC={mm['AUC']:.4f} AP={mm['AP']:.4f} Brier={mm['Brier']:.4f} | "
                f"Int={mm['Calibration_intercept']:+.3f} Slope={mm['Calibration_slope']:.3f}"
            )

    result = pd.DataFrame(rows)
    save_csv(result, "04_按剩余ICU观察时间分层与限制分析_主分析11变量_GBDT_LR.csv")

    # 审稿回复核心结果（简化列）
    core = result[[
        "Subset", "Model", "N", "Events", "Event_rate_percent",
        "AUC", "AUC_95CI_low", "AUC_95CI_high",
        "AP", "AP_95CI_low", "AP_95CI_high",
        "Brier", "Brier_95CI_low", "Brier_95CI_high",
        "Calibration_intercept", "Calibration_intercept_95CI_low", "Calibration_intercept_95CI_high",
        "Calibration_slope", "Calibration_slope_95CI_low", "Calibration_slope_95CI_high",
    ]].copy()
    save_csv(core, "05_审稿回复核心表_剩余ICU观察机会_主分析11变量.csv")
    return test_strata, result, core


# ============================================================
# 6. README
# ============================================================
def write_readme(split_path, pred_path, audit_path):
    txt = f"""
Unequal post-landmark observation / ICU-exit sensitivity analysis
Current primary 11-predictor analysis

Input files
-----------
Fixed primary split / predictor inventory:
{split_path}

Locked patient-level calibrated probabilities from the current Primary11 model:
{pred_path}

ICU LOS / in-hospital death source used for descriptive follow-up opportunity:
{audit_path}

Design
------
1. No model was re-tuned or re-fitted in this script.
2. The original locked Test partition (n=1011; events=407) was identified by Study_row_id.
3. GBDT and LR probabilities were taken directly from the current Primary11 calibrated-prediction file.
4. Remaining_ICU_days = ICU_LOS - 2.
5. Prespecified descriptive strata: 0-1, 2-3, and >=4 remaining ICU days.
6. Restriction analyses: >=2 and >=4 remaining ICU days.
7. AUC, AP, Brier score, calibration intercept and calibration slope are reported with
   1000 patient-level bootstrap percentile 95% intervals.

Interpretation limits
---------------------
- Exact infection-onset / first-suspicion times are unavailable.
- Exact ICU-exit and death times are unavailable; In_hospital_death is a binary descriptive field.
- Cox/Fine-Gray models are therefore not fitted.
- Total ICU_LOS may itself be influenced by pulmonary infection; stratification by remaining ICU LOS
  is descriptive and post hoc, not a causal adjustment for follow-up duration.
- The operational estimand is the probability of a later-documented HAP/VAP event after the 48-hour
  landmark and before ICU exit, not a fixed-horizon incidence probability.
"""
    path = OUTPUT_DIR / "README_随访时间敏感性分析_11变量.txt"
    path.write_text(txt.strip() + "\n", encoding="utf-8")
    print(f"  已保存: {path}")


# ============================================================
# 7. MAIN
# ============================================================
def main():
    print("=" * 88)
    print("Unequal follow-up opportunity sensitivity analysis — Primary11 fixed predictions")
    print("=" * 88)
    print(f"BASE_DIR  : {BASE_DIR}")
    print(f"OUTPUT_DIR: {OUTPUT_DIR}")

    full_df, test_df, split_path, pred_path, audit_path = load_and_validate()

    print("\n[1/2] 描述剩余ICU观察机会与院内死亡...")
    desc, tests, strata = descriptive_followup(full_df)
    print(desc.to_string(index=False))
    print("\n总体观察机会分层：")
    print(strata.to_string(index=False))

    print("\n[2/2] 固定Primary11 Test概率：GBDT/LR分层与限制分析...")
    test_strata, perf, core = evaluate_test_strata(test_df)

    write_readme(split_path, pred_path, audit_path)

    print("\n" + "=" * 88)
    print("完成")
    print("=" * 88)
    print(f"输出目录：{OUTPUT_DIR}")
    print("\n请把下面3个文件发给我：")
    print("1) 00_剩余ICU观察时间与院内死亡_按结局描述.csv")
    print("2) 01_剩余ICU观察时间分层_总体结局率.csv")
    print("3) 04_按剩余ICU观察时间分层与限制分析_主分析11变量_GBDT_LR.csv")


if __name__ == "__main__":
    main()
