# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 15:44:42 2026

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 21:59:01 2026

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 12:12:29 2026

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 16:09:52 2026

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 14:11:20 2026

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 2026

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
HISTORICAL EXPLORATORY ANALYSIS (BSA-inclusive)
肺部感染预测 - 机器学习分类分析（8模型 · 10特征 · Platt 校准完整版）
======================================================================
  · 数据路径 : <restricted local data directory>
  · 文件名   : 英文列名准备变量筛选.csv
  · 输出目录 : <local output directory>
  · 结局     : Pulmonary_infection
  · 预测变量（10个）:
      BSA(Broad_spectrum_antibiotics), NEU(NEUT_abs),
      Intubation(Intubation_tracheotomy), MV(Mechanical_ventilation),
      LDH, LYM(LYMPH_abs), BUN, CCI, FIB, Surgery

====== 校准方法学要点（与 ICH 高费用识别模型一致）======
  ▪ 贝叶斯超参数优化：10 折分层 CV
  ▪ Platt Scaling 概率校准：10 折 CV（与调参同口径，避免信息泄漏）
  ▪ Youden 最优阈值：沿用历史归档实现，基于训练集校准后预测概率确定（非严格 cross-fitted OOF）
  ▪ 参数冻结：超参数、Platt (A,B)、阈值 τ* 训练阶段一次性冻结
  ▪ 测试集：30% 分层抽样，全程无任何重新拟合
  ▪ 校准评估：Brier + 校准截距 + 校准斜率（理想 0 / 0 / 1）+ HL 检验
  ▪ 输出：Platt (A,B) 参数表、校准前后完整对比表、4 面板校准曲线图

内存安全特性：
  · BayesSearchCV n_jobs=2（默认），MLP 用 n_jobs=1
  · CalibratedClassifierCV 内部不并行，避免嵌套并行
  · 每训练完一个模型立即保存至 checkpoints/，崩溃后可续训
  · compress=3 压缩保存；gc.collect() 主动释放内存

⚠️ 如仍报 WinError 1455（页面文件不足）：
   · 关闭其他占内存程序（浏览器/Office 等）
   · 控制面板 → 系统 → 高级 → 性能设置 → 虚拟内存 → 自定义
     初始值=物理内存，最大值=物理内存 2-3倍 → 重启
   · 或把 GLOBAL_N_JOBS 改为 1 / 把 CALIBRATION_CV 改为 5
======================================================================
"""

import warnings
warnings.filterwarnings('ignore')
import argparse
import os, logging, gc, sys, re
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('xgboost').setLevel(logging.CRITICAL)
logging.getLogger('lightgbm').setLevel(logging.CRITICAL)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
import time, joblib
import sklearn
from scipy import stats
from scipy.stats import chi2
from scipy.special import logit
from scipy.optimize import minimize

from sklearn.model_selection import (train_test_split, learning_curve, StratifiedKFold)
from sklearn.preprocessing import MinMaxScaler
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (roc_auc_score, roc_curve,
                             precision_recall_curve, average_precision_score,
                             confusion_matrix, balanced_accuracy_score,
                             matthews_corrcoef, brier_score_loss)

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
# 配置
# ============================================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

# ======== 全局图表字体加粗放大设置（已整体放大，并取消图标题） ========
plt.rcParams['font.size']             = 18          # 全局默认字体大小
plt.rcParams['font.weight']           = 'bold'      # 全局默认字体加粗
plt.rcParams['axes.labelsize']        = 22          # x/y 轴标签
plt.rcParams['axes.labelweight']      = 'bold'
plt.rcParams['axes.titlesize']        = 0           # 不显示图标题
plt.rcParams['axes.titleweight']      = 'bold'
plt.rcParams['xtick.labelsize']       = 18          # x 轴刻度标签
plt.rcParams['ytick.labelsize']       = 18          # y 轴刻度标签
plt.rcParams['legend.fontsize']       = 16          # 图例字体
plt.rcParams['legend.title_fontsize'] = 17          # 图例标题字体
plt.rcParams['figure.titlesize']      = 0           # 不显示总标题
plt.rcParams['figure.titleweight']    = 'bold'
plt.rcParams['axes.linewidth']        = 1.8         # 坐标轴线加粗
plt.rcParams['xtick.major.width']     = 1.8
plt.rcParams['ytick.major.width']     = 1.8
plt.rcParams['xtick.major.size']      = 6
plt.rcParams['ytick.major.size']      = 6
print(f"📦 sklearn version: {sklearn.__version__}")

# ★★★ Public-repository path configuration ★★★
# Patient-level data remain local/restricted and are never bundled with the repository.
_parser = argparse.ArgumentParser(
    description="Historical exploratory BSA-inclusive 10-predictor / 8-model analysis."
)
_parser.add_argument(
    "--data-file",
    default=os.environ.get("AIS_ICU_BSA10_DATA_FILE"),
    help="Restricted analysis dataset CSV. Alternatively set AIS_ICU_BSA10_DATA_FILE.",
)
_parser.add_argument(
    "--output",
    default=os.environ.get("AIS_ICU_BSA10_OUTPUT", "outputs/historical_bsa_inclusive"),
    help="Local output directory (default: outputs/historical_bsa_inclusive).",
)
_parser.add_argument(
    "--export-patient-level",
    action="store_true",
    help=(
        "Explicitly export local patient-level prediction files needed for Table S11/S12 reconstruction. "
        "Do not commit these files to the public repository."
    ),
)
_args = _parser.parse_args()

if not _args.data_file:
    raise SystemExit(
        "A restricted input dataset is required. Use --data-file PATH or set AIS_ICU_BSA10_DATA_FILE."
    )

data_file   = os.path.abspath(os.path.expanduser(_args.data_file))
output_path = os.path.abspath(os.path.expanduser(_args.output))
EXPORT_PATIENT_LEVEL = bool(_args.export_patient_level)
ckpt_dir    = os.path.join(output_path, "checkpoints")
os.makedirs(output_path, exist_ok=True)
os.makedirs(ckpt_dir, exist_ok=True)

# ★★★ 结局 + 10个预测变量 ★★★
OUTCOME = 'Pulmonary_infection'
FEATURES_USED = [
    'Broad_spectrum_antibiotics',   # BSA
    'NEUT_abs',                     # NEU
    'Intubation_tracheotomy',       # Intubation
    'Mechanical_ventilation',       # MV
    'LDH',
    'LYMPH_abs',                    # LYM
    'BUN',
    'CCI',
    'FIB',
    'Surgery',
]

# 并行度控制
GLOBAL_N_JOBS = 2
MODEL_N_JOBS = {'MLP':1, 'RF':2}
COMPRESS_LEVEL = 3

# 概率校准配置
CALIBRATION_METHOD = 'sigmoid'   # Platt Scaling
CALIBRATION_CV     = 10          # 与超参数 CV 同口径

def save_fig(fn):
    base, _ = os.path.splitext(fn)
    out_fn = base + '.jpg'
    plt.savefig(os.path.join(output_path, out_fn), dpi=300,
                bbox_inches='tight', facecolor='white', format='jpg')
    print(f"  📊 已保存: {out_fn}")

def save_csv(df, fn):
    df.to_csv(os.path.join(output_path, fn), index=False, encoding='utf-8-sig')
    print(f"  📄 已保存: {fn}")

DISPLAY_NAME_MAP = {
    'Pulmonary_infection':        'Pulmonary Infection',
    'Broad_spectrum_antibiotics': 'Broad-spectrum Antibiotic Use (BSA)',
    'NEUT_abs':                   'Absolute Neutrophil Count (NEU)',
    'Intubation_tracheotomy':     'Intubation/Tracheotomy',
    'Mechanical_ventilation':     'Mechanical Ventilation (MV)',
    'LDH':                        'Lactate Dehydrogenase (LDH)',
    'LYMPH_abs':                  'Absolute Lymphocyte Count (LYM)',
    'BUN':                        'Blood Urea Nitrogen (BUN)',
    'CCI':                        'Charlson Comorbidity Index (CCI)',
    'FIB':                        'Fibrinogen (FIB)',
    'Surgery':                    'Surgery',
}

MODEL_FULL_NAME = {
    'RF':'Random Forest','GBDT':'Gradient Boosting Decision Tree',
    'LR':'Logistic Regression','NB':'Naive Bayes','DT':'Decision Tree',
    'LightGBM':'LightGBM','XGBoost':'XGBoost',
    'MLP':'Multilayer Perceptron',
}

MODEL_COLORS = {
    'RF':'#3498DB','GBDT':'#8E44AD','LR':'#1ABC9C','NB':'#F39C12',
    'DT':'#95A5A6','LightGBM':'#2ECC71','XGBoost':'#E74C3C',
    'MLP':'#2980B9',
}

# ============================================================
# ROC彩带图 / 圆盘放射图 — 专用配置（图14/15/16 使用）
# —— 以下常量与独立脚本「机器学习roc曲线与雷达图代码.py」保持完全一致 ——
# ============================================================
# 反向查表：full_name → key
FULL_TO_KEY = {v: k for k, v in MODEL_FULL_NAME.items()}

# 模型缩写（键为 MODEL_FULL_NAME 的 full_name，用于圆盘图中心显示）
MODEL_ABBR = {
    'LightGBM':'LGBM','XGBoost':'XGB',
    'Gradient Boosting Decision Tree':'GBDT','Multilayer Perceptron':'MLP',
    'Random Forest':'RF','Logistic Regression':'LR',
    'Naive Bayes':'NB','Decision Tree':'DT',
}

# 模型白名单：只在这里列出的 8 个模型会出现在图中
ALLOWED_MODELS = {
    'LightGBM', 'XGBoost',
    'Gradient Boosting Decision Tree',
    'Multilayer Perceptron',
    'Random Forest', 'Logistic Regression',
    'Naive Bayes', 'Decision Tree',
}

# ─── 圆盘图：指标配色 ───
METRIC_COLORS = {
    'Accuracy':  '#1f4e7a',
    'Precision': '#3d8ec4',
    'Recall':    '#a5c9df',
    'F1 Score':  '#f6cfc0',
    'AUC':       '#e38a4a',
}
METRIC_ORDER = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']
# CSV 用的是 Sensitivity / F1，参考图用的是 Recall / F1 Score
CSV_COL_MAP = {
    'Accuracy':'Accuracy', 'Precision':'Precision',
    'Recall':'Sensitivity', 'F1 Score':'F1', 'AUC':'AUC',
}

# 图14/15/16 的字号（已整体放大；图标题后续统一移除）
ROC_FS_AXIS_LABEL    = 22   # ROC x/y 轴标题
ROC_FS_TITLE         = 0    # 不显示 ROC 图标题
ROC_FS_LEGEND        = 16   # ROC 图例
ROC_FS_TICK          = 20   # ROC 刻度数字
ROC_FS_AUC_VALUE     = 20   # 右侧彩带上的 AUC 数字

DISK_FS_METRIC_LABEL = 11   # 外圈指标名适当缩小，避免相互重叠
DISK_FS_BAR_VALUE    = 13
DISK_FS_GRID_TICK    = 12
DISK_FS_MODEL_ABBR   = 20
DISK_FS_LEGEND       = 20

# PR / DCA 两张主曲线图：进一步放大并加粗，做成更接近论文成图的观感
PR_DCA_FS_LABEL  = 30
PR_DCA_FS_TICK   = 25
PR_DCA_FS_LEGEND = 19

# 双拼测试集校准曲线：坐标字号与其他曲线统一，但图例单独收紧
# 否则长图例会把画面挤坏，导致看起来发“方”、比例不协调
CAL2_FS_LABEL   = 26
CAL2_FS_TICK    = 22
CAL2_FS_LEGEND  = 10.5
CAL2_BOX_ASPECT = 0.72

# ============================================================
# 统计检验函数
# ============================================================
def _compute_midrank(x):
    J=np.argsort(x); Z=x[J]; N=len(x); T=np.zeros(N,dtype=np.float64)
    i=0
    while i<N:
        j=i
        while j<N and Z[j]==Z[i]: j+=1
        T[i:j]=0.5*(i+j-1); i=j
    T2=np.empty(N,dtype=np.float64); T2[J]=T+1; return T2

def _fast_delong(pst, m):
    n=pst.shape[1]-m; k=pst.shape[0]
    pos=pst[:,:m]; neg=pst[:,m:]
    tx=np.empty([k,m],dtype=np.float64); ty=np.empty([k,n],dtype=np.float64); tz=np.empty([k,m+n],dtype=np.float64)
    for r in range(k):
        tx[r]=_compute_midrank(pos[r]); ty[r]=_compute_midrank(neg[r]); tz[r]=_compute_midrank(pst[r])
    aucs=tz[:,:m].sum(axis=1)/m/n-float(m+1.0)/2.0/n
    v01=(tz[:,:m]-tx)/n; v10=1.0-(tz[:,m:]-ty)/m
    cov=np.cov(v01)/m+np.cov(v10)/n; return aucs,cov

def delong_test(y_true,p1,p2):
    y=np.array(y_true); p1=np.array(p1); p2=np.array(p2)
    order=np.argsort(y)[::-1]; pst=np.vstack([p1[order],p2[order]])
    aucs,cov=_fast_delong(pst,int(y.sum()))
    se=np.sqrt(max(cov[0,0]+cov[1,1]-2*cov[0,1],1e-20))
    z=(aucs[0]-aucs[1])/se; p=2*stats.norm.sf(abs(z))
    return float(aucs[0]),float(aucs[1]),float(z),float(p)

def hosmer_lemeshow_test(y_true,y_proba,n_bins=10):
    y=np.array(y_true); yp=np.array(y_proba)
    order=np.argsort(yp); ys=y[order]; yps=yp[order]
    bins=np.array_split(np.arange(len(y)),n_bins); hl=0.0; rows=[]
    for b in bins:
        if not len(b): continue
        op=ys[b].sum(); ep=yps[b].sum()
        on=len(b)-op; en=len(b)-ep
        if ep>0: hl+=(op-ep)**2/ep
        if en>0: hl+=(on-en)**2/en
        rows.append({'n':len(b),'obs_pos':int(op),'exp_pos':round(ep,2)})
    return float(hl),float(1-chi2.cdf(hl,n_bins-2)),pd.DataFrame(rows)

# ============================================================
# 概率校准相关函数（移植自 ICH 高费用识别模型）
# ============================================================
def calibration_intercept_slope(y_true, y_proba, eps=1e-6):
    """
    计算校准截距(calibration intercept)与校准斜率(calibration slope)
    理想值：intercept = 0, slope = 1

    - intercept: 将斜率固定为 1，以 y ~ offset(logit(p)) + β₀ 的 GLM
                  估计截距 β₀，反映"整体预测水平"的偏差
                  （>0: 系统性低估事件概率；<0: 高估）
    - slope:     以 y ~ β₀ + β₁·logit(p) 的 logistic regression 估计斜率 β₁
                  （<1: 过度自信；>1: 过度保守）
    """
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(y_proba, dtype=float), eps, 1-eps)
    lp = logit(p)

    # Calibration slope: 无约束逻辑回归
    try:
        lr = LogisticRegression(penalty=None, solver='lbfgs', max_iter=2000)
    except (TypeError, ValueError):
        # 老版 sklearn 用大 C 近似无正则化
        lr = LogisticRegression(penalty='l2', C=1e12, solver='lbfgs', max_iter=2000)
    lr.fit(lp.reshape(-1, 1), y)
    slope = float(lr.coef_[0, 0])

    # Calibration intercept: 斜率固定为 1，仅估计截距
    # 等价于：logit(P(y=1)) = logit(p) + β₀
    def neg_ll(b):
        eta = lp + b[0]
        # 数值稳定的 log-sigmoid
        log_p   = -np.logaddexp(0, -eta)
        log_1mp = -np.logaddexp(0, eta)
        return -(y*log_p + (1-y)*log_1mp).sum()
    res = minimize(neg_ll, x0=[0.0], method='L-BFGS-B')
    intercept = float(res.x[0])

    return {'intercept': intercept, 'slope': slope}


def extract_platt_params(calibrated_model):
    """
    从 CalibratedClassifierCV 提取各折 sigmoid 校准参数 (A, B)
    校准函数：p_calibrated = 1 / (1 + exp(A * f + B))
              其中 f 为基模型的 decision function / 正类概率输出

    兼容 sklearn 多版本：
      - sklearn < 1.3 : cc.calibrators_（带下划线后缀）
      - sklearn >= 1.3: cc.calibrators（不带下划线后缀）
    """
    As, Bs = [], []
    for cc in calibrated_model.calibrated_classifiers_:
        # 兼容新旧版本属性名
        calibrators = getattr(cc, 'calibrators', None)
        if calibrators is None:
            calibrators = getattr(cc, 'calibrators_', None)
        if calibrators is None:
            return {'A_mean': np.nan, 'A_std': np.nan,
                    'B_mean': np.nan, 'B_std': np.nan,
                    'A_list': [], 'B_list': []}
        # 二分类只有一个校准器
        cal = calibrators[0]
        try:
            As.append(float(cal.a_))
            Bs.append(float(cal.b_))
        except AttributeError:
            # 若为 isotonic 校准器则没有 a_/b_
            return {'A_mean': np.nan, 'A_std': np.nan,
                    'B_mean': np.nan, 'B_std': np.nan,
                    'A_list': [], 'B_list': []}
    return {
        'A_mean': float(np.mean(As)), 'A_std': float(np.std(As)),
        'B_mean': float(np.mean(Bs)), 'B_std': float(np.std(Bs)),
        'A_list': As, 'B_list': Bs
    }


def calibrate_model(base_estimator, X, y, method='sigmoid', cv=10):
    """
    使用 CalibratedClassifierCV 对基模型做概率校准
    - method='sigmoid' : Platt Scaling（2 参数，样本有限时稳健）
    - method='isotonic': 保序回归（非参数，需更多样本）
    - cv=10            : 与超参数调优 CV 同口径，方法学叙述统一

    注：CalibratedClassifierCV 内部会克隆基模型并训练 cv 份，
    对每份在其验证折上用 predict_proba/decision_function 输出拟合 sigmoid。
    预测时返回 cv 份校准器输出的平均值（软投票）。
    """
    try:
        cal = CalibratedClassifierCV(estimator=base_estimator,
                                     method=method, cv=cv, n_jobs=1)
    except TypeError:
        # 兼容老版本 sklearn (< 1.2) 的 base_estimator 参数名
        cal = CalibratedClassifierCV(base_estimator=base_estimator,
                                     method=method, cv=cv, n_jobs=1)
    cal.fit(X, y)
    return cal

# ============================================================
# Bootstrap / 指标 / 阈值
# ============================================================
def bootstrap_metrics(y_true,y_pred,y_proba,n_bootstrap=1000,ci=0.95):
    np.random.seed(42); n=len(y_true)
    keys=['AUC','Sensitivity','Specificity','Precision','NPV','F1',
          'Accuracy','Balanced_Acc','MCC','Youden Index','Brier Score']
    boot={k:[] for k in keys}
    for _ in range(n_bootstrap):
        idx=np.random.randint(0,n,n)
        yt=np.array(y_true)[idx]; yc=np.array(y_pred)[idx]; yp=np.array(y_proba)[idx]
        if len(np.unique(yt))<2: continue
        try:
            tn,fp,fn,tp=confusion_matrix(yt,yc).ravel()
            boot['Sensitivity'].append(tp/(tp+fn) if (tp+fn) else 0)
            boot['Specificity'].append(tn/(tn+fp) if (tn+fp) else 0)
            boot['Precision'].append(tp/(tp+fp) if (tp+fp) else 0)
            boot['NPV'].append(tn/(tn+fn) if (tn+fn) else 0)
            boot['F1'].append(2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) else 0)
            boot['Accuracy'].append((tp+tn)/(tp+tn+fp+fn))
            boot['Youden Index'].append((tp/(tp+fn)+tn/(tn+fp)-1) if ((tp+fn) and (tn+fp)) else 0)
            boot['AUC'].append(roc_auc_score(yt,yp))
            boot['Balanced_Acc'].append(balanced_accuracy_score(yt,yc))
            boot['MCC'].append(matthews_corrcoef(yt,yc))
            boot['Brier Score'].append(brier_score_loss(yt,yp))
        except Exception: continue
    alpha=(1-ci)/2; results={}
    for k,vals in boot.items():
        if not vals: continue
        v=np.array(vals)
        results[k]={'mean':np.mean(v),'ci_lower':np.percentile(v,alpha*100),
                    'ci_upper':np.percentile(v,(1-alpha)*100),
                    'ci_str':f"{np.mean(v):.3f} ({np.percentile(v,alpha*100):.3f}-{np.percentile(v,(1-alpha)*100):.3f})"}
    return results

def check_overfitting(name,tr,te):
    gap=tr-te
    if gap>0.10: s="⚠️  HIGH overfitting risk"
    elif gap>0.05: s="⚡ Mild overfitting (acceptable)"
    elif gap>-0.02: s="✅ Good generalization"
    else: s="🤔 Test > Train (rare)"
    print(f"  {MODEL_FULL_NAME.get(name,name):38s}: Gap={gap:+.4f} | {s}")
    return gap

def calculate_metrics(y_true,y_pred,y_proba):
    tn,fp,fn,tp=confusion_matrix(y_true,y_pred).ravel()
    return {
        'Sensitivity':tp/(tp+fn) if (tp+fn) else 0,
        'Specificity':tn/(tn+fp) if (tn+fp) else 0,
        'Accuracy':(tp+tn)/(tp+tn+fp+fn),
        'Precision':tp/(tp+fp) if (tp+fp) else 0,
        'NPV':tn/(tn+fn) if (tn+fn) else 0,
        'F1':2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) else 0,
        'Youden Index':(tp/(tp+fn)+tn/(tn+fp)-1) if ((tp+fn) and (tn+fp)) else 0,
        'AUC':roc_auc_score(y_true,y_proba),
        'Balanced_Acc':balanced_accuracy_score(y_true,y_pred),
        'MCC':matthews_corrcoef(y_true,y_pred),
        'Brier Score':brier_score_loss(y_true,y_proba),
        'AP':average_precision_score(y_true,y_proba),
    }

def find_optimal_threshold(model,X,y):
    """Historical archived implementation: Youden threshold from calibrated training-set probabilities (not strict cross-fitted OOF)."""
    yp=model.predict_proba(X)[:,1]; best_s,best_t=-np.inf,0.5
    for t in np.arange(0.05,0.95,0.005):
        yc=(yp>=t).astype(int); cm=confusion_matrix(y,yc)
        if cm.shape==(2,2):
            tn,fp,fn,tp=cm.ravel()
            s=(tp/(tp+fn)+tn/(tn+fp)-1) if ((tp+fn) and (tn+fp)) else 0
        else: s=0
        if s>best_s: best_s,best_t=s,t
    return best_t

# ============================================================
# 数据加载 + 特征选择
# ============================================================
print("="*70)
print("🚀 Pulmonary Infection Prediction — ML Classification (8 Models)")
print("   [ 10 Features · Platt Scaling Calibration · Checkpoint-Enabled ]")
print("="*70)

data=pd.read_csv(data_file, encoding='utf-8-sig')
data.columns=data.columns.str.replace('\ufeff','').str.strip()
print(f"\n✅ 原始数据: {data.shape[0]} 行 × {data.shape[1]} 列")

# Preserve a stable local row identifier only for optional restricted prediction export.
# If the archived source already contains Study_row_id, use it; otherwise fall back to
# the 1-based original row number. This identifier is not used for model fitting.
if 'Study_row_id' in data.columns:
    _study_row_id_all = data['Study_row_id'].copy()
else:
    _study_row_id_all = pd.Series(np.arange(1, len(data) + 1), index=data.index, name='Study_row_id')

missing = [c for c in FEATURES_USED + [OUTCOME] if c not in data.columns]
if missing:
    print(f"\n❌ 缺失列: {missing}"); sys.exit(1)

# 选择最终建模数据
data = data[[OUTCOME] + FEATURES_USED].copy()
print(f"\n🎯 建模数据: {data.shape[0]} 行 × {data.shape[1]} 列")
print(f"   结局: {OUTCOME}")
print(f"   特征({len(FEATURES_USED)}): {FEATURES_USED}")
print(f"   校准: method={CALIBRATION_METHOD}, cv={CALIBRATION_CV}")

if EXPORT_PATIENT_LEVEL:
    _model_export = data.copy()
    _model_export.insert(0, 'Study_row_id', _study_row_id_all.loc[_model_export.index].values)
    save_csv(_model_export, 'restricted_modeling_dataset_10features.csv')
else:
    print("  🔒 Patient-level modeling dataset export skipped (use --export-patient-level only for local restricted QA).")

# 结局处理
if data[OUTCOME].dtype == object:
    data[OUTCOME] = data[OUTCOME].map({'No':0,'Yes':1,'no':0,'yes':1,'0':0,'1':1,0:0,1:1})
X_raw = data[FEATURES_USED].copy()
y     = data[OUTCOME].astype(int)

# 自动识别变量类型
binary_features=[]; continuous_features=[]
for col in X_raw.columns:
    if X_raw[col].nunique()<=5 or X_raw[col].dtype==object:
        binary_features.append(col)
    else:
        continuous_features.append(col)

print(f"\n🔍 Feature types:")
print(f"   Categorical ({len(binary_features)}): {binary_features}")
print(f"   Continuous  ({len(continuous_features)}): {continuous_features}")
print(f"\n📊 Outcome: Negative={sum(y==0)}, Positive={sum(y==1)} ({sum(y==1)/len(y)*100:.1f}%)")

# 缺失值处理
missing_info = X_raw.isna().sum()
if missing_info.sum() > 0:
    print(f"\n⚠️ 缺失值:")
    for col, n_miss in missing_info[missing_info>0].items():
        print(f"   {col}: {n_miss}")
    for col in continuous_features:
        if X_raw[col].isna().any(): X_raw[col] = X_raw[col].fillna(X_raw[col].median())
    for col in binary_features:
        if X_raw[col].isna().any(): X_raw[col] = X_raw[col].fillna(X_raw[col].mode()[0])
    print(f"   ✓ 已填充")
else:
    print(f"\n✅ 无缺失值")

# 拆分 + 标准化
X_train_raw,X_test_raw,y_train,y_test=train_test_split(
    X_raw,y,test_size=0.3,random_state=42,stratify=y)
print(f"\n📊 Split: Train={len(y_train)} | Test={len(y_test)}")

scaler=MinMaxScaler(); X_train_s=X_train_raw.copy(); X_test_s=X_test_raw.copy()
exist_cont=[f for f in continuous_features if f in X_train_raw.columns]
if exist_cont:
    X_train_s[exist_cont]=scaler.fit_transform(X_train_raw[exist_cont])
    X_test_s[exist_cont] =scaler.transform(X_test_raw[exist_cont])

X_train_arr=X_train_s.values; X_test_arr=X_test_s.values
feature_names=list(X_raw.columns)

joblib.dump(scaler,        os.path.join(output_path,'标准化器.pkl'), compress=COMPRESS_LEVEL)
joblib.dump(feature_names, os.path.join(output_path,'特征名称列表.pkl'), compress=COMPRESS_LEVEL)

# ============================================================
# 模型配置
# ============================================================
print(f"\n{'='*70}\n🤖 Model Configuration\n{'='*70}")

models_config = {
    'RF': {
        'model': RandomForestClassifier(random_state=42, n_jobs=1, oob_score=True),
        'params': {
            'n_estimators':Integer(300,600),'max_depth':Integer(4,10),
            'min_samples_split':Integer(5,20),'min_samples_leaf':Integer(5,15),
            'max_features':Categorical(['sqrt']),
            'criterion':Categorical(['gini','entropy']),
            'class_weight':Categorical(['balanced',None]),
            'max_samples':Real(0.6,0.85),
        }
    },
    'GBDT': {
        'model': GradientBoostingClassifier(random_state=42),
        'params': {
            'n_estimators':Integer(100,400),'max_depth':Integer(3,7),
            'learning_rate':Real(0.01,0.15,prior='log-uniform'),
            'min_samples_split':Integer(5,30),'min_samples_leaf':Integer(5,20),
            'subsample':Real(0.7,0.9),'max_features':Categorical(['sqrt','log2']),
        }
    },
    'LR': {
        'model': LogisticRegression(random_state=42, max_iter=3000),
        'params': {
            'C':Real(0.001,100,prior='log-uniform'),
            'penalty':Categorical(['l2']),
            'solver':Categorical(['lbfgs','saga','newton-cg']),
        }
    },
    'NB': {
        'model': GaussianNB(),
        'params': {'var_smoothing':Real(1e-11,1e-5,prior='log-uniform')}
    },
    'DT': {
        'model': DecisionTreeClassifier(random_state=42),
        'params': {
            'max_depth':Integer(3,15),'min_samples_split':Integer(10,50),
            'min_samples_leaf':Integer(5,20),
            'criterion':Categorical(['gini','entropy']),
            'max_features':Categorical(['sqrt','log2']),
            'ccp_alpha':Real(0.001,0.05),
        }
    },
    'LightGBM': {
        'model': LGBMClassifier(random_state=42, verbosity=-1, n_jobs=1, force_col_wise=True),
        'params': {
            'n_estimators':Integer(100,400),'max_depth':Integer(3,7),
            'learning_rate':Real(0.01,0.1,prior='log-uniform'),
            'num_leaves':Integer(10,40),'min_child_samples':Integer(20,60),
            'reg_alpha':Real(0.05,2.0,prior='log-uniform'),
            'reg_lambda':Real(0.05,2.0,prior='log-uniform'),
            'feature_fraction':Real(0.5,0.85),'bagging_fraction':Real(0.5,0.85),
            'bagging_freq':Integer(1,7),
        }
    },
    'XGBoost': {
        'model': XGBClassifier(random_state=42, verbosity=0, eval_metric='logloss', n_jobs=1),
        'params': {
            'n_estimators':Integer(100,400),'max_depth':Integer(3,6),
            'learning_rate':Real(0.01,0.1,prior='log-uniform'),
            'subsample':Real(0.6,0.85),'colsample_bytree':Real(0.6,0.85),
            'reg_alpha':Real(0.01,2.0,prior='log-uniform'),
            'reg_lambda':Real(0.1,5.0,prior='log-uniform'),
            'min_child_weight':Integer(3,10),'gamma':Real(0.1,1.0),
        }
    },
    'MLP': {
        'model': MLPClassifier(random_state=42, max_iter=500,
                               hidden_layer_sizes=(64,32),
                               early_stopping=True, validation_fraction=0.1),
        'params': {
            'activation':Categorical(['relu','tanh']),
            'alpha':Real(1e-4,1e-1,prior='log-uniform'),
            'learning_rate_init':Real(1e-4,5e-3,prior='log-uniform'),
            'batch_size':Integer(32,128),
        }
    },
}

MODEL_ORDER_FIXED = list(models_config.keys())
for name in MODEL_ORDER_FIXED:
    njobs = MODEL_N_JOBS.get(name, GLOBAL_N_JOBS)
    print(f"  - {name:10s} ({MODEL_FULL_NAME[name]:32s}) n_jobs={njobs}")

MODEL_N_ITER = {'RF':60, 'LightGBM':50, 'XGBoost':50}
DEFAULT_N_ITER = 40

# ============================================================
# 模型训练（贝叶斯调参 → Platt 10-fold CV 校准 → 阈值冻结 → 断点续训）
# ============================================================
print(f"\n{'='*70}")
print(f"🏋️  Training pipeline:")
print(f"     1) Bayesian hyperparameter search (10-fold CV, scoring=roc_auc)")
print(f"     2) Platt Scaling calibration (method={CALIBRATION_METHOD}, cv={CALIBRATION_CV})")
print(f"     3) Youden threshold from calibrated training-set probabilities (historical archived implementation; not strict cross-fitted OOF)")
print(f"     4) Checkpoint after each model")
print(f"{'='*70}")

# 容器
best_models={}            # 校准后的最终模型
raw_models={}             # 校准前的基模型（贝叶斯搜索结果）
optimal_thresholds={}
test_results={}; train_results={}
test_probas={}; train_probas={}               # 校准后概率
test_probas_raw={}; train_probas_raw={}       # 校准前概率
cv_aucs={}; best_params_dict={}
platt_params={}           # Platt (A, B) 参数
calib_metrics={}          # 校准截距/斜率（训练/测试 × 校准前/后）

def ckpt_path(name): return os.path.join(ckpt_dir, f'ckpt_{name}.pkl')

total_start=time.time()

for name, config in models_config.items():
    print(f"\n{'─'*58}")
    print(f"🔄 Model: {MODEL_FULL_NAME[name]}")

    cpath = ckpt_path(name)
    if os.path.exists(cpath):
        try:
            saved = joblib.load(cpath)
            best_models[name]=saved['model']; raw_models[name]=saved['raw_model']
            optimal_thresholds[name]=saved['threshold']
            test_results[name]=saved['tm_te']; train_results[name]=saved['tm_tr']
            test_probas[name]=saved['yp_te']; train_probas[name]=saved['yp_tr']
            test_probas_raw[name]=saved['yp_te_raw']; train_probas_raw[name]=saved['yp_tr_raw']
            cv_aucs[name]=saved['cv_auc']; best_params_dict[name]=saved['best_params']
            platt_params[name]=saved['platt_params']
            calib_metrics[name]=saved['calib_metrics']
            print(f"  ✅ 从断点加载")
            print(f"  CV-AUC: {saved['cv_auc']:.4f} | Test AUC (cal): {saved['tm_te']['AUC']:.4f}")
            cmt=calib_metrics[name]['test_cal']
            print(f"  📐 Calibration (Test, calibrated): Int={cmt['intercept']:+.3f}, Slope={cmt['slope']:.3f}")
            continue
        except Exception as e:
            print(f"  ⚠️ 断点加载失败 ({e})，重新训练")

    n_iter=MODEL_N_ITER.get(name,DEFAULT_N_ITER)
    njobs =MODEL_N_JOBS.get(name,GLOBAL_N_JOBS)
    print(f"  Iterations={n_iter}  n_jobs={njobs}")
    t0=time.time()
    try:
        # ─── 1. 贝叶斯搜索最优超参数 ───
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bayes=BayesSearchCV(
                config['model'], config['params'],
                n_iter=n_iter,
                cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
                scoring='roc_auc', random_state=42,
                n_jobs=njobs, verbose=0, refit=True,
            )
            bayes.fit(X_train_arr, y_train)

        mdl_raw = bayes.best_estimator_
        raw_models[name] = mdl_raw
        cv_aucs[name] = bayes.best_score_
        best_params_dict[name] = dict(bayes.best_params_)

        # ─── 2. Platt Scaling 概率校准（10 折 CV）───
        print(f"  ⚙️  应用概率校准 ({CALIBRATION_METHOD}, cv={CALIBRATION_CV}) ...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mdl = calibrate_model(mdl_raw, X_train_arr, y_train,
                                  method=CALIBRATION_METHOD,
                                  cv=CALIBRATION_CV)

        # ─── 3. 历史归档实现：基于训练集校准后预测确定Youden阈值（非严格cross-fitted OOF）───
        opt_thresh = find_optimal_threshold(mdl, X_train_arr, y_train)

        # 校准后概率
        yp_te=mdl.predict_proba(X_test_arr)[:,1]
        yc_te=(yp_te>=opt_thresh).astype(int)
        tm_te=calculate_metrics(y_test, yc_te, yp_te)

        yp_tr=mdl.predict_proba(X_train_arr)[:,1]
        yc_tr=(yp_tr>=opt_thresh).astype(int)
        tm_tr=calculate_metrics(y_train, yc_tr, yp_tr)

        # 校准前概率（对比用）
        yp_te_raw=mdl_raw.predict_proba(X_test_arr)[:,1]
        yp_tr_raw=mdl_raw.predict_proba(X_train_arr)[:,1]

        # ─── 4. 保存结果 ───
        best_models[name]=mdl; optimal_thresholds[name]=opt_thresh
        test_results[name]=tm_te; train_results[name]=tm_tr
        test_probas[name]=yp_te; train_probas[name]=yp_tr
        test_probas_raw[name]=yp_te_raw; train_probas_raw[name]=yp_tr_raw

        # Platt (A, B) 参数提取
        platt_params[name] = extract_platt_params(mdl)

        # 校准截距/斜率（训练/测试 × 校准前/后）
        calib_metrics[name] = {
            'train_raw': calibration_intercept_slope(y_train, yp_tr_raw),
            'train_cal': calibration_intercept_slope(y_train, yp_tr),
            'test_raw':  calibration_intercept_slope(y_test,  yp_te_raw),
            'test_cal':  calibration_intercept_slope(y_test,  yp_te),
        }

        # 输出摘要
        brier_raw  = brier_score_loss(y_test, yp_te_raw)
        brier_cal  = brier_score_loss(y_test, yp_te)
        auc_raw    = roc_auc_score(y_test, yp_te_raw)
        auc_cal    = roc_auc_score(y_test, yp_te)
        cm_te_cal  = calib_metrics[name]['test_cal']

        print(f"  CV-AUC: {bayes.best_score_:.4f}")
        print(f"  Train AUC (cal): {tm_tr['AUC']:.4f} | F1={tm_tr['F1']:.4f}")
        print(f"  Test  AUC (cal): {tm_te['AUC']:.4f} | F1={tm_te['F1']:.4f} | Threshold={opt_thresh:.3f}")
        print(f"  📐 校准前 → 后（测试集）: "
              f"AUC {auc_raw:.4f} → {auc_cal:.4f} | "
              f"Brier {brier_raw:.4f} → {brier_cal:.4f}")
        print(f"  📐 校准后 测试集: "
              f"Intercept={cm_te_cal['intercept']:+.3f}, "
              f"Slope={cm_te_cal['slope']:.3f}")
        pp = platt_params[name]
        if not np.isnan(pp['A_mean']):
            print(f"  📐 Platt (A, B): A={pp['A_mean']:+.3f} (±{pp['A_std']:.3f}), "
                  f"B={pp['B_mean']:+.3f} (±{pp['B_std']:.3f})")
        check_overfitting(name, tm_tr['AUC'], tm_te['AUC'])
        print(f"  Best params: {best_params_dict[name]}")
        print(f"  ⏱️  {(time.time()-t0)/60:.1f} min")

        # ─── 5. 保存断点 ───
        try:
            joblib.dump({
                'model':mdl, 'raw_model':mdl_raw, 'threshold':opt_thresh,
                'tm_te':tm_te, 'tm_tr':tm_tr,
                'yp_te':yp_te, 'yp_tr':yp_tr,
                'yp_te_raw':yp_te_raw, 'yp_tr_raw':yp_tr_raw,
                'cv_auc':bayes.best_score_,
                'best_params':best_params_dict[name],
                'platt_params':platt_params[name],
                'calib_metrics':calib_metrics[name],
            }, cpath, compress=COMPRESS_LEVEL)
            print(f"  💾 断点已保存: ckpt_{name}.pkl")
        except Exception as save_err:
            print(f"  ⚠️ 断点保存失败: {save_err}")

        del bayes; gc.collect()

    except Exception as e:
        print(f"  ❌ Failed: {e}")
        import traceback; traceback.print_exc()
        gc.collect()

print(f"\n✅ 全部模型处理完成. Total: {(time.time()-total_start)/60:.1f} min")
print(f"成功模型: {len(best_models)}/{len(models_config)}")

if len(best_models)==0:
    print("\n❌ 没有任何模型训练成功，退出"); sys.exit(1)

print(f"\n{'='*70}\n📊 Overfitting Diagnostic Summary\n{'='*70}")
for name in best_models:
    check_overfitting(name, train_results[name]['AUC'], test_results[name]['AUC'])

model_order=sorted(best_models.keys(), key=lambda n: test_results[n]['AUC'], reverse=True)


# ============================================================
# 保存结果 CSV（含校准相关输出）
# ============================================================

# ─── Optional restricted patient-level prediction export ───
# These files are deliberately disabled by default because they contain one row per patient.
# Use --export-patient-level only in a local restricted workspace; never commit the generated
# CSV files to the public repository.
if EXPORT_PATIENT_LEVEL:
    _test_ids = _study_row_id_all.loc[X_test_raw.index].to_numpy()
    _train_ids = _study_row_id_all.loc[X_train_raw.index].to_numpy()

    probas_df=pd.DataFrame(test_probas)
    probas_df.columns=[f'{MODEL_FULL_NAME[n]} Prob' for n in probas_df.columns]
    probas_df.insert(0, 'Study_row_id', _test_ids)
    probas_df['True Label']=y_test.values
    save_csv(probas_df,'historical_BSA10_test_predictions_calibrated_with_StudyRowID.csv')

    probas_raw_df=pd.DataFrame(test_probas_raw)
    probas_raw_df.columns=[f'{MODEL_FULL_NAME[n]} Prob' for n in probas_raw_df.columns]
    probas_raw_df.insert(0, 'Study_row_id', _test_ids)
    probas_raw_df['True Label']=y_test.values
    save_csv(probas_raw_df,'historical_BSA10_test_predictions_raw_with_StudyRowID.csv')

    # Training probabilities are retained only for local historical figure/QA reconstruction.
    probas_train_df=pd.DataFrame(train_probas)
    probas_train_df.columns=[f'{MODEL_FULL_NAME[n]} Prob' for n in probas_train_df.columns]
    probas_train_df.insert(0, 'Study_row_id', _train_ids)
    probas_train_df['True Label']=y_train.values
    save_csv(probas_train_df,'historical_BSA10_train_predictions_calibrated_with_StudyRowID.csv')

    probas_train_raw_df=pd.DataFrame(train_probas_raw)
    probas_train_raw_df.columns=[f'{MODEL_FULL_NAME[n]} Prob' for n in probas_train_raw_df.columns]
    probas_train_raw_df.insert(0, 'Study_row_id', _train_ids)
    probas_train_raw_df['True Label']=y_train.values
    save_csv(probas_train_raw_df,'historical_BSA10_train_predictions_raw_with_StudyRowID.csv')
else:
    print("  🔒 Patient-level prediction CSV export skipped (use --export-patient-level for restricted local reconstruction).")

# ─── 模型文件 ───
joblib.dump(best_models,        os.path.join(output_path,'best_models_calibrated.pkl'), compress=COMPRESS_LEVEL)
joblib.dump(raw_models,         os.path.join(output_path,'raw_models_uncalibrated.pkl'), compress=COMPRESS_LEVEL)
joblib.dump(optimal_thresholds, os.path.join(output_path,'最优阈值字典.pkl'), compress=COMPRESS_LEVEL)
joblib.dump(platt_params,       os.path.join(output_path,'Platt参数字典.pkl'), compress=COMPRESS_LEVEL)
joblib.dump(calib_metrics,      os.path.join(output_path,'校准指标字典.pkl'), compress=COMPRESS_LEVEL)

print(f"\n💡 模型保存路径:")
print(f"   · 校准后模型: best_models_calibrated.pkl")
print(f"   · 校准前模型: raw_models_uncalibrated.pkl")
print(f"   · 单模型断点: {ckpt_dir}")

# ─── 输出：Platt Scaling 参数表 ───
platt_rows=[]
for n in model_order:
    pp = platt_params[n]
    platt_rows.append({
        'Model': MODEL_FULL_NAME[n],
        'A_mean':    round(pp['A_mean'], 4) if not np.isnan(pp['A_mean']) else 'N/A',
        'A_std':     round(pp['A_std'], 4)  if not np.isnan(pp['A_std'])  else 'N/A',
        'B_mean':    round(pp['B_mean'], 4) if not np.isnan(pp['B_mean']) else 'N/A',
        'B_std':     round(pp['B_std'], 4)  if not np.isnan(pp['B_std'])  else 'N/A',
        'Threshold': round(optimal_thresholds[n], 3),
        'Calibration_Scheme': f"{CALIBRATION_METHOD}(cv={CALIBRATION_CV})"
    })
save_csv(pd.DataFrame(platt_rows), 'Platt_Scaling参数表.csv')

# ─── 输出：完整校准指标对比表（训练/测试 × 校准前/后）───
full_cal_rows=[]
for n in model_order:
    cm = calib_metrics[n]
    # 训练集
    full_cal_rows.append({
        'Model': MODEL_FULL_NAME[n], 'Dataset':'Train', 'Calibration':'Raw',
        'Brier':     round(brier_score_loss(y_train, train_probas_raw[n]), 4),
        'Intercept': round(cm['train_raw']['intercept'], 4),
        'Slope':     round(cm['train_raw']['slope'], 4)
    })
    full_cal_rows.append({
        'Model': MODEL_FULL_NAME[n], 'Dataset':'Train', 'Calibration':'Calibrated',
        'Brier':     round(brier_score_loss(y_train, train_probas[n]), 4),
        'Intercept': round(cm['train_cal']['intercept'], 4),
        'Slope':     round(cm['train_cal']['slope'], 4)
    })
    # 测试集
    full_cal_rows.append({
        'Model': MODEL_FULL_NAME[n], 'Dataset':'Test', 'Calibration':'Raw',
        'Brier':     round(brier_score_loss(y_test, test_probas_raw[n]), 4),
        'Intercept': round(cm['test_raw']['intercept'], 4),
        'Slope':     round(cm['test_raw']['slope'], 4)
    })
    full_cal_rows.append({
        'Model': MODEL_FULL_NAME[n], 'Dataset':'Test', 'Calibration':'Calibrated',
        'Brier':     round(brier_score_loss(y_test, test_probas[n]), 4),
        'Intercept': round(cm['test_cal']['intercept'], 4),
        'Slope':     round(cm['test_cal']['slope'], 4)
    })
save_csv(pd.DataFrame(full_cal_rows), '校准指标完整对比_训练测试_校准前后.csv')

# ─── 输出：校准前后简要对比表（测试集）───
cal_cmp=[]
for n in model_order:
    cal_cmp.append({
        'Model': MODEL_FULL_NAME[n],
        'AUC_Raw':          round(roc_auc_score(y_test, test_probas_raw[n]), 4),
        'AUC_Calibrated':   round(roc_auc_score(y_test, test_probas[n]), 4),
        'Brier_Raw':        round(brier_score_loss(y_test, test_probas_raw[n]), 4),
        'Brier_Calibrated': round(brier_score_loss(y_test, test_probas[n]), 4),
        'Brier_Improvement':
            round(brier_score_loss(y_test, test_probas_raw[n])
                  - brier_score_loss(y_test, test_probas[n]), 4),
        'Intercept_Calibrated': round(calib_metrics[n]['test_cal']['intercept'], 4),
        'Slope_Calibrated':     round(calib_metrics[n]['test_cal']['slope'], 4)
    })
save_csv(pd.DataFrame(cal_cmp), '校准前后对比_测试集.csv')

# Bootstrap
print(f"\n📊 Bootstrap 1000 iterations (on calibrated probabilities)...")
boot_test={}; boot_train={}
for name in model_order:
    print(f"  {MODEL_FULL_NAME[name]}...", end=' ', flush=True)
    yp=test_probas[name]; yc=(yp>=optimal_thresholds[name]).astype(int)
    boot_test[name]=bootstrap_metrics(y_test,yc,yp)
    ytp=train_probas[name]; ytc=(ytp>=optimal_thresholds[name]).astype(int)
    boot_train[name]=bootstrap_metrics(y_train,ytc,ytp)
    print("✓")

key_metrics=['AUC','Sensitivity','Specificity','Precision','NPV','F1','MCC','Accuracy','Youden Index','Brier Score']

def make_point_df(results_dict, thresholds):
    rows=[]
    for name in model_order:
        row={'Model':MODEL_FULL_NAME[name],'Optimal Threshold':round(thresholds[name],4)}
        for k,v in results_dict[name].items():
            if k!='AP': row[k]=round(v,4)
        rows.append(row)
    return pd.DataFrame(rows)

save_csv(make_point_df(test_results, optimal_thresholds), '模型评估_测试集_点估计.csv')
save_csv(make_point_df(train_results, optimal_thresholds), '模型评估_训练集_点估计.csv')

ci_rows=[{'Model':MODEL_FULL_NAME[n],
          **{m:boot_test[n][m]['ci_str'] for m in key_metrics if m in boot_test[n]}}
         for n in model_order]
save_csv(pd.DataFrame(ci_rows),'模型评估_测试集_95置信区间.csv')

# ─── 训练集 95%CI CSV（新增，供图17 训练集圆盘图使用）───
ci_rows_train=[{'Model':MODEL_FULL_NAME[n],
                **{m:boot_train[n][m]['ci_str'] for m in key_metrics if m in boot_train[n]}}
               for n in model_order]
save_csv(pd.DataFrame(ci_rows_train),'模型评估_训练集_95置信区间.csv')

cmp_rows=[]
for name in model_order:
    ta=train_results[name]['AUC']; ea=test_results[name]['AUC']; gap=ta-ea
    cmp_rows.append({'Model':MODEL_FULL_NAME[name],
                     'Train AUC':f"{ta:.4f}",'Train AUC 95%CI':boot_train[name]['AUC']['ci_str'],
                     'Test AUC':f"{ea:.4f}",'Test AUC 95%CI':boot_test[name]['AUC']['ci_str'],
                     'Gap':f"{gap:+.4f}",
                     'Status':'✓ Good' if abs(gap)<0.05 else ('⚡ Mild' if gap<0.10 else '⚠ High')})
save_csv(pd.DataFrame(cmp_rows),'训练集测试集AUC对比.csv')

# DeLong
best_name=model_order[0]; best_proba=test_probas[best_name]
print(f"\n📊 DeLong Test — Reference: {MODEL_FULL_NAME[best_name]}")
delong_rows=[]
for name in model_order:
    if name==best_name: continue
    a1,a2,z,p=delong_test(y_test,best_proba,test_probas[name])
    sig="***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns"
    delong_rows.append({'Comparison Model':MODEL_FULL_NAME[name],'Comparison AUC':round(a2,4),
                        'Reference Model':MODEL_FULL_NAME[best_name],'Reference AUC':round(a1,4),
                        'ΔAUC':round(a1-a2,4),'Z':round(z,3),'P Value':round(p,4),'Sig':sig})
    print(f"  {MODEL_FULL_NAME[name]:38s} ΔAUC={a1-a2:.4f}  p={p:.4f} {sig}")
save_csv(pd.DataFrame(delong_rows),'DeLong检验结果.csv')

# HL 检验（基于校准后概率）
print(f"\n📊 Hosmer-Lemeshow Test (on calibrated probabilities)")
hl_rows=[]
for name in model_order:
    hl_s,hl_p,_=hosmer_lemeshow_test(y_test,test_probas[name])
    brier=brier_score_loss(y_test,test_probas[name])
    cal="Well-calibrated (p>0.05)" if hl_p>0.05 else "Poor calibration (p≤0.05)"
    print(f"  {MODEL_FULL_NAME[name]:38s} HL={hl_s:.2f}  p={hl_p:.4f}  {cal}")
    hl_rows.append({'Model':MODEL_FULL_NAME[name],'HL Statistic':round(hl_s,4),
                    'P Value':round(hl_p,4),'Calibration':cal,'Brier Score':round(brier,4)})
save_csv(pd.DataFrame(hl_rows),'Hosmer-Lemeshow校准检验结果.csv')


# ============================================================
# 绘图
# ============================================================
print(f"\n{'='*70}\n📈 Generating Figures\n{'='*70}")

def get_color(n): return MODEL_COLORS.get(n,'#333333')
def get_label(n): return MODEL_FULL_NAME[n]

# 图1: ROC 训练+测试（校准后）
fig,ax=plt.subplots(figsize=(13,10),facecolor='white')
for name in model_order:
    fpr,tpr,_=roc_curve(y_test,test_probas[name])
    auc_te=test_results[name]['AUC']
    ci_lo=boot_test[name]['AUC']['ci_lower']; ci_hi=boot_test[name]['AUC']['ci_upper']
    ax.plot(fpr,tpr,color=get_color(name),lw=2.0,
            label=f"{get_label(name)}  AUC={auc_te:.3f} ({ci_lo:.3f}-{ci_hi:.3f})")
    fpr_tr,tpr_tr,_=roc_curve(y_train,train_probas[name])
    ax.plot(fpr_tr,tpr_tr,color=get_color(name),lw=1.2,ls='--',alpha=0.4,
            label=f"{get_label(name)} (Train) AUC={train_results[name]['AUC']:.3f}")
ax.plot([0,1],[0,1],'k--',lw=1,alpha=0.5)
ax.set_xlim([-0.02,1.0]); ax.set_ylim([-0.02,1.05])
ax.set_xlabel("False Positive Rate (1 - Specificity)",fontsize=22,fontweight='bold')
ax.set_ylabel("True Positive Rate (Sensitivity)",fontsize=22,fontweight='bold')
# 标题已按要求删除
leg=ax.legend(loc='lower right',fontsize=12,prop={'weight':'bold','size':12}); ax.grid(True,alpha=0.3)
plt.tight_layout(); save_fig('ROC曲线_训练集与测试集.png'); plt.close()

# 图2: ROC 仅测试集（校准后）
fig,ax=plt.subplots(figsize=(10,8),facecolor='white')
for name in model_order:
    fpr,tpr,_=roc_curve(y_test,test_probas[name])
    auc_te=test_results[name]['AUC']
    ci_lo=boot_test[name]['AUC']['ci_lower']; ci_hi=boot_test[name]['AUC']['ci_upper']
    ax.plot(fpr,tpr,color=get_color(name),lw=2.5,
            label=f"{get_label(name)}  {auc_te:.3f} ({ci_lo:.3f}-{ci_hi:.3f})")
ax.plot([0,1],[0,1],'k--',lw=1,alpha=0.5)
ax.set_xlim([-0.02,1.0]); ax.set_ylim([-0.02,1.05])
ax.set_xlabel("False Positive Rate (1 - Specificity)",fontsize=22,fontweight='bold')
ax.set_ylabel("True Positive Rate (Sensitivity)",fontsize=22,fontweight='bold')
# 标题已按要求删除
leg=ax.legend(loc='lower right',fontsize=15,title='Model  AUC (95% CI)',
              prop={'weight':'bold','size':15},title_fontsize=16)
plt.setp(leg.get_title(),fontweight='bold')
ax.grid(True,alpha=0.3); plt.tight_layout()
save_fig('ROC曲线_测试集.png'); plt.close()

# 图3: PR / AP 曲线（优化版：放大字体、阶梯线、图例保留在图内、去标题）
fig, ax = plt.subplots(figsize=(15, 9), facecolor='white')
baseline = float(y_test.mean())
pr_ap_rows = []
for name in model_order:
    prec, rec, _ = precision_recall_curve(y_test, test_probas[name])
    ap = average_precision_score(y_test, test_probas[name])
    # PR 曲线使用阶梯线显示，更清爽，也更符合 PR 曲线常见展示方式
    ax.step(rec, prec, where='post', color=get_color(name), lw=3.0,
            label=f"{get_label(name)}  AP={ap:.3f}")
    pr_ap_rows.append({'Model': get_label(name), 'AP': round(ap, 4)})

ax.axhline(y=baseline, ls='--', color='gray', lw=2.2,
           label=f'No-skill ({baseline:.3f})')
ax.set_xlim([-0.02, 1.02])
# 去掉下方过多空白，让曲线主体更饱满；如需完整 0-1 轴，可改回 ax.set_ylim([0, 1.05])
ax.set_ylim([max(0, baseline - 0.08), 1.03])
ax.set_xlabel("Recall", fontsize=PR_DCA_FS_LABEL, fontweight='bold')
ax.set_ylabel("Precision", fontsize=PR_DCA_FS_LABEL, fontweight='bold')
ax.tick_params(axis='both', labelsize=PR_DCA_FS_TICK, width=2.0, length=7)
for lb in ax.get_xticklabels() + ax.get_yticklabels():
    lb.set_fontweight('bold')
# 图例保留在图内左下区域，尽量避开主曲线；使用白底增强可读性
leg = ax.legend(loc='lower left', bbox_to_anchor=(0.02, 0.02),
                fontsize=PR_DCA_FS_LEGEND, prop={'weight': 'bold', 'size':PR_DCA_FS_LEGEND},
                frameon=True, fancybox=True, framealpha=0.92,
                borderpad=0.6, labelspacing=0.5, handlelength=2.4)
leg.get_frame().set_facecolor('white')
leg.get_frame().set_edgecolor('#CCCCCC')
ax.grid(True, alpha=0.3)
plt.tight_layout()
save_fig('精确率召回率曲线_APC_测试集.png')
plt.close()
save_csv(pd.DataFrame(pr_ap_rows), '精确率召回率曲线_AP值汇总.csv')

# ============================================================
# 图4: 校准曲线 4 面板（训练/测试 × 校准前/后）
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(18, 14), facecolor='white')
# 每个面板额外带上 calib_metrics 的 key，用于查找 intercept / slope
panels = [
    (axes[0,0], train_probas_raw, y_train, 'Training — Before Calibration (Raw)',                              'train_raw'),
    (axes[0,1], train_probas,     y_train, f'Training — After Calibration ({CALIBRATION_METHOD.title()})',     'train_cal'),
    (axes[1,0], test_probas_raw,  y_test,  'Test — Before Calibration (Raw)',                                  'test_raw'),
    (axes[1,1], test_probas,      y_test,  f'Test — After Calibration ({CALIBRATION_METHOD.title()})',         'test_cal'),
]
for ax, probas, y_ref, title, cm_key in panels:
    ax.plot([0,1],[0,1],'k--',lw=1.5,alpha=0.6,label='Perfect calibration')
    for n in model_order:
        prob_true, prob_pred = calibration_curve(y_ref, probas[n],
                                                 n_bins=10, strategy='quantile')
        brier = brier_score_loss(y_ref, probas[n])
        inter = calib_metrics[n][cm_key]['intercept']
        slope = calib_metrics[n][cm_key]['slope']
        ax.plot(prob_pred, prob_true, 'o-', color=get_color(n), lw=2,
                ms=5,
                label=f"{get_label(n)}  Brier={brier:.3f}, Int={inter:+.2f}, Slope={slope:.2f}")
    ax.set_xlim([-0.02,1.02]); ax.set_ylim([-0.02,1.05])
    ax.set_xlabel('Mean Predicted Probability', fontsize=18, fontweight='bold')
    ax.set_ylabel('Fraction of Positives',      fontsize=18, fontweight='bold')
    # 标题已按要求删除
    ax.legend(loc='upper left', fontsize=11, prop={'weight':'bold','size':11}); ax.grid(alpha=0.3)
# 标题已按要求删除
plt.tight_layout(); save_fig('校准曲线_4面板_校准前后对比.png'); plt.close()

# ============================================================
# 图4b: 测试集 校准曲线 双面板（校准前 / 校准后）— 不带 suptitle
# 与图4 同字号/同配色/同布局，仅保留下半部分（Test 行）
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(20, 7.2), facecolor='white')
panels_test = [
    (axes[0], test_probas_raw, y_test, 'Test — Before Calibration (Raw)',
     'test_raw'),
    (axes[1], test_probas,     y_test, f'Test — After Calibration ({CALIBRATION_METHOD.title()})',
     'test_cal'),
]
for ax, probas, y_ref, title, cm_key in panels_test:
    ax.plot([0,1],[0,1],'k--',lw=1.5,alpha=0.6,label='Perfect calibration')
    for n in model_order:
        prob_true, prob_pred = calibration_curve(y_ref, probas[n],
                                                 n_bins=10, strategy='quantile')
        brier = brier_score_loss(y_ref, probas[n])
        inter = calib_metrics[n][cm_key]['intercept']
        slope = calib_metrics[n][cm_key]['slope']
        ax.plot(prob_pred, prob_true, 'o-', color=get_color(n), lw=2.0,
                ms=4.6,
                label=f"{get_label(n)}  Brier={brier:.3f}, Int={inter:+.2f}, Slope={slope:.2f}")
    ax.set_xlim([-0.02,1.02]); ax.set_ylim([-0.02,1.05])
    ax.set_xlabel('Mean Predicted Probability', fontsize=CAL2_FS_LABEL, fontweight='bold')
    ax.set_ylabel('Fraction of Positives',      fontsize=CAL2_FS_LABEL, fontweight='bold')
    ax.tick_params(axis='both', labelsize=CAL2_FS_TICK, width=1.6, length=6)
    for lb in ax.get_xticklabels() + ax.get_yticklabels():
        lb.set_fontweight('bold')
    # 调整为更接近四拼图的横向比例，避免每个子图显得偏“方”
    ax.set_box_aspect(CAL2_BOX_ASPECT)
    # 图例单独缩小，保留在图内左上角，避免喧宾夺主
    leg = ax.legend(loc='upper left', bbox_to_anchor=(0.015, 0.995),
                    fontsize=CAL2_FS_LEGEND,
                    prop={'weight':'bold','size':CAL2_FS_LEGEND},
                    frameon=True, fancybox=True, framealpha=0.88,
                    borderpad=0.35, labelspacing=0.35, handlelength=2.0,
                    borderaxespad=0.15)
    leg.get_frame().set_facecolor('white')
    leg.get_frame().set_edgecolor('#BFBFBF')
    ax.grid(alpha=0.3)
# ★ 注意：不添加 plt.suptitle(...)，此图无总标题
plt.subplots_adjust(wspace=0.16)
plt.tight_layout()
save_fig('校准曲线_测试集_校准前后对比.png'); plt.close()

# ============================================================
# 图5: 测试集校准后单图（论文用，含 Brier + 截距 + 斜率）
# ============================================================
fig, ax = plt.subplots(figsize=(11, 9), facecolor='white')
ax.plot([0,1],[0,1],'k--',lw=1.5,alpha=0.6,label='Perfect calibration')
for n in model_order:
    prob_true, prob_pred = calibration_curve(y_test, test_probas[n],
                                             n_bins=10, strategy='quantile')
    brier = brier_score_loss(y_test, test_probas[n])
    inter = calib_metrics[n]['test_cal']['intercept']
    slope = calib_metrics[n]['test_cal']['slope']
    ax.plot(prob_pred, prob_true, 'o-', color=get_color(n), lw=2.2, markersize=7,
            label=f"{get_label(n)}  Brier={brier:.3f}, Int={inter:+.2f}, Slope={slope:.2f}")
ax.set_xlim([-0.02,1.02]); ax.set_ylim([-0.02,1.05])
ax.set_xlabel('Mean Predicted Probability', fontsize=22, fontweight='bold')
ax.set_ylabel('Fraction of Positives',      fontsize=22, fontweight='bold')
# 标题已按要求删除
ax.legend(loc='upper left', fontsize=15, prop={'weight':'bold','size':15}); ax.grid(alpha=0.3)
plt.tight_layout(); save_fig('校准曲线_测试集_校准后.png'); plt.close()

# ============================================================
# 图6: 每个模型独立的校准曲线分图（含 HL / Brier / Int / Slope）
# ============================================================
n_rows_cal=(len(model_order)+1)//2
fig,axes=plt.subplots(n_rows_cal,2,figsize=(14,n_rows_cal*4.5),facecolor='white')
axes=axes.flatten()
for idx,name in enumerate(model_order):
    ax=axes[idx]
    ax.plot([0,1],[0,1],'k--',lw=1.5,label='Perfect',alpha=0.6)
    frac,mpred=calibration_curve(y_test,test_probas[name],n_bins=10,strategy='quantile')
    ax.plot(mpred,frac,'o-',color=get_color(name),lw=2,ms=6,label='Calibrated')
    hl_s,hl_p,_=hosmer_lemeshow_test(y_test,test_probas[name])
    brier=brier_score_loss(y_test,test_probas[name])
    inter = calib_metrics[name]['test_cal']['intercept']
    slope = calib_metrics[name]['test_cal']['slope']
    ax.text(0.04,0.95,
            f"HL={hl_s:.2f}\np={'<0.001' if hl_p<0.001 else f'{hl_p:.3f}'}\n"
            f"Brier={brier:.3f}\nInt={inter:+.2f}\nSlope={slope:.2f}",
            transform=ax.transAxes,fontsize=15,va='top',fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4',facecolor='lightyellow',alpha=0.8))
    ax.set_xlim([0,1]); ax.set_ylim([0,1])
    ax.set_xlabel("Predicted Probability",fontsize=17,fontweight='bold')
    ax.set_ylabel("Observed Frequency",fontsize=17,fontweight='bold')
    # 标题已按要求删除
    ax.legend(loc='lower right',fontsize=13,prop={'weight':'bold','size':13}); ax.grid(True,alpha=0.3)
for idx in range(len(model_order),len(axes)): axes[idx].axis('off')
# 标题已按要求删除
plt.tight_layout(); save_fig('校准曲线_含HL检验_测试集.png'); plt.close()

# ============================================================
# 图7: DCA（基于校准后概率）
# ============================================================
def calc_nb(yt,yp,t):
    if t>=0.999: return 0.0
    n=len(yt); tp=np.sum((yp>=t)&(yt==1)); fp=np.sum((yp>=t)&(yt==0))
    return tp/n-fp/n*(t/(1-t))
thresholds_dca=np.arange(0.01,0.99,0.005); prev=float(y_test.mean())
fig, ax = plt.subplots(figsize=(15, 9), facecolor='white')
nb_all = [prev - (1 - prev) * t / (1 - t) if t < 0.999 else 0 for t in thresholds_dca]
ax.plot(thresholds_dca, nb_all, 'k-', lw=2.6, label='Treat All')
ax.axhline(y=0, color='gray', ls='--', lw=2.4, label='Treat None')
for name in model_order:
    nbs = [calc_nb(y_test.values, test_probas[name], t) for t in thresholds_dca]
    ax.plot(thresholds_dca, nbs, color=get_color(name), lw=3.0,
            label=get_label(name), alpha=0.9)
ax.set_xlim([0, 1.0]); ax.set_ylim([-0.10, 0.55])
ax.set_xlabel("Threshold Probability", fontsize=PR_DCA_FS_LABEL, fontweight='bold')
ax.set_ylabel("Net Benefit", fontsize=PR_DCA_FS_LABEL, fontweight='bold')
ax.tick_params(axis='both', labelsize=PR_DCA_FS_TICK, width=2.0, length=7)
for lb in ax.get_xticklabels() + ax.get_yticklabels():
    lb.set_fontweight('bold')
# 图例改为放在图内右上角，风格参考校准曲线图例
leg = ax.legend(loc='upper right', bbox_to_anchor=(0.985, 0.985),
                fontsize=PR_DCA_FS_LEGEND, prop={'weight': 'bold', 'size':PR_DCA_FS_LEGEND},
                frameon=True, fancybox=True, framealpha=0.90,
                borderpad=0.5, labelspacing=0.5, handlelength=2.4)
leg.get_frame().set_facecolor('white')
leg.get_frame().set_edgecolor('#BFBFBF')
ax.grid(True, alpha=0.3)
plt.tight_layout()
save_fig('DCA决策曲线分析.png')
plt.close()

# ============================================================
# 图7b: DCA 阈值净获益表（临床决策辅助）
# ─────────────────────────────────────────────────────
# 配合图7（DCA 决策曲线）使用：
#   · 图7  从"曲线"视角看模型在所有阈值上的获益形状
#   · 图7b 从"表格"视角列出临床关键阈值下的精确数值与最优策略
# ─────────────────────────────────────────────────────
# 临床解读：
#   阈值 t = 临床医生愿意干预的最低风险概率
#   FP:TP 代价比 = (1-t):t   例: t=0.25 → 1 真阳性可换 3 个假阳性
#   净获益 NB = TP/n - FP/n × t/(1-t)
# 输出：
#   1. CSV  DCA阈值净获益表_测试集.csv      详细 NB 矩阵 + 最优策略 + 每千人净受益
#   2. CSV  DCA阈值模型摘要表_测试集.csv    模型视角摘要 (NB-AUC、有效阈值区间、峰值)
#   3. PNG  DCA阈值净获益热力表_测试集.png  彩色热力表（金边 = 该阈值最佳策略）
# ============================================================
import matplotlib.colors as mcolors

# ── 临床关键阈值 + 风险解读 ──
key_thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
ratio_labels   = ['1:19', '1:9', '1:5.7', '1:4', '1:3', '1:2.3', '1:1.5', '1:1']
clinical_interp = {
    0.05: 'Very low (high sensitivity, screening)',
    0.10: 'Low (early intervention)',
    0.15: 'Low-moderate (broad screening)',
    0.20: 'Moderate (active intervention)',
    0.25: 'Moderate (standard clinical)',
    0.30: 'Moderate-high (cautious)',
    0.40: 'High (conservative)',
    0.50: 'High (FP/TP balanced)',
}

prev_test = float(y_test.mean())

# ─── (A) 详细 NB 表 ─────────────────────────────────────
detail_rows = []
for t, ratio in zip(key_thresholds, ratio_labels):
    row = {
        'Threshold':               f"{t:.2f}",
        'Threshold(%)':            f"{int(t*100)}%",
        'FP:TP_Cost_Ratio':        ratio,
        'Clinical_Interpretation': clinical_interp[t],
        'Treat_All_NB':            round(prev_test - (1-prev_test)*t/(1-t), 4),
        'Treat_None_NB':           0.0000,
    }
    model_nbs = {}
    for name in model_order:
        full = MODEL_FULL_NAME[name]
        nb_v = round(calc_nb(y_test.values, test_probas[name], t), 4)
        row[f'{full}_NB'] = nb_v
        model_nbs[full]   = nb_v
    best_model = max(model_nbs, key=model_nbs.get)
    best_nb    = model_nbs[best_model]
    default_nb = max(row['Treat_All_NB'], row['Treat_None_NB'])
    row['Best_Model']                   = best_model
    row['Best_Model_NB']                = round(best_nb, 4)
    row['Best_Default_NB']              = round(default_nb, 4)
    row['ΔNB(Best_vs_Default)']         = round(best_nb - default_nb, 4)
    row['Net_TP_per_1000']              = int(round(best_nb * 1000))
    row['Net_Gain_vs_Default_per_1000'] = int(round((best_nb - default_nb) * 1000))
    detail_rows.append(row)
df_dca_detail = pd.DataFrame(detail_rows)
save_csv(df_dca_detail, 'DCA阈值净获益表_测试集.csv')

# ─── (B) 模型摘要表 ─────────────────────────────────────
fine_thr    = np.arange(0.01, 0.99, 0.005)
nb_all_fine = np.array([prev_test - (1-prev_test)*t/(1-t) if t < 0.999 else 0
                        for t in fine_thr])
# numpy 2.x 已把 np.trapz 改名为 np.trapezoid，做版本兼容
_trapz_fn = getattr(np, 'trapezoid', None) or np.trapz
summary_rows = []
for name in model_order:
    full     = MODEL_FULL_NAME[name]
    nbs_fine = np.array([calc_nb(y_test.values, test_probas[name], t) for t in fine_thr])
    nb_auc   = float(_trapz_fn(np.maximum(nbs_fine, 0), fine_thr))
    better   = nbs_fine > np.maximum(nb_all_fine, 0) + 1e-6
    if better.any():
        useful_lo, useful_hi = float(fine_thr[better].min()), float(fine_thr[better].max())
        useful_range = f"{useful_lo:.2f} – {useful_hi:.2f}"
        useful_width = round(useful_hi - useful_lo, 3)
    else:
        useful_range, useful_width = '—', 0.0
    idx_max = int(nbs_fine.argmax())
    peak_t  = float(fine_thr[idx_max]); peak_nb = float(nbs_fine[idx_max])
    srow = {'Model': full}
    for t in key_thresholds:
        srow[f'NB@{int(t*100)}%'] = round(calc_nb(y_test.values, test_probas[name], t), 4)
    srow['Peak_NB']                = round(peak_nb, 4)
    srow['Peak_Threshold']         = f"{peak_t:.2f}"
    srow['Useful_Threshold_Range'] = useful_range  # NB > Treat All & Treat None 的阈值区间
    srow['Useful_Range_Width']     = useful_width
    srow['NB_AUC(0.01-0.99)']      = round(nb_auc, 4)
    summary_rows.append(srow)
df_dca_summary = pd.DataFrame(summary_rows) \
                   .sort_values('NB_AUC(0.01-0.99)', ascending=False) \
                   .reset_index(drop=True)
save_csv(df_dca_summary, 'DCA阈值模型摘要表_测试集.csv')

# ─── (C) 美观热力表 PNG ─────────────────────────────────
# 模型按"关键阈值上的平均 NB"降序排列
_model_score = []
for name in model_order:
    nbs = [calc_nb(y_test.values, test_probas[name], t) for t in key_thresholds]
    _model_score.append((name, float(np.mean(nbs))))
_model_score.sort(key=lambda x: x[1], reverse=True)
sorted_models = [m for m, _ in _model_score]

row_labels = [MODEL_FULL_NAME[m] for m in sorted_models] + ['Treat All', 'Treat None']
n_rows = len(row_labels); n_cols = len(key_thresholds)

matrix = np.zeros((n_rows, n_cols))
for i, name in enumerate(sorted_models):
    for j, t in enumerate(key_thresholds):
        matrix[i, j] = calc_nb(y_test.values, test_probas[name], t)
for j, t in enumerate(key_thresholds):
    matrix[n_rows-2, j] = prev_test - (1-prev_test)*t/(1-t)
matrix[n_rows-1, :] = 0.0

# 颜色：0 居中（红→白→绿）
absmax = max(abs(matrix.min()), abs(matrix.max()), 1e-6)
norm_h = mcolors.TwoSlopeNorm(vmin=-absmax, vcenter=0, vmax=absmax)
cmap_h = plt.cm.RdYlGn

fig, ax = plt.subplots(figsize=(14, 8.5), facecolor='white')
im = ax.imshow(matrix, cmap=cmap_h, norm=norm_h, aspect='auto')

# 单元格数值（颜色根据底色亮度自动黑/白）
for i in range(n_rows):
    for j in range(n_cols):
        v   = matrix[i, j]
        rgb = cmap_h(norm_h(v))[:3]
        lightness = 0.299*rgb[0] + 0.587*rgb[1] + 0.114*rgb[2]
        tc = 'black' if lightness > 0.55 else 'white'
        ax.text(j, i, f"{v:+.3f}",
                ha='center', va='center',
                fontsize=13, fontweight='bold', color=tc)

# 每列最佳策略（金边，排除 Treat None）
for j in range(n_cols):
    best_i = int(matrix[:n_rows-1, j].argmax())
    ax.add_patch(plt.Rectangle((j-0.48, best_i-0.48), 0.96, 0.96,
                               fill=False, edgecolor='gold', lw=3, zorder=5))

# 行标签：默认策略两行用灰斜体
ax.set_yticks(range(n_rows))
ax.set_yticklabels(row_labels, fontsize=13, fontweight='bold')
for i, lab in enumerate(ax.get_yticklabels()):
    if row_labels[i] in ['Treat All', 'Treat None']:
        lab.set_color('#666666'); lab.set_style('italic')

# 列标签：阈值 + FP:TP 代价比
ax.set_xticks(range(n_cols))
ax.set_xticklabels([f"{int(t*100)}%\n({r})" for t, r in zip(key_thresholds, ratio_labels)],
                   fontsize=13, fontweight='bold')

ax.set_xlabel('Threshold Probability   (FP : TP cost ratio acceptable to clinician)',
              fontsize=16, fontweight='bold', labelpad=10)
ax.set_ylabel('Model / Default strategy', fontsize=16, fontweight='bold', labelpad=10)
# 标题已按要求删除

# 白色网格让单元格边界分明
ax.set_xticks(np.arange(n_cols+1)-0.5, minor=True)
ax.set_yticks(np.arange(n_rows+1)-0.5, minor=True)
ax.grid(which='minor', color='white', linewidth=2)
ax.tick_params(which='minor', length=0)

# colorbar
cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
cbar.set_label('Net Benefit', fontsize=15, fontweight='bold')
cbar.ax.tick_params(labelsize=10)

plt.tight_layout()
save_fig('DCA阈值净获益热力表_测试集.png'); plt.close()
del matrix

# ============================================================
# 图8: 混淆矩阵
# ============================================================
n_models=len(model_order)
n_cols=5; n_rows=(n_models+n_cols-1)//n_cols
fig,axes=plt.subplots(n_rows,n_cols,figsize=(22,4.5*n_rows),facecolor='white')
axes=np.atleast_2d(axes).flatten()
for idx,name in enumerate(model_order):
    ax=axes[idx]
    yp=(test_probas[name]>=optimal_thresholds[name]).astype(int)
    cm=confusion_matrix(y_test,yp)
    ax.imshow(cm,cmap='Blues',aspect='auto')
    for i in range(2):
        for j in range(2):
            clr='white' if cm[i,j]>cm.max()/2 else 'black'
            ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=22,fontweight='bold',color=clr)
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(['Negative','Positive'],fontsize=16,fontweight='bold')
    ax.set_yticklabels(['Negative','Positive'],fontsize=16,fontweight='bold')
    ax.set_xlabel('Predicted',fontsize=17,fontweight='bold'); ax.set_ylabel('Actual',fontsize=17,fontweight='bold')
    s=test_results[name]['Sensitivity']; sp=test_results[name]['Specificity']
    # 标题已按要求删除
for idx in range(len(model_order),len(axes)): axes[idx].axis('off')
# 标题已按要求删除
plt.tight_layout(); save_fig('混淆矩阵_8模型.png'); plt.close()

# ============================================================
# 图9: 学习曲线（使用校准前基模型 raw_models 做诊断，更快）
# ============================================================
fig,axes=plt.subplots(n_rows,n_cols,figsize=(22,4.5*n_rows),facecolor='white')
axes=np.atleast_2d(axes).flatten()
for idx,name in enumerate(model_order):
    ax=axes[idx]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tr_sz,tr_sc,cv_sc=learning_curve(
                raw_models[name],X_train_arr,y_train,
                cv=5, n_jobs=GLOBAL_N_JOBS,
                train_sizes=np.linspace(0.1,1.0,10),scoring='roc_auc')
        tr_m,tr_s=np.mean(tr_sc,1),np.std(tr_sc,1)
        cv_m,cv_s=np.mean(cv_sc,1),np.std(cv_sc,1)
        ax.fill_between(tr_sz,tr_m-tr_s,tr_m+tr_s,alpha=0.1,color='steelblue')
        ax.fill_between(tr_sz,cv_m-cv_s,cv_m+cv_s,alpha=0.1,color='darkorange')
        ax.plot(tr_sz,tr_m,'o-',color='steelblue',lw=2,label='Training')
        ax.plot(tr_sz,cv_m,'o-',color='darkorange',lw=2,label='CV')
        ax.set_ylim([0.5,1.02])
    except Exception as err:
        ax.text(0.5,0.5,f'Error:{str(err)[:20]}',ha='center',va='center',
                transform=ax.transAxes,fontsize=15,fontweight='bold')
    # 标题已按要求删除
    ax.set_xlabel('Training Samples',fontsize=16,fontweight='bold'); ax.set_ylabel('AUC',fontsize=16,fontweight='bold')
    ax.legend(loc='lower right',fontsize=13,prop={'weight':'bold','size':13}); ax.grid(True,alpha=0.3)
for idx in range(len(model_order),len(axes)): axes[idx].axis('off')
# 标题已按要求删除
plt.tight_layout(); save_fig('学习曲线_过拟合诊断.png'); plt.close()

# ============================================================
# 图10: AUC对比柱状图
# ============================================================
fig,ax=plt.subplots(figsize=(14,7),facecolor='white')
x=np.arange(len(model_order)); w=0.35
tr_aucs=[train_results[n]['AUC'] for n in model_order]
te_aucs=[test_results[n]['AUC'] for n in model_order]
bars1=ax.bar(x-w/2,tr_aucs,w,label='Training Set',color='#3498DB',alpha=0.85)
bars2=ax.bar(x+w/2,te_aucs,w,label='Test Set',color='#E74C3C',alpha=0.85)
for bar,val in zip(bars1,tr_aucs):
    ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.005,
            f'{val:.3f}',ha='center',va='bottom',fontsize=13,fontweight='bold')
for bar,val in zip(bars2,te_aucs):
    ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.005,
            f'{val:.3f}',ha='center',va='bottom',fontsize=13,fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([get_label(n) for n in model_order],rotation=25,ha='right',fontsize=16,fontweight='bold')
ax.set_ylabel('AUC',fontsize=22,fontweight='bold')
# 标题已按要求删除
ax.legend(fontsize=17,prop={'weight':'bold','size':17}); ax.set_ylim([0.5,1.08]); ax.grid(True,alpha=0.3,axis='y')
plt.tight_layout(); save_fig('AUC对比_训练集与测试集.png'); plt.close()

# ============================================================
# 图11: 雷达图
# ============================================================
radar_metrics=['AUC','Sensitivity','Specificity','Precision','F1','MCC']
N=len(radar_metrics); angles=np.linspace(0,2*np.pi,N,endpoint=False).tolist(); angles+=angles[:1]
fig,ax=plt.subplots(figsize=(10,10),subplot_kw=dict(polar=True),facecolor='white')
for name in model_order:
    vals=[max(0,test_results[name].get(m,0)) for m in radar_metrics]; vals+=vals[:1]
    ax.plot(angles,vals,'o-',lw=2,label=get_label(name),color=get_color(name))
    ax.fill(angles,vals,alpha=0.06,color=get_color(name))
ax.set_xticks(angles[:-1]); ax.set_xticklabels(radar_metrics,fontsize=17,fontweight='bold')
ax.set_ylim(0,1)
# 标题已按要求删除
ax.legend(loc='upper left',bbox_to_anchor=(1.15,1.05),fontsize=15,prop={'weight':'bold','size':15})
ax.grid(True,alpha=0.3); plt.tight_layout()
save_fig('性能雷达图_8模型.png'); plt.close()

# ============================================================
# 图12: 热力图
# ============================================================
hm_metrics=['AUC','Sensitivity','Specificity','Precision','NPV','F1','MCC','Accuracy','Brier Score']
hm_data=np.array([[test_results[n].get(m,0) for m in hm_metrics] for n in model_order])
fig,ax=plt.subplots(figsize=(14,8),facecolor='white')
im=ax.imshow(hm_data,cmap='RdYlGn',aspect='auto',vmin=0,vmax=1)
ax.set_xticks(range(len(hm_metrics)))
ax.set_xticklabels(hm_metrics,fontsize=17,rotation=30,ha='right',fontweight='bold')
ax.set_yticks(range(len(model_order)))
ax.set_yticklabels([get_label(n) for n in model_order],fontsize=17,fontweight='bold')
for i in range(len(model_order)):
    for j in range(len(hm_metrics)):
        val=hm_data[i,j]; clr='black' if 0.3<val<0.75 else 'white'
        ax.text(j,i,f'{val:.3f}',ha='center',va='center',fontsize=15,color=clr,fontweight='bold')
cbar=plt.colorbar(im,ax=ax,label='Value (lower Brier = better)')
cbar.ax.yaxis.label.set_fontweight('bold')
cbar.ax.yaxis.label.set_fontsize(14)
cbar.ax.tick_params(labelsize=12)
for t in cbar.ax.get_yticklabels(): t.set_fontweight('bold')
# 标题已按要求删除
plt.tight_layout(); save_fig('综合指标热力图.png'); plt.close()

# ============================================================
# 图13: AUC排名+CI
# ============================================================
fig,ax=plt.subplots(figsize=(10,7),facecolor='white')
for i,name in enumerate(model_order):
    am=boot_test[name]['AUC']['mean']
    al=boot_test[name]['AUC']['ci_lower']
    ah=boot_test[name]['AUC']['ci_upper']
    ax.barh(i,am,color=get_color(name),alpha=0.8,height=0.6)
    ax.plot([al,ah],[i,i],'k|-',lw=2,ms=8)
    ax.text(ah+0.003,i,f"{am:.3f} ({al:.3f}-{ah:.3f})",va='center',fontsize=15,fontweight='bold')
ax.set_yticks(range(len(model_order)))
ax.set_yticklabels([get_label(n) for n in model_order],fontsize=17,fontweight='bold')
ax.set_xlabel('AUC (95% CI)',fontsize=22,fontweight='bold')
# 标题已按要求删除
ax.set_xlim([0.5,1.08]); ax.grid(True,alpha=0.3,axis='x')
plt.tight_layout(); save_fig('AUC排名与置信区间.png'); plt.close()

# ============================================================
# 图14 / 图15 / 图16 — 完全按照独立脚本「机器学习roc曲线与雷达图代码.py」的样式
# ============================================================
# 为了让这 3 张图完全对齐独立脚本的视觉效果（轴线粗细、刻度线等），
# 在 rc_context 内临时恢复独立脚本的 rcParams：
#   · 只保留字体加粗（font.weight / axes.labelweight / axes.titleweight = bold）
#   · 其他轴线 / 刻度线的 width 回到 matplotlib 默认值
# 离开 with 块后，主脚本的全局 rcParams 自动恢复，不影响后续代码。
_ROC_RADAR_RC = {
    'font.sans-serif'     : ['Arial', 'DejaVu Sans', 'SimHei'],
    'axes.unicode_minus'  : False,
    'font.weight'         : 'bold',
    'axes.labelweight'    : 'bold',
    'axes.titleweight'    : 'bold',
    # 以下回到 matplotlib 默认，避免继承主脚本加粗放大的 rcParams
    'font.size'           : 14.0,
    'axes.labelsize'      : 16,
    'axes.titlesize'      : 'large',
    'xtick.labelsize'     : 14,
    'ytick.labelsize'     : 14,
    'legend.fontsize'     : 16,
    'figure.titlesize'    : 'large',
    'figure.titleweight'  : 'normal',
    'axes.linewidth'      : 0.8,
    'xtick.major.width'   : 0.8,
    'ytick.major.width'   : 0.8,
    'xtick.major.size'    : 3.5,
    'ytick.major.size'    : 3.5,
}

# ============================================================
# 图14 / 图15: ROC曲线 + 右侧 AUC 彩带（测试集 / 训练集）
# ============================================================
def plot_roc_with_auc_bar(y_true, probas_dict, title, save_name,
                          metrics_csv_path=None):
    """
    左侧 ROC 曲线 + 右侧模型色块 + AUC 彩色条带（Blues 渐变）
    按 AUC 降序排列。

    ┌─────────────────────────────────────────────────────────────────┐
    │ 关键修改：当传入 metrics_csv_path 时，右侧彩带显示的 AUC 数字     │
    │ 与排序均直接从 CSV（bootstrap 均值表）读取，确保与结果表完全一致。│
    │ ROC 曲线形状仍由真实概率绘出（不变）。                            │
    └─────────────────────────────────────────────────────────────────┘
    """
    model_order = list(probas_dict.keys())

    # ── 若指定了 CSV，则建立 {内部key: bootstrap均值AUC} 映射 ──
    auc_from_csv = None
    if metrics_csv_path is not None and os.path.exists(metrics_csv_path):
        df_m = pd.read_csv(metrics_csv_path, encoding='utf-8-sig')
        df_m.columns = df_m.columns.str.replace('\ufeff', '').str.strip()

        def _strip_ci(s):
            m = re.match(r'\s*([\d.]+)', str(s))
            return float(m.group(1)) if m else np.nan

        # CSV 用全名（如 "Random Forest"），probas_dict 用缩写键（如 "RF"），
        # 借助 MODEL_FULL_NAME 反查
        full_to_auc = {row['Model']: _strip_ci(row['AUC'])
                       for _, row in df_m.iterrows()}
        auc_from_csv = {}
        for k in model_order:
            full = MODEL_FULL_NAME.get(k, k)
            if full in full_to_auc and not np.isnan(full_to_auc[full]):
                auc_from_csv[k] = full_to_auc[full]
        print(f"  📋 [{save_name}] AUC 数字与排序读取自 CSV "
              f"({len(auc_from_csv)}/{len(model_order)} 模型匹配成功)")

    roc_data = []
    for name in model_order:
        fpr, tpr, _ = roc_curve(y_true, probas_dict[name])
        # 优先使用 CSV 里的 bootstrap 均值 AUC（与结果表对齐）
        if auc_from_csv is not None and name in auc_from_csv:
            auc_val = auc_from_csv[name]
        else:
            auc_val = roc_auc_score(y_true, probas_dict[name])
        roc_data.append({
            'name': name, 'fpr': fpr, 'tpr': tpr,
            'auc':  auc_val,
        })
    roc_sorted = sorted(roc_data, key=lambda d: d['auc'], reverse=True)

    fig = plt.figure(figsize=(14, 8.5), facecolor='white')
    gs  = gridspec.GridSpec(1, 2, width_ratios=[4, 1], wspace=0.10)

    # ─── 左侧 ROC ───
    ax1 = plt.subplot(gs[0])
    for rd in roc_data:
        ax1.plot(rd['fpr'], rd['tpr'],
                 lw=2.5, color=get_color(rd['name']),
                 label=get_label(rd['name']))
    ax1.plot([0, 1], [0, 1], 'r--', lw=1.5, alpha=0.8)
    ax1.set_xlim([-0.02, 1.02]); ax1.set_ylim([-0.02, 1.02])
    ax1.set_xlabel("False Positive Rate (1 - Specificity)",
                   fontsize=ROC_FS_AXIS_LABEL, fontweight='bold')
    ax1.set_ylabel("True Positive Rate (Sensitivity)",
                   fontsize=ROC_FS_AXIS_LABEL, fontweight='bold')
    # 标题已按要求删除
    ax1.legend(loc='lower right', fontsize=ROC_FS_LEGEND, frameon=False, prop={'weight': 'bold', 'size': ROC_FS_LEGEND})
    ax1.tick_params(axis='both', labelsize=ROC_FS_TICK)
    for lb in ax1.get_xticklabels() + ax1.get_yticklabels():
        lb.set_fontweight('bold')

    # ─── 右侧 AUC 彩带 ───
    ax2 = plt.subplot(gs[1]); ax2.axis('off')
    n_m  = len(roc_sorted)
    aucs = [d['auc'] for d in roc_sorted]
    norm = Normalize(vmin=min(aucs) - 1e-6, vmax=max(aucs) + 1e-6)
    cmap_auc = plt.cm.Blues

    bar_h, gap = 0.85, 0.10
    for rank, rd in enumerate(roc_sorted):
        y_center = n_m - 1 - rank
        y_bot    = y_center - bar_h / 2

        # 左色块：模型颜色
        ax2.add_patch(plt.Rectangle((0.0, y_bot), 0.45, bar_h,
                                    facecolor=get_color(rd['name']),
                                    edgecolor='white', linewidth=1.5))
        # 右色块：Blues 渐变
        auc_color = cmap_auc(norm(rd['auc']))
        ax2.add_patch(plt.Rectangle((0.50, y_bot), 0.50, bar_h,
                                    facecolor=auc_color,
                                    edgecolor='white', linewidth=1.5))
        # AUC 数字：深底白字，浅底黑字
        lightness = (auc_color[0]*299 + auc_color[1]*587 + auc_color[2]*114) / 1000
        txt_color = 'white' if lightness < 0.55 else '#1C1C1C'
        ax2.text(0.75, y_center, f"{rd['auc']:.3f}",
                 ha='center', va='center',
                 fontsize=ROC_FS_AUC_VALUE, fontweight='bold',
                 color=txt_color)

    ax2.set_xlim(-0.02, 1.02)
    ax2.set_ylim(-0.5, n_m - 0.5 + gap)

    plt.tight_layout()
    save_fig(save_name); plt.close()

# —— 图14 / 图15 / 图16: 在 rc_context 内绘制（样式完全对齐独立脚本）——
with plt.rc_context(_ROC_RADAR_RC):
    # 图14: 测试集 ROC+AUC 彩带
    # ★ 传入测试集 CSV 路径，使右侧彩带 AUC 数字与排序与结果表完全一致
    plot_roc_with_auc_bar(
        y_test.values, test_probas,
        title='ROC Curve — Test Set Model Comparison',
        save_name='ROC曲线_AUC彩带_测试集.png',
        metrics_csv_path=os.path.join(output_path,
                                      '模型评估_测试集_95置信区间.csv'))

    # 图15: 训练集 ROC+AUC 彩带
    # ★ 传入训练集 CSV 路径，使右侧彩带 AUC 数字与排序与结果表完全一致
    plot_roc_with_auc_bar(
        y_train.values, train_probas,
        title='ROC Curve — Training Set Model Comparison',
        save_name='ROC曲线_AUC彩带_训练集.png',
        metrics_csv_path=os.path.join(output_path,
                                      '模型评估_训练集_95置信区间.csv'))

# ============================================================
# 图16: 圆盘放射图（8 模型 × 5 指标）
# —— 函数体与独立脚本「机器学习roc曲线与雷达图代码.py」完全一致 ——
# ============================================================
def plot_disk_radial_chart(metrics_csv_path, save_name):
    """
    从 "模型评估_测试集_95置信区间.csv" 读取 8 模型 × 5 指标，
    绘制圆盘放射图（按 AUC 降序顺时针排列）
    """
    df = pd.read_csv(metrics_csv_path, encoding='utf-8-sig')
    df.columns = df.columns.str.replace('\ufeff', '').str.strip()

    # ★ 仅保留 ALLOWED_MODELS 白名单内的行
    before = len(df)
    df = df[df['Model'].isin(ALLOWED_MODELS)].reset_index(drop=True)
    after = len(df)
    if before != after:
        skipped = [m for m in df['Model'].unique() if m not in ALLOWED_MODELS]
        dropped_models = set(pd.read_csv(metrics_csv_path, encoding='utf-8-sig')
                             ['Model'].unique()) - ALLOWED_MODELS
        print(f"   🚫 已过滤非白名单模型: {sorted(dropped_models)}")
    print(f"   保留 {after} 个模型")

    def _strip_ci(s):
        m = re.match(r'\s*([\d.]+)', str(s))
        return float(m.group(1)) if m else np.nan

    df['_auc_val'] = df['AUC'].apply(_strip_ci)
    df = df.sort_values('_auc_val', ascending=False).reset_index(drop=True)

    model_labels = df['Model'].tolist()
    model_abbrs  = [MODEL_ABBR.get(m, m) for m in model_labels]
    n_models     = len(model_labels)

    data = {m: df[CSV_COL_MAP[m]].apply(_strip_ci).values for m in METRIC_ORDER}
    print(f"  模型排序（AUC 降序）: {model_abbrs}")

    # 几何
    R_INNER, R_OUTER, R_RING_LABEL = 0.30, 1.00, 1.03
    BAR_GAP_FRAC = 0.15
    GRID_VALUES  = [0.2, 0.4, 0.6, 0.8, 1.0]
    BG_COLOR     = '#FAFAFA'

    fig = plt.figure(figsize=(16, 16), facecolor='white')
    ax  = fig.add_subplot(111, projection='polar')
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    sector_span   = 2 * np.pi / n_models
    inner_pad_rad = np.deg2rad(5.0)
    bar_region    = sector_span - 2 * inner_pad_rad
    n_metrics     = len(METRIC_ORDER)
    bar_width     = bar_region / n_metrics * (1 - BAR_GAP_FRAC)
    bar_step      = bar_region / n_metrics

    def _v_to_r(v):
        v = max(0.0, min(1.0, float(v)))
        return R_INNER + v * (R_OUTER - R_INNER)

    # 1) 统一浅灰底（避免相邻扇区 bar 抗锯齿伪线 → 中央伪竖线）
    _bg = np.linspace(0, 2*np.pi, 361)
    ax.fill(_bg, [R_OUTER]*361, color=BG_COLOR, edgecolor='none', zorder=0)

    # 2) 【已去除】内部同心刻度圈
    # 原先在 0.2/0.4/0.6/0.8/1.0 处都画了一圈，现在只保留最外圈（R_OUTER 那条），
    # 其他四圈删掉。最外圈会由下面"扇区外缘包边"自然形成，这里不再单独画。

    # 2b) 每扇区外缘弧形实线包边（参考图视觉特征：每组柱外面一条弧带把它们包起来）
    for _m in range(n_models):
        _ts = _m * sector_span + inner_pad_rad
        _te = (_m + 1) * sector_span - inner_pad_rad
        _arc = np.linspace(_ts, _te, 60)
        ax.plot(_arc, [R_OUTER]*60, color='#888888', lw=1.0, zorder=2)
        # 弧带两端沿径向向内小段，形成"包住柱"的效果
        ax.plot([_ts, _ts], [R_INNER, R_OUTER], color='#888888', lw=1.0, zorder=2)
        ax.plot([_te, _te], [R_INNER, R_OUTER], color='#888888', lw=1.0, zorder=2)

    # 3) 每扇区 5 根径向柱 + 柱顶数字
    for m_idx in range(n_models):
        theta_start = m_idx * sector_span + inner_pad_rad
        for b_idx, metric in enumerate(METRIC_ORDER):
            val = data[metric][m_idx]
            tc  = theta_start + (b_idx + 0.5) * bar_step
            r_top = _v_to_r(val)
            ax.bar(tc, r_top - R_INNER,
                   width=bar_width, bottom=R_INNER,
                   color=METRIC_COLORS[metric],
                   edgecolor='white', linewidth=0.8, zorder=3)
            text_r = r_top - 0.05 if (r_top - R_INNER) > 0.15 else r_top + 0.04
            td = np.rad2deg(tc)
            rd = -td if not (90 < (td % 360) < 270) else -td + 180
            ax.text(tc, text_r, f"{val:.3f}",
                    ha='center', va='center',
                    rotation=rd, rotation_mode='anchor',
                    fontsize=DISK_FS_BAR_VALUE, fontweight='bold',
                    color='#1C1C1C', zorder=5)

        # 4) 扇区左边缘 0.2/.../1.0 刻度
        tick_theta = m_idx * sector_span + 0.02
        for gv in GRID_VALUES:
            r = _v_to_r(gv)
            td = np.rad2deg(tick_theta)
            rt = -td if not (90 < (td % 360) < 270) else -td + 180
            ax.text(tick_theta, r, f"{gv:.1f}",
                    ha='left', va='center',
                    fontsize=DISK_FS_GRID_TICK, fontweight='bold',
                    color='#555555',
                    rotation=rt, rotation_mode='anchor', zorder=4)

    # 5) 外圈指标名
    for m_idx in range(n_models):
        theta_start = m_idx * sector_span + inner_pad_rad
        for b_idx, metric in enumerate(METRIC_ORDER):
            tc = theta_start + (b_idx + 0.5) * bar_step
            td = np.rad2deg(tc)
            rd = -td if not (90 < (td % 360) < 270) else -td + 180
            ax.text(tc, R_RING_LABEL, metric,
                    ha='center', va='center',
                    rotation=rd, rotation_mode='anchor',
                    fontsize=DISK_FS_METRIC_LABEL, fontweight='bold',
                    color='#1C1C1C', zorder=5)

    # 6) 【不画】扇区分隔径向线（这是中央竖线的主要肇因，已彻底移除）

    # 7) 中央白圆（用 fill + plot，避免 ax.bar(width=2π) 留接缝）
    _cc = np.linspace(0, 2*np.pi, 361)
    ax.fill(_cc, [R_INNER]*361, color='white', edgecolor='none', zorder=6)
    ax.plot(_cc, [R_INNER]*361, color='#BBBBBB', lw=1.0, zorder=6.5)

    # 8) 模型简写
    for m_idx in range(n_models):
        tcs = m_idx * sector_span + sector_span / 2
        td = np.rad2deg(tcs)
        rd = -td if not (90 < (td % 360) < 270) else -td + 180
        ax.text(tcs, R_INNER * 0.60, model_abbrs[m_idx],
                ha='center', va='center',
                rotation=rd, rotation_mode='anchor',
                fontsize=DISK_FS_MODEL_ABBR, fontweight='bold',
                color='#1C1C1C', zorder=8)

    # 9) 美化
    ax.set_ylim(0, R_RING_LABEL + 0.10)
    ax.set_yticks([]); ax.set_xticks([])
    ax.spines['polar'].set_visible(False)
    ax.grid(False)

    # 10) 底部图例
    handles = [mpatches.Patch(facecolor=METRIC_COLORS[m], edgecolor='white', label=m)
               for m in METRIC_ORDER]
    fig.legend(handles=handles, loc='lower center', ncol=len(METRIC_ORDER),
               frameon=False, fontsize=DISK_FS_LEGEND,
               bbox_to_anchor=(0.5, 0.035),
               prop={'weight': 'bold', 'size': DISK_FS_LEGEND},
               handlelength=2.2, handleheight=1.2,
               columnspacing=1.8, labelspacing=0.8)

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    save_fig(save_name); plt.close()

# —— 图16: 圆盘放射图（同在 rc_context 内绘制）——
# 数据源: 主脚本前面已保存的 "模型评估_测试集_95置信区间.csv"
with plt.rc_context(_ROC_RADAR_RC):
    plot_disk_radial_chart(
        metrics_csv_path=os.path.join(output_path, '模型评估_测试集_95置信区间.csv'),
        save_name='圆盘放射图_8模型5指标.png')

# —— 图17: 训练集圆盘放射图（与图16同一字体/配色/几何，数据源改为训练集 CSV）——
# 数据源: 主脚本前面已保存的 "模型评估_训练集_95置信区间.csv"
with plt.rc_context(_ROC_RADAR_RC):
    plot_disk_radial_chart(
        metrics_csv_path=os.path.join(output_path, '模型评估_训练集_95置信区间.csv'),
        save_name='圆盘放射图_8模型5指标_训练集.png')

# ============================================================
# 最终汇总
# ============================================================
print(f"\n{'='*70}\n✅ All tasks complete!\n{'='*70}")
best_name=model_order[0]
print(f"\n🏆 Best Model (Test AUC, Calibrated): {get_label(best_name)}")
print(f"   Train AUC: {boot_train[best_name]['AUC']['ci_str']}")
print(f"   Test  AUC: {boot_test[best_name]['AUC']['ci_str']}")
print(f"   Gap: {train_results[best_name]['AUC']-test_results[best_name]['AUC']:+.4f}")

print(f"\n{'Rank':<5} {'Model':<38} {'Train AUC':<10} {'Test AUC':<10} {'Gap':<8} Status")
print("─"*78)
for i,name in enumerate(model_order):
    ta=train_results[name]['AUC']; ea=test_results[name]['AUC']; gap=ta-ea
    s='✓' if abs(gap)<0.05 else ('⚡' if gap<0.10 else '⚠')
    print(f"{i+1:<5} {get_label(name):<38} {ta:.4f}     {ea:.4f}     {gap:+.4f}  {s}")

print(f"\n📐 校准指标（测试集，校准后）:")
print(f"{'Model':<38}{'Brier':>10}{'Intercept':>12}{'Slope':>10}")
print("─"*70)
for name in model_order:
    cm = calib_metrics[name]['test_cal']
    br = brier_score_loss(y_test, test_probas[name])
    print(f"{get_label(name):<38}{br:>10.4f}{cm['intercept']:>+12.4f}{cm['slope']:>10.4f}")

print(f"\n📐 Platt Scaling 参数 (A, B):")
print(f"{'Model':<38}{'A (mean±std)':>22}{'B (mean±std)':>22}")
print("─"*82)
for name in model_order:
    pp = platt_params[name]
    if not np.isnan(pp['A_mean']):
        print(f"{get_label(name):<38}"
              f"{pp['A_mean']:>+12.3f} ±{pp['A_std']:>6.3f}"
              f"{pp['B_mean']:>+14.3f} ±{pp['B_std']:>6.3f}")
    else:
        print(f"{get_label(name):<38}{'N/A':>22}{'N/A':>22}")

print(f"\n📁 Output: {output_path}")
print(f"{'='*70}\n  Analysis complete!\n{'='*70}")
