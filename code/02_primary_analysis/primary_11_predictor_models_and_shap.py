# -*- coding: utf-8 -*-
# v6: Figure 2B/测试集ROC-AUC彩带改为直接显示当前校准后预测概率计算的AUC点估计（4位小数），不再读取bootstrap均值；其余建模、统计与SHAP流程不变。
# v5: AUC彩带最低色阶增强（最低值不再与白底融合）；其余建模、统计与SHAP流程保持v4不变。
"""
Created on Sat Aug  8 22:25:38 2026

@author: Administrator
"""

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
肺部感染预测 - 机器学习分类分析（8模型 · 11特征 · Platt校准 + 严格cross-fit阈值 + SHAP完整版）
======================================================================
  · 数据路径 : E:\新建文件夹\第三次修稿
  · 文件名   : BSAfree_feature_selection_input_exactsplit.csv
  · 输出目录 : E:\新建文件夹\第三次修稿\主分析11变量_八模型完整重跑_SHAP
  · 结局     : Pulmonary_infection
  · 预测变量（11个，审稿后主分析重筛）:
      NEU(NEUT_abs), Intubation(Intubation_tracheotomy), MV(Mechanical_ventilation),
      LDH, LYM(LYMPH_abs), BUN, CCI, FIB, Surgery, Diuretics, TCO2(CO2)

====== 校准方法学要点（与 ICH 高费用识别模型一致）======
  ▪ 贝叶斯超参数优化：10 折分层 CV
  ▪ Platt Scaling 概率校准：10 折 CV（与调参同口径，避免信息泄漏）
  ▪ Youden 最优阈值：基于严格 outer-10-fold cross-fitted 校准训练概率确定
  ▪ 参数冻结：超参数与阈值均在训练阶段确定；最终模型仅在Train拟合
  ▪ 测试集：沿用原主分析固定30% Test身份（n=1011, events=407），不重新随机划分
  ▪ 校准评估：Brier + 校准截距 + 校准斜率（理想 0 / 0 / 1）+ HL 检验
  ▪ 输出：完整性能/校准/DCA/ROC/PR/混淆矩阵/学习曲线等原始图表 + GBDT SHAP整套图表

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
from sklearn.base import clone
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

# ★★★ 路径：审稿后 11 变量主分析 ★★★
# 输入文件必须保留原始主分析固定 Train/Test 身份（Primary_split）。
data_path   = r"E:\新建文件夹\第三次修稿"
data_file   = os.path.join(data_path, "BSAfree_feature_selection_input_exactsplit.csv")
output_path = os.path.join(data_path, "主分析11变量_八模型完整重跑_SHAP")
ckpt_dir    = os.path.join(output_path, "checkpoints")
os.makedirs(output_path, exist_ok=True)
os.makedirs(ckpt_dir, exist_ok=True)

# ★★★ 结局 + 审稿后重新筛选得到的 11 个主分析预测变量 ★★★
OUTCOME = 'Pulmonary_infection'
ID_COL = 'Study_row_id'
SPLIT_COL = 'Primary_split'
FEATURES_USED = [
    'NEUT_abs',                     # NEU
    'Intubation_tracheotomy',       # Intubation
    'Mechanical_ventilation',       # MV
    'LDH',
    'LYMPH_abs',                    # LYM
    'BUN',
    'CCI',
    'FIB',
    'Surgery',
    'Diuretics',
    'CO2',                          # TCO2 in manuscript / R output
]

# SHAP模型选择
# 默认固定使用用户原始SHAP脚本指定的GBDT。
# 这样SHAP解释对象不会因为一次重新调参的细小CV排序变化而自动切换；
# 当前程序仍保留其他选择模式，必要时可手动切换。
# 可选模式：
#   'best_tree_train_cv'  : 推荐；训练期CV中最佳树模型
#   'best_tree_test_auc'  : Test观察AUC最高的树模型（post hoc，仅描述性）
#   'best_overall_test_auc': Test观察AUC最高的任意模型；非树模型会用KernelExplainer，且无TreeSHAP交互矩阵
#   'fixed'               : 固定使用SHAP_FIXED_MODEL_KEY
RUN_SHAP = True
SHAP_SELECTION_MODE = 'fixed'
SHAP_FIXED_MODEL_KEY = 'GBDT'
SHAP_TREE_MODELS = ['RF', 'GBDT', 'DT', 'LightGBM', 'XGBoost']
RUN_SHAP_INTERACTIONS = True

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
    'NEUT_abs':                   'Absolute Neutrophil Count (NEU)',
    'Intubation_tracheotomy':     'Intubation/Tracheotomy',
    'Mechanical_ventilation':     'Mechanical Ventilation (MV)',
    'LDH':                        'Lactate Dehydrogenase (LDH)',
    'LYMPH_abs':                  'Absolute Lymphocyte Count (LYM)',
    'BUN':                        'Blood Urea Nitrogen (BUN)',
    'CCI':                        'Charlson Comorbidity Index (CCI)',
    'FIB':                        'Fibrinogen (FIB)',
    'Surgery':                    'Surgery',
    'Diuretics':                  'Diuretics',
    'CO2':                        'Total Carbon Dioxide (TCO2)',
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
    """Youden 指数最优阈值，基于校准后 OOF 预测概率确定"""
    yp=model.predict_proba(X)[:,1]; best_s,best_t=-np.inf,0.5
    for t in np.arange(0.05,0.95,0.005):
        yc=(yp>=t).astype(int); cm=confusion_matrix(y,yc)
        if cm.shape==(2,2):
            tn,fp,fn,tp=cm.ravel()
            s=(tp/(tp+fn)+tn/(tn+fp)-1) if ((tp+fn) and (tn+fp)) else 0
        else: s=0
        if s>best_s: best_s,best_t=s,t
    return best_t



def choose_youden_threshold_from_probabilities(y_true, y_proba):
    """从真正的cross-fitted训练概率中选择Youden最大阈值。"""
    fpr, tpr, thresholds = roc_curve(np.asarray(y_true), np.asarray(y_proba))
    valid = np.isfinite(thresholds)
    fpr, tpr, thresholds = fpr[valid], tpr[valid], thresholds[valid]
    j = tpr - fpr
    idx = int(np.argmax(j))
    return float(thresholds[idx]), float(j[idx])


def strict_cross_fitted_train_probabilities(train_raw_df, y_train_series,
                                             feature_names, continuous_names,
                                             base_estimator,
                                             outer_cv=10, calibration_cv=10,
                                             seed=42):
    """
    严格outer cross-fitting，仅用于分类阈值估计：
    - 每个outer-training fold重新拟合连续变量MinMaxScaler；
    - 超参数固定为完整Train中BayesSearchCV得到的值，不在outer fold内重调；
    - outer-training内部再做Platt calibration；
    - 每名训练患者只获得一次来自未见过该患者的outer-validation概率。

    该步骤不是完整nested validation；完整重复nested分析在独立修稿脚本中完成。
    """
    y_arr = np.asarray(y_train_series).astype(int)
    outer = StratifiedKFold(n_splits=outer_cv, shuffle=True, random_state=seed)
    oof = np.full(len(train_raw_df), np.nan, dtype=float)
    fold_rows = []

    for fold, (tr_idx, va_idx) in enumerate(outer.split(train_raw_df, y_arr), start=1):
        tr_df = train_raw_df.iloc[tr_idx].copy()
        va_df = train_raw_df.iloc[va_idx].copy()
        y_tr = y_arr[tr_idx]
        y_va = y_arr[va_idx]

        Xtr = tr_df[feature_names].copy()
        Xva = va_df[feature_names].copy()
        scaler_fold = MinMaxScaler()
        if continuous_names:
            Xtr.loc[:, continuous_names] = scaler_fold.fit_transform(Xtr[continuous_names])
            Xva.loc[:, continuous_names] = scaler_fold.transform(Xva[continuous_names])

        est = clone(base_estimator)
        cal = calibrate_model(est, Xtr.values, y_tr,
                              method=CALIBRATION_METHOD,
                              cv=calibration_cv)
        pva = cal.predict_proba(Xva.values)[:, 1]
        oof[va_idx] = pva
        fold_rows.append({
            'Outer_fold': fold,
            'Outer_train_n': int(len(tr_idx)),
            'Outer_train_events': int(y_tr.sum()),
            'Outer_validation_n': int(len(va_idx)),
            'Outer_validation_events': int(y_va.sum()),
            'Outer_validation_AUC': float(roc_auc_score(y_va, pva)),
            'Outer_validation_AP': float(average_precision_score(y_va, pva)),
            'Outer_validation_Brier': float(brier_score_loss(y_va, pva)),
        })

    if np.isnan(oof).any():
        raise RuntimeError('strict cross-fitted OOF probability contains NA')
    return oof, pd.DataFrame(fold_rows)


def holm_adjust_pvalues(pvalues):
    """Holm step-down multiplicity adjustment; returns adjusted p values in original order."""
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj_sorted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running = max(running, val)
        adj_sorted[rank] = min(running, 1.0)
    out = np.empty(m, dtype=float)
    for rank, idx in enumerate(order):
        out[idx] = adj_sorted[rank]
    return out

# ============================================================
# 数据加载 + 固定原始 Train/Test 身份 + 标准化
# ============================================================
print("="*70)
print("🚀 Pulmonary Infection Prediction — ML Classification (8 Models)")
print("   [ 11 Features · Fixed Original Split · Platt Calibration · Strict Cross-fit Threshold · SHAP ]")
print("="*70)

data = pd.read_csv(data_file, encoding='utf-8-sig')
data.columns = data.columns.str.replace('\ufeff','').str.strip()
print(f"\n✅ 输入数据: {data.shape[0]} 行 × {data.shape[1]} 列")

required_cols = [ID_COL, SPLIT_COL, OUTCOME] + FEATURES_USED
missing = [c for c in required_cols if c not in data.columns]
if missing:
    print(f"\n❌ 缺失列: {missing}"); sys.exit(1)

if len(data) != 3368:
    raise ValueError(f"总样本应为3368，实际={len(data)}")
if data[ID_COL].duplicated().any():
    raise ValueError(f"{ID_COL} 存在重复，停止分析。")

# 只保留本次主分析需要的ID、固定划分、结局和11变量。
data = data[required_cols].copy()
if data[OUTCOME].dtype == object:
    data[OUTCOME] = data[OUTCOME].map({'No':0,'Yes':1,'no':0,'yes':1,'0':0,'1':1,0:0,1:1})
data[OUTCOME] = data[OUTCOME].astype(int)

# 固定原始主分析 Train/Test 身份，不重新调用 train_test_split。
train_df = data[data[SPLIT_COL].astype(str).str.strip().str.lower() == 'train'].copy().reset_index(drop=True)
test_df  = data[data[SPLIT_COL].astype(str).str.strip().str.lower() == 'test'].copy().reset_index(drop=True)
if len(train_df) != 2357 or int(train_df[OUTCOME].sum()) != 950:
    raise ValueError(f"Train身份不匹配：n={len(train_df)}, events={int(train_df[OUTCOME].sum())}")
if len(test_df) != 1011 or int(test_df[OUTCOME].sum()) != 407:
    raise ValueError(f"Test身份不匹配：n={len(test_df)}, events={int(test_df[OUTCOME].sum())}")

# 本输入文件是审稿期锁定的analysis-ready exact-split数据；11变量若仍有缺失则停止，避免静默再填补。
if train_df[FEATURES_USED].isna().any().any() or test_df[FEATURES_USED].isna().any().any():
    miss_train = train_df[FEATURES_USED].isna().sum()
    miss_test = test_df[FEATURES_USED].isna().sum()
    raise ValueError(f"11个主分析变量仍存在缺失值。Train={miss_train[miss_train>0].to_dict()}, Test={miss_test[miss_test>0].to_dict()}")

X_train_raw = train_df[FEATURES_USED].copy().reset_index(drop=True)
X_test_raw  = test_df[FEATURES_USED].copy().reset_index(drop=True)
y_train = train_df[OUTCOME].astype(int).reset_index(drop=True)
y_test  = test_df[OUTCOME].astype(int).reset_index(drop=True)
train_ids = train_df[ID_COL].reset_index(drop=True)
test_ids  = test_df[ID_COL].reset_index(drop=True)

# 供后续SHAP相关矩阵使用：仍保留全部3368例的11变量原单位值。
X_raw = data[FEATURES_USED].copy().reset_index(drop=True)
y = data[OUTCOME].astype(int).reset_index(drop=True)

# 自动识别变量类型（保持原始机器学习代码逻辑）
binary_features=[]; continuous_features=[]
for col in FEATURES_USED:
    if train_df[col].nunique(dropna=True) <= 5 or train_df[col].dtype == object:
        binary_features.append(col)
    else:
        continuous_features.append(col)

print(f"\n🎯 固定划分核对：Train={len(y_train)} (events={int(y_train.sum())}) | Test={len(y_test)} (events={int(y_test.sum())})")
print(f"   特征({len(FEATURES_USED)}): {FEATURES_USED}")
print(f"   Categorical/binary ({len(binary_features)}): {binary_features}")
print(f"   Continuous ({len(continuous_features)}): {continuous_features}")
print(f"   校准: method={CALIBRATION_METHOD}, cv={CALIBRATION_CV}")

save_csv(data, '建模数据集_11features_固定原始TrainTest身份.csv')

# 连续变量仅在固定Train拟合MinMaxScaler，再原样应用于Test。
scaler = MinMaxScaler()
X_train_s = X_train_raw.copy()
X_test_s = X_test_raw.copy()
exist_cont = [f for f in continuous_features if f in X_train_raw.columns]
if exist_cont:
    X_train_s.loc[:, exist_cont] = scaler.fit_transform(X_train_raw[exist_cont])
    X_test_s.loc[:, exist_cont]  = scaler.transform(X_test_raw[exist_cont])

X_train_arr = X_train_s.values
X_test_arr  = X_test_s.values
feature_names = list(FEATURES_USED)

joblib.dump(scaler,        os.path.join(output_path,'标准化器.pkl'), compress=COMPRESS_LEVEL)
joblib.dump(feature_names, os.path.join(output_path,'特征名称列表.pkl'), compress=COMPRESS_LEVEL)

# 记录分析设计，避免后续混淆9变量/11变量版本。
analysis_design = {
    'N_total': int(len(data)),
    'Train_N': int(len(train_df)), 'Train_events': int(y_train.sum()),
    'Test_N': int(len(test_df)), 'Test_events': int(y_test.sum()),
    'Features': FEATURES_USED,
    'Fixed_original_split': True,
    'Bayes_tuning_CV': 10,
    'Platt_calibration_CV': CALIBRATION_CV,
    'Threshold_method': 'strict outer 10-fold cross-fitted calibrated training probabilities; hyperparameters fixed from full-Train Bayesian search',
    'SHAP_selection_mode': SHAP_SELECTION_MODE,
    'SHAP_fixed_model_if_requested': SHAP_FIXED_MODEL_KEY,
}
with open(os.path.join(output_path, '00_分析设计_11变量主分析.json'), 'w', encoding='utf-8') as f:
    import json
    json.dump(analysis_design, f, ensure_ascii=False, indent=2)

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
print(f"     3) Youden threshold from strict outer-10-fold cross-fitted calibrated Train probabilities")
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
strict_oof_probas={}      # 仅用于训练期阈值估计
strict_oof_fold_rows=[]
strict_oof_summary_rows=[]

def ckpt_path(name): return os.path.join(ckpt_dir, f'ckpt_{name}.pkl')

total_start=time.time()

for name, config in models_config.items():
    print(f"\n{'─'*58}")
    print(f"🔄 Model: {MODEL_FULL_NAME[name]}")

    cpath = ckpt_path(name)
    if os.path.exists(cpath):
        try:
            saved = joblib.load(cpath)
            if saved.get('features') != FEATURES_USED or saved.get('test_row_ids') != test_ids.tolist():
                raise ValueError('断点的变量集或Test身份与当前11变量主分析不一致')
            best_models[name]=saved['model']; raw_models[name]=saved['raw_model']
            optimal_thresholds[name]=saved['threshold']
            test_results[name]=saved['tm_te']; train_results[name]=saved['tm_tr']
            test_probas[name]=saved['yp_te']; train_probas[name]=saved['yp_tr']
            test_probas_raw[name]=saved['yp_te_raw']; train_probas_raw[name]=saved['yp_tr_raw']
            cv_aucs[name]=saved['cv_auc']; best_params_dict[name]=saved['best_params']
            platt_params[name]=saved['platt_params']
            calib_metrics[name]=saved['calib_metrics']
            strict_oof_probas[name]=np.asarray(saved['p_oof'])
            strict_oof_fold_rows.extend(saved.get('oof_fold_rows', []))
            strict_oof_summary_rows.append(saved.get('oof_summary_row', {}))
            print(f"  ✅ 从11变量断点加载")
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

        # ─── 3. 严格outer 10-fold cross-fitting生成训练期OOF概率，并据此锁定Youden阈值 ───
        print("  ⚙️  生成strict cross-fitted训练概率用于阈值估计 ...")
        p_oof, fold_df = strict_cross_fitted_train_probabilities(
            X_train_raw, y_train, FEATURES_USED, exist_cont, mdl_raw,
            outer_cv=10, calibration_cv=CALIBRATION_CV, seed=42
        )
        opt_thresh, oof_youden = choose_youden_threshold_from_probabilities(y_train, p_oof)
        strict_oof_probas[name] = p_oof
        fold_df.insert(0, 'Model', MODEL_FULL_NAME[name])
        strict_oof_fold_rows.extend(fold_df.to_dict('records'))
        oof_pred = (p_oof >= opt_thresh).astype(int)
        oof_metrics = calculate_metrics(y_train, oof_pred, p_oof)
        oof_summary_row = {
            'Model': MODEL_FULL_NAME[name],
            'Threshold': float(opt_thresh),
            'OOF_AUC': float(oof_metrics['AUC']),
            'OOF_AP': float(oof_metrics['AP']),
            'OOF_Brier': float(oof_metrics['Brier Score']),
            'OOF_Sensitivity': float(oof_metrics['Sensitivity']),
            'OOF_Specificity': float(oof_metrics['Specificity']),
            'OOF_MCC': float(oof_metrics['MCC']),
            'OOF_Youden': float(oof_metrics['Youden Index']),
        }
        strict_oof_summary_rows.append(oof_summary_row)
        print(f"  🔒 strict OOF threshold={opt_thresh:.4f} | OOF AUC={oof_metrics['AUC']:.4f} | Youden={oof_metrics['Youden Index']:.4f}")

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
                'features': FEATURES_USED,
                'test_row_ids': test_ids.tolist(),
                'p_oof': p_oof,
                'oof_fold_rows': fold_df.to_dict('records'),
                'oof_summary_row': oof_summary_row,
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

# ─── 预测概率（校准前、校准后）───
probas_df=pd.DataFrame(test_probas)
probas_df.columns=[f'{MODEL_FULL_NAME[n]} Prob' for n in probas_df.columns]
probas_df['True Label']=y_test.values
save_csv(probas_df,'测试集预测概率_校准后.csv')
probas_df_id = probas_df.copy(); probas_df_id.insert(0, ID_COL, test_ids.values)
save_csv(probas_df_id,'测试集逐患者预测概率_校准后_带StudyRowID.csv')

probas_raw_df=pd.DataFrame(test_probas_raw)
probas_raw_df.columns=[f'{MODEL_FULL_NAME[n]} Prob' for n in probas_raw_df.columns]
probas_raw_df['True Label']=y_test.values
save_csv(probas_raw_df,'测试集预测概率_校准前.csv')

# ─── 训练集预测概率（新增，供独立绘图脚本 / 图15 使用）───
probas_train_df=pd.DataFrame(train_probas)
probas_train_df.columns=[f'{MODEL_FULL_NAME[n]} Prob' for n in probas_train_df.columns]
probas_train_df['True Label']=y_train.values
save_csv(probas_train_df,'训练集预测概率_校准后.csv')

probas_train_raw_df=pd.DataFrame(train_probas_raw)
probas_train_raw_df.columns=[f'{MODEL_FULL_NAME[n]} Prob' for n in probas_train_raw_df.columns]
probas_train_raw_df['True Label']=y_train.values
save_csv(probas_train_raw_df,'训练集预测概率_校准前.csv')


# ─── strict cross-fitted训练OOF概率与阈值审计输出 ───
oof_prob_df = pd.DataFrame({ID_COL: train_ids.values, 'True Label': y_train.values})
for n in model_order:
    oof_prob_df[f'{MODEL_FULL_NAME[n]} Strict OOF Prob'] = strict_oof_probas[n]
save_csv(oof_prob_df, '训练集严格crossfit_OOF预测概率.csv')
save_csv(pd.DataFrame(strict_oof_summary_rows), '训练集严格crossfit_OOF阈值摘要.csv')
save_csv(pd.DataFrame(strict_oof_fold_rows), '训练集严格crossfit_OOF逐折结果.csv')

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
delong_df = pd.DataFrame(delong_rows)
if len(delong_df):
    delong_df['P_Holm'] = holm_adjust_pvalues(delong_df['P Value'].astype(float).values)
    delong_df['Sig_Holm'] = delong_df['P_Holm'].apply(
        lambda p: '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    )
save_csv(delong_df,'DeLong检验结果_含Holm校正.csv')

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
# 审稿后 focused DCA：0.25-0.40 + 1000次paired patient-level bootstrap
# ============================================================
print("\n📊 Focused DCA reanalysis (thresholds 0.25, 0.30, 0.40; bootstrap uncertainty)")

def calc_nb_revised(yt, yp, t):
    yt = np.asarray(yt).astype(int); yp = np.asarray(yp, dtype=float)
    n = len(yt)
    tp = int(np.sum((yp >= t) & (yt == 1)))
    fp = int(np.sum((yp >= t) & (yt == 0)))
    return tp / n - fp / n * (t / (1 - t))


def focused_dca_bootstrap(y_true, probas_dict, thresholds=(0.25,0.30,0.40), n_boot=1000, seed=42):
    yv = np.asarray(y_true).astype(int)
    n = len(yv); n_events = int(yv.sum()); prev = float(yv.mean())
    rng = np.random.default_rng(seed)
    rows = []
    for name in model_order:
        p = np.asarray(probas_dict[name], dtype=float)
        for t in thresholds:
            pred = p >= t
            tp = int(np.sum(pred & (yv == 1)))
            fp = int(np.sum(pred & (yv == 0)))
            fn = int(np.sum((~pred) & (yv == 1)))
            flagged_n = int(pred.sum())
            nb = calc_nb_revised(yv, p, t)
            treat_all = prev - (1-prev) * t/(1-t)
            default_nb = max(treat_all, 0.0)
            delta = nb - default_nb
            nb_boot=[]; d_boot=[]
            for _ in range(n_boot):
                idx = rng.integers(0, n, n)
                yb = yv[idx]; pb = p[idx]
                if len(np.unique(yb)) < 2:
                    continue
                prev_b = float(yb.mean())
                nb_b = calc_nb_revised(yb, pb, t)
                ta_b = prev_b - (1-prev_b)*t/(1-t)
                def_b = max(ta_b, 0.0)
                nb_boot.append(nb_b); d_boot.append(nb_b-def_b)
            nb_lo, nb_hi = np.percentile(nb_boot, [2.5,97.5])
            d_lo, d_hi = np.percentile(d_boot, [2.5,97.5])
            rows.append({
                'Model': MODEL_FULL_NAME[name], 'Threshold': t,
                'Flagged_n': flagged_n, 'Flagged_pct': flagged_n/n*100,
                'Missed_events_n': fn, 'Missed_events_pct': fn/n_events*100,
                'Net_Benefit': nb, 'NB_95CI_low': nb_lo, 'NB_95CI_high': nb_hi,
                'Best_Default_NB': default_nb,
                'Delta_NB_vs_best_default': delta,
                'Delta_NB_95CI_low': d_lo, 'Delta_NB_95CI_high': d_hi,
                'Bootstrap_successful': len(nb_boot),
            })
    return pd.DataFrame(rows)

focused_dca_df = focused_dca_bootstrap(y_test, test_probas)
save_csv(focused_dca_df, 'DCA_focused_0.25_0.40_八模型_bootstrap.csv')

# Focused curve with pointwise bootstrap bands for GBDT and LR.
focused_curve_models = [m for m in ['GBDT','LR'] if m in test_probas]
focused_grid = np.arange(0.25, 0.4001, 0.005)
prev_focus = float(y_test.mean())
rng_focus = np.random.default_rng(4242)
fig, ax = plt.subplots(figsize=(15, 9), facecolor='white')
for m in focused_curve_models:
    p = np.asarray(test_probas[m], dtype=float)
    point = np.array([calc_nb_revised(y_test.values, p, t) for t in focused_grid])
    boot_mat = np.empty((1000, len(focused_grid)), dtype=float)
    for b in range(1000):
        idx = rng_focus.integers(0, len(y_test), len(y_test))
        yb = y_test.values[idx]; pb = p[idx]
        boot_mat[b,:] = [calc_nb_revised(yb, pb, t) for t in focused_grid]
    lo = np.percentile(boot_mat, 2.5, axis=0)
    hi = np.percentile(boot_mat, 97.5, axis=0)
    ax.plot(focused_grid, point, lw=3.0, color=MODEL_COLORS[m], label=MODEL_FULL_NAME[m])
    ax.fill_between(focused_grid, lo, hi, color=MODEL_COLORS[m], alpha=0.15)
nb_all_focus = np.array([prev_focus-(1-prev_focus)*t/(1-t) for t in focused_grid])
ax.plot(focused_grid, nb_all_focus, 'k-', lw=2.4, label='Treat All')
ax.axhline(0, color='gray', ls='--', lw=2.2, label='Treat None')
ax.set_xlim(0.25,0.40)
ax.set_xlabel('Threshold Probability', fontsize=PR_DCA_FS_LABEL, fontweight='bold')
ax.set_ylabel('Net Benefit', fontsize=PR_DCA_FS_LABEL, fontweight='bold')
ax.tick_params(axis='both', labelsize=PR_DCA_FS_TICK, width=2.0, length=7)
for lb in ax.get_xticklabels()+ax.get_yticklabels(): lb.set_fontweight('bold')
ax.legend(loc='best', prop={'weight':'bold','size':PR_DCA_FS_LEGEND}, frameon=True)
ax.grid(True, alpha=0.3)
plt.tight_layout(); save_fig('DCA_focused_0.25_0.40_GBDT_LR_bootstrap.png'); plt.close()

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
def plot_roc_with_auc_bar(y_true, probas_dict, title, save_name):
    """
    左侧 ROC 曲线 + 右侧模型色块 + AUC 彩色条带（Blues 渐变），
    按当前数据集的 AUC 点估计降序排列。

    重要：右侧 AUC 数字直接由传入的 y_true 与校准后预测概率 probas_dict
    计算（roc_auc_score），与主分析点估计使用同一概率来源；不再从
    bootstrap 95%区间 CSV 中读取 bootstrap 均值。这样可避免图中 AUC
    标签与 Table S13 / “模型评估_*_点估计.csv”不一致。
    """
    model_order = list(probas_dict.keys())

    roc_data = []
    for name in model_order:
        fpr, tpr, _ = roc_curve(y_true, probas_dict[name])
        # Canonical AUC point estimate: exactly the same definition used in calculate_metrics().
        auc_val = float(roc_auc_score(y_true, probas_dict[name]))
        roc_data.append({
            'name': name, 'fpr': fpr, 'tpr': tpr,
            'auc': auc_val,
        })
    roc_sorted = sorted(roc_data, key=lambda d: d['auc'], reverse=True)

    print(f"  📌 [{save_name}] AUC彩带使用当前校准后预测概率的point estimate；"
          f"不使用bootstrap均值。")

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
    ax1.legend(loc='lower right', fontsize=ROC_FS_LEGEND, frameon=False,
               prop={'weight': 'bold', 'size': ROC_FS_LEGEND})
    ax1.tick_params(axis='both', labelsize=ROC_FS_TICK)
    for lb in ax1.get_xticklabels() + ax1.get_yticklabels():
        lb.set_fontweight('bold')

    # ─── 右侧 AUC 彩带 ───
    ax2 = plt.subplot(gs[1]); ax2.axis('off')
    n_m  = len(roc_sorted)
    aucs = [d['auc'] for d in roc_sorted]
    norm = Normalize(vmin=min(aucs) - 1e-6, vmax=max(aucs) + 1e-6)
    cmap_auc = plt.cm.Blues

    # AUC彩带颜色增强：最低AUC仍为最浅色，但不会与白底融合。
    AUC_CMAP_FLOOR = 0.28

    bar_h, gap = 0.85, 0.10
    for rank, rd in enumerate(roc_sorted):
        y_center = n_m - 1 - rank
        y_bot    = y_center - bar_h / 2

        # 左色块：模型颜色
        ax2.add_patch(plt.Rectangle((0.0, y_bot), 0.45, bar_h,
                                    facecolor=get_color(rd['name']),
                                    edgecolor='white', linewidth=1.5))
        # 右色块：增强后的Blues渐变
        auc_norm = float(norm(rd['auc']))
        auc_level = AUC_CMAP_FLOOR + (1.0 - AUC_CMAP_FLOOR) * auc_norm
        auc_color = cmap_auc(auc_level)
        ax2.add_patch(plt.Rectangle((0.50, y_bot), 0.50, bar_h,
                                    facecolor=auc_color,
                                    edgecolor='#B7C3CE', linewidth=1.2))
        # AUC 数字：显示4位小数，直接与Table S13点估计对齐
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
    # AUC数字与排序直接使用当前校准后测试概率计算的point estimate，
    # 与Table S13/模型评估_测试集_点估计.csv保持同一统计定义。
    plot_roc_with_auc_bar(
        y_test.values, test_probas,
        title='ROC Curve — Test Set Model Comparison',
        save_name='ROC曲线_AUC彩带_测试集.png')

    # 图15: 训练集 ROC+AUC 彩带
    plot_roc_with_auc_bar(
        y_train.values, train_probas,
        title='ROC Curve — Training Set Model Comparison',
        save_name='ROC曲线_AUC彩带_训练集.png')

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



# ============================================================
# SHAP 可解释性分析（直接使用本次11变量主分析内存中的同一模型、同一scaler、同一Test患者）
# ============================================================
if RUN_SHAP:
    import math
    import shap
    from scipy.stats import pearsonr
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.patches import Rectangle, FancyArrowPatch, FancyArrow
    from matplotlib.cm import ScalarMappable
    from matplotlib.transforms import blended_transform_factory

    # ------------------------------------------------------------
    # 选择SHAP解释模型。默认只使用训练期CV，不让Test集参与模型选择。
    # ------------------------------------------------------------
    available_models = [m for m in best_models if m in raw_models]
    available_tree_models = [m for m in SHAP_TREE_MODELS if m in available_models]

    if not available_models:
        raise RuntimeError('没有可用于SHAP的已训练模型。')

    if SHAP_SELECTION_MODE == 'fixed':
        MODEL_KEY = SHAP_FIXED_MODEL_KEY
        if MODEL_KEY not in available_models:
            raise KeyError(f"固定SHAP模型 {MODEL_KEY} 不可用，可选={available_models}")
        selection_note = f'fixed a priori/configured model: {MODEL_KEY}'

    elif SHAP_SELECTION_MODE == 'best_tree_train_cv':
        if not available_tree_models:
            raise RuntimeError('没有可用树模型，无法执行best_tree_train_cv。')
        MODEL_KEY = max(available_tree_models, key=lambda m: cv_aucs[m])
        selection_note = (
            f'highest Bayesian 10-fold CV AUC among tree models in Train: '
            f'{MODEL_KEY}, CV AUC={cv_aucs[MODEL_KEY]:.6f}'
        )

    elif SHAP_SELECTION_MODE == 'best_tree_test_auc':
        if not available_tree_models:
            raise RuntimeError('没有可用树模型，无法执行best_tree_test_auc。')
        MODEL_KEY = max(available_tree_models, key=lambda m: test_results[m]['AUC'])
        selection_note = (
            f'post hoc highest observed Test AUC among tree models: '
            f'{MODEL_KEY}, Test AUC={test_results[MODEL_KEY]["AUC"]:.6f}'
        )

    elif SHAP_SELECTION_MODE == 'best_overall_test_auc':
        MODEL_KEY = max(available_models, key=lambda m: test_results[m]['AUC'])
        selection_note = (
            f'post hoc highest observed Test AUC overall: '
            f'{MODEL_KEY}, Test AUC={test_results[MODEL_KEY]["AUC"]:.6f}'
        )

    else:
        raise ValueError(
            f'未知 SHAP_SELECTION_MODE={SHAP_SELECTION_MODE}; '
            "可选 fixed / best_tree_train_cv / best_tree_test_auc / best_overall_test_auc"
        )

    print("\n" + "="*88)
    print(f"SHAP解释模型={MODEL_KEY}；选择规则={SHAP_SELECTION_MODE}；11变量主分析Test集 n={len(y_test)}")
    print(f"选择依据: {selection_note}")
    print("="*88)

    shap_output_path = os.path.join(
        output_path, f"SHAP可解释性分析_{MODEL_KEY}_11变量"
    )
    os.makedirs(shap_output_path, exist_ok=True)

    model = best_models[MODEL_KEY]       # calibrated model: only for calibrated predicted probability/risk ranking
    raw_model = raw_models[MODEL_KEY]    # SHAP解释基模型，避免对CalibratedClassifierCV直接解释

    # 把实际选择结果写回分析设计JSON，便于论文/复核追溯。
    analysis_design['SHAP_selected_model'] = MODEL_KEY
    analysis_design['SHAP_selection_note'] = selection_note
    with open(os.path.join(output_path, '00_分析设计_11变量主分析.json'), 'w', encoding='utf-8') as f:
        json.dump(analysis_design, f, ensure_ascii=False, indent=2)
    feature_names = list(FEATURES_USED)

    # 直接使用主分析已经锁定并缩放好的同一Test患者，不再重新train_test_split、不再重新填补。
    X_test_df = X_test_s[feature_names].copy().reset_index(drop=True)
    X_test_scaled = X_test_df.values
    X_test_original = X_test_raw[feature_names].copy().reset_index(drop=True)
    X_full_original = X_raw[feature_names].copy().reset_index(drop=True)
    y_full = y.reset_index(drop=True)
    n_features = len(feature_names)

    print(f"  模型: {MODEL_FULL_NAME[MODEL_KEY]}")
    print(f"  特征({n_features}): {feature_names}")
    print(f"  SHAP输出目录: {shap_output_path}")

    plt.rcParams['font.sans-serif'] = ['Arial', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titleweight'] = 'bold'

    # ============================================================
    # 全局字号放大系数
    # ============================================================
    FONT_SCALE = 1.45

    plt.rcParams['font.size']        = 10   * FONT_SCALE
    plt.rcParams['axes.titlesize']   = 14   * FONT_SCALE
    plt.rcParams['axes.labelsize']   = 13   * FONT_SCALE
    plt.rcParams['xtick.labelsize']  = 12   * FONT_SCALE
    plt.rcParams['ytick.labelsize']  = 12   * FONT_SCALE
    plt.rcParams['legend.fontsize']  = 11   * FONT_SCALE
    plt.rcParams['legend.title_fontsize'] = 12 * FONT_SCALE

    SHAP_CBAR_LABEL_SIZE = int(17 * FONT_SCALE)
    SHAP_CBAR_TICK_SIZE  = int(15 * FONT_SCALE)


    def style_shap_summary_colorbar(fig, main_ax=None,
                                    label_size=SHAP_CBAR_LABEL_SIZE,
                                    tick_size=SHAP_CBAR_TICK_SIZE,
                                    labelpad=18):
        if fig is None or len(fig.axes) < 2:
            return
        cbar_ax = fig.axes[-1]
        if main_ax is not None and cbar_ax is main_ax:
            return
        cbar_ax.set_ylabel('Feature value', fontsize=label_size,
                           fontweight='bold', labelpad=labelpad)
        cbar_ax.yaxis.label.set_fontsize(label_size)
        cbar_ax.yaxis.label.set_fontweight('bold')
        cbar_ax.tick_params(axis='y', labelsize=tick_size, width=1.6, length=0)
        cbar_ax.tick_params(axis='x', labelsize=tick_size, width=1.6, length=0)
        for label in cbar_ax.get_yticklabels() + cbar_ax.get_xticklabels():
            label.set_fontsize(tick_size)
            label.set_fontweight('bold')
        for txt in cbar_ax.texts:
            txt.set_fontsize(tick_size)
            txt.set_fontweight('bold')

    # ============================================================
    # 统一图片尺寸
    # ============================================================
    FIG_SINGLE    = (10, 8)
    FIG_WATERFALL = (10, 8)
    FIG_FORCE     = (22, 5.2)
    FIG_HEATMAP   = (14, 8)

    # ============================================================
    # 配色方案
    # ============================================================
    SCHEME = 1

    _SCHEMES = {
        1: dict(
            PRIMARY='#2A6B7C', SECONDARY='#3D9B8F', LIGHT='#A3E4D7', DARK='#1B4F5A',
            WARM='#E8795A', WARM2='#C0392B',
            CMAP_LIST=['#1B4F5A','#2A6B7C','#3D9B8F','#A3E4D7','#F5D6C3','#E8795A','#C0392B'],
            BAR_LIST=['#1B4F5A','#22616E','#2A6B7C','#327D8A','#3D9B8F','#48A89A','#53B5A5','#5FC2B0','#6BCFBB','#7ED8C4','#91E1CD','#A3E4D7','#B5EDE0','#C8F2E8','#DAF7F0','#ECF9F5'],
        ),
        2: dict(
            PRIMARY='#3B3F8C', SECONDARY='#7B4FAF', LIGHT='#C9B8F0', DARK='#1E2063',
            WARM='#D4A017', WARM2='#A07800',
            CMAP_LIST=['#1E2063','#3B3F8C','#7B4FAF','#C9B8F0','#F5EBB8','#D4A017','#A07800'],
            BAR_LIST=['#1E2063','#2A2D78','#3B3F8C','#5255A5','#6A6DBF','#7B4FAF','#9466C4','#AC80D8','#C29AEB','#C9B8F0','#D6C8F5','#E2D8FA','#EDE8FC','#F4F0FE','#F9F7FF','#FFFFFF'],
        ),
        3: dict(
            PRIMARY='#1A6B3C', SECONDARY='#2EAA6E', LIGHT='#A8E6C3', DARK='#0E4025',
            WARM='#E8735A', WARM2='#C04030',
            CMAP_LIST=['#0E4025','#1A6B3C','#2EAA6E','#A8E6C3','#F5D5C8','#E8735A','#C04030'],
            BAR_LIST=['#0E4025','#145530','#1A6B3C','#228248','#2A9954','#2EAA6E','#46BD86','#5FD09E','#78E2B6','#92EDCA','#A8E6C3','#BFF0D5','#D4F5E5','#E8FAF1','#F4FCF8','#FAFFFC'],
        ),
        4: dict(
            PRIMARY='#0D2B55', SECONDARY='#1565C0', LIGHT='#90CAF9', DARK='#071830',
            WARM='#C8A200', WARM2='#8B6F00',
            CMAP_LIST=['#071830','#0D2B55','#1565C0','#90CAF9','#FFF9C4','#C8A200','#8B6F00'],
            BAR_LIST=['#071830','#0D2B55','#153670','#1C4288','#1565C0','#2577D3','#4092E1','#5EADEF','#82C4F7','#90CAF9','#AAD6FB','#C0E3FD','#D5EFFE','#E8F6FF','#F4FBFF','#FAFEFF'],
        ),
        5: dict(
            PRIMARY='#1F4E79', SECONDARY='#2E75B6', LIGHT='#A6C8E8', DARK='#0D2E4F',
            WARM='#E76F51', WARM2='#C44536',
            CMAP_LIST=['#0D2E4F','#2E75B6','#A6C8E8','#FDEAD7','#F5A88F','#E76F51','#C44536'],
            BAR_LIST=['#0D2E4F','#153862','#1D4275','#254C88','#2D569B','#3560AE','#3E6BC1','#497AD0','#5A8AD8','#6B9AE0','#7CAAE8','#8DB9EB','#9FC7EE','#B1D5F1','#C3E3F4','#D5EBF7'],
        ),
        6: dict(
            PRIMARY='#6B7E91', SECONDARY='#8FA4B8', LIGHT='#C5D0DA', DARK='#3E4E5E',
            WARM='#C08769', WARM2='#956549',
            CMAP_LIST=['#3E4E5E','#6B7E91','#8FA4B8','#C5D0DA','#EDD9C9','#C08769','#956549'],
            BAR_LIST=['#3E4E5E','#495B6E','#546880','#607591','#6B7E91','#7890A4','#8FA4B8','#A0B3C4','#B2C2D0','#C5D0DA','#D3DCE4','#DFE6EC','#E8EDF1','#EFF2F5','#F4F6F8','#F9FAFB'],
        ),
        7: dict(
            PRIMARY='#2E5E5E', SECONDARY='#4A8080', LIGHT='#A4C5C5', DARK='#1A3E3E',
            WARM='#C8484E', WARM2='#9A3840',
            CMAP_LIST=['#1A3E3E','#2E5E5E','#4A8080','#A4C5C5','#E8C5C0','#C8484E','#9A3840'],
            BAR_LIST=['#1A3E3E','#234D4D','#2E5E5E','#3A6F6F','#4A8080','#5C9191','#70A2A2','#84B3B3','#A4C5C5','#B8D3D3','#C8DDDD','#D5E5E5','#E0EBEB','#E9F0F0','#F0F5F5','#F6F9F9'],
        ),
        8: dict(
            PRIMARY='#264653', SECONDARY='#2A9D8F', LIGHT='#A8DADC', DARK='#1B2E3A',
            WARM='#E9C46A', WARM2='#E76F51',
            CMAP_LIST=['#1B2E3A','#264653','#2A9D8F','#A8DADC','#F4E0A6','#E9C46A','#E76F51'],
            BAR_LIST=['#1B2E3A','#21384A','#264653','#2A5862','#2E6B72','#318282','#2A9D8F','#43ADA0','#5BBDB2','#74CDC4','#8BD8D0','#A8DADC','#BFE4E6','#D3ECED','#E3F2F3','#F0F8F8'],
        ),
        9: dict(
            PRIMARY='#2E1A47', SECONDARY='#5D3A8E', LIGHT='#B8A4D4', DARK='#1B0F2B',
            WARM='#E8B04B', WARM2='#B87F22',
            CMAP_LIST=['#1B0F2B','#2E1A47','#5D3A8E','#B8A4D4','#F5E6C4','#E8B04B','#B87F22'],
            BAR_LIST=['#1B0F2B','#271542','#2E1A47','#3D2560','#4D3078','#5D3A8E','#744FA3','#8B65B8','#A27ACD','#B493D6','#B8A4D4','#C3B3DC','#CEC0E3','#D9CEEB','#E3DBF2','#EDE8F8'],
        ),
    }

    _s = _SCHEMES[SCHEME]
    COLOR_PRIMARY   = _s['PRIMARY']
    COLOR_SECONDARY = _s['SECONDARY']
    COLOR_LIGHT     = _s['LIGHT']
    COLOR_DARK      = _s['DARK']
    COLOR_WARM      = _s['WARM']
    COLOR_WARM2     = _s['WARM2']

    SHAP_CMAP = LinearSegmentedColormap.from_list(
        f'shap_scheme{SCHEME}', _s['CMAP_LIST'], N=256
    )

    BAR_COLORS_BASE = _s['BAR_LIST'] * 3
    # 按当前实际特征数截取颜色。整合SHAP脚本时此行遗漏会导致图2报NameError。
    BAR_COLORS = BAR_COLORS_BASE[:n_features]

    _SCHEME_NAMES = {
        1:'深海青绿', 2:'暮色紫蓝', 3:'森林翡翠', 4:'极夜蓝金',
        5:'珊瑚海蓝', 6:'莫兰迪雾色', 7:'青瓷朱砂', 8:'薄荷日落', 9:'紫水晶金',
    }
    print(f"  🎨 当前配色方案: {SCHEME} — {_SCHEME_NAMES[SCHEME]}")

    # ============================================================
    def save_shap_fig(filename):
        plt.savefig(os.path.join(shap_output_path, filename),
                    dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  📊 已保存: {filename}")
    # ─── 特征显示名称映射 ───
    DISPLAY_NAME_MAP = {
        'NEUT_abs': 'NEU',
        'Intubation_tracheotomy': 'Intubation',
        'Mechanical_ventilation': 'MV',
        'LDH': 'LDH',
        'LYMPH_abs': 'LYM',
        'BUN': 'BUN',
        'CCI': 'CCI',
        'FIB': 'FIB',
        'Surgery': 'Surgery',
        'Diuretics': 'Diuretics',
        'CO2': 'TCO2',
    }
    display_names  = [DISPLAY_NAME_MAP.get(n, n) for n in feature_names]
    X_test_display = X_test_original.copy()
    X_test_display.columns = display_names

    print(f"  显示名映射: {dict(zip(feature_names, display_names))}")

    # ============================================================
    # 计算 SHAP 值
    # ============================================================
    print("\n📊 计算SHAP值...")

    TREE_MODELS = {'RF', 'GBDT', 'DT', 'LightGBM', 'XGBoost'}
    if MODEL_KEY in TREE_MODELS:
        explainer = shap.TreeExplainer(raw_model)
        shap_values_test = explainer.shap_values(X_test_df)
    else:
        print(f"  ⚠️ {MODEL_KEY} 不是树模型，改用 KernelExplainer（可能较慢）...")
        bg = shap.sample(X_test_df, min(50, len(X_test_df)), random_state=42)
        explainer = shap.KernelExplainer(lambda x: raw_model.predict_proba(x)[:, 1], bg)
        shap_values_test = explainer.shap_values(X_test_df, nsamples=100)

    if isinstance(shap_values_test, list):
        shap_values_test = shap_values_test[1]

    if shap_values_test.ndim == 3:
        shap_values_test = shap_values_test[:, :, 1]

    base_value = explainer.expected_value
    if isinstance(base_value, np.ndarray):
        base_value = float(base_value[-1])
    elif isinstance(base_value, (list, tuple)):
        base_value = float(base_value[-1])
    else:
        base_value = float(base_value)

    mean_abs_shap = np.abs(shap_values_test).mean(axis=0)
    y_proba_test  = np.asarray(test_probas[MODEL_KEY], dtype=float)

    print(f"  ✅ SHAP值计算完成")
    print(f"  SHAP shape: {shap_values_test.shape}")
    print(f"  Base value (raw model): {base_value:.3f}")

    # ============================================================
    # 图1: SHAP 蜂群图
    # ============================================================
    print("\n📈 绘制SHAP蜂群图...")

    fig, ax = plt.subplots(figsize=FIG_SINGLE, facecolor='white')
    shap.summary_plot(
        shap_values_test, X_test_display,
        plot_type='dot', show=False, max_display=n_features,
        cmap=SHAP_CMAP, plot_size=None
    )
    ax = plt.gca()
    ax.set_title('')
    ax.set_xlabel('SHAP Value', fontsize=int(14*FONT_SCALE), fontweight='bold')
    ax.tick_params(axis='y', labelsize=int(13*FONT_SCALE))
    ax.tick_params(axis='x', labelsize=int(12*FONT_SCALE))
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
    style_shap_summary_colorbar(fig, ax)
    fig.set_size_inches(FIG_SINGLE)
    plt.tight_layout()
    save_shap_fig('图1_SHAP蜂群图_特征对模型输出的影响.png')
    plt.close()

    # ============================================================
    # 图2: SHAP 特征重要性条形图
    # ============================================================
    print("  绘制SHAP特征重要性条形图...")

    importance_df = pd.DataFrame({
        'Feature':   display_names,
        'Mean_SHAP': mean_abs_shap
    }).sort_values('Mean_SHAP', ascending=True)

    fig, ax = plt.subplots(figsize=FIG_SINGLE, facecolor='white')
    n_feat = len(importance_df)
    colors = BAR_COLORS[:n_feat][::-1]

    bars = ax.barh(range(n_feat), importance_df['Mean_SHAP'].values,
                   color=colors, edgecolor='white', linewidth=0.5, height=0.7)

    ax.set_yticks(range(n_feat))
    ax.set_yticklabels(importance_df['Feature'].values, fontsize=int(13*FONT_SCALE), fontweight='bold')
    ax.set_xlabel('Mean |SHAP Value|', fontsize=int(14*FONT_SCALE), fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.tick_params(axis='x', labelsize=int(12*FONT_SCALE))
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')

    for bar, val in zip(bars, importance_df['Mean_SHAP'].values):
        ax.text(val + 0.001, bar.get_y() + bar.get_height()/2, f'{val:.3f}',
                va='center', ha='left', fontsize=int(11*FONT_SCALE), fontweight='bold', color=COLOR_DARK)

    plt.tight_layout()
    save_shap_fig('图2_SHAP特征重要性排序.png')
    plt.close()

    # ============================================================
    # 图3: SHAP 小提琴图
    # ============================================================
    print("  绘制SHAP小提琴图...")

    fig, ax = plt.subplots(figsize=FIG_SINGLE, facecolor='white')
    shap.summary_plot(
        shap_values_test, X_test_display,
        plot_type='violin', show=False, max_display=n_features,
        cmap=SHAP_CMAP, plot_size=None
    )
    ax = plt.gca()
    ax.set_title('')
    ax.set_xlabel('SHAP Value', fontsize=int(14*FONT_SCALE), fontweight='bold')
    ax.tick_params(axis='y', labelsize=int(13*FONT_SCALE))
    ax.tick_params(axis='x', labelsize=int(12*FONT_SCALE))
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
    style_shap_summary_colorbar(fig, ax)
    fig.set_size_inches(FIG_SINGLE)
    plt.tight_layout()
    save_shap_fig('图3_SHAP小提琴图_特征值分布.png')
    plt.close()

    # ============================================================
    # 图4: SHAP 散点图
    # ============================================================
    print(f"  绘制SHAP散点图（全部{n_features}个特征）...")

    all_idx_sorted = np.argsort(mean_abs_shap)[::-1]

    n_cols = min(5, n_features)
    n_rows = math.ceil(n_features / n_cols)
    fig_w  = 5.5 * n_cols
    fig_h  = 5.0 * n_rows

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), facecolor='white')
    axes = np.array(axes).flatten()

    BINARY_JITTER_WIDTH = 0.18
    BINARY_XLIM         = (-0.7, 1.7)

    INT_DISCRETE_JITTER_WIDTH = 0.15
    INT_DISCRETE_MAX_LEVELS   = 15

    FORCE_JITTER_FEATURES = {'CCI'}

    _jitter_rng = np.random.RandomState(42)


    def _is_integer_discrete(x_vals, max_levels=INT_DISCRETE_MAX_LEVELS):
        x = np.asarray(x_vals, dtype=float)
        uniq = np.unique(np.round(x, 6))
        if uniq.size > max_levels or uniq.size < 2:
            return False
        diffs = np.diff(uniq)
        return bool(np.all(np.abs(diffs - 1.0) < 0.05))


    for idx, feat_idx in enumerate(all_idx_sorted):
        ax = axes[idx]
        feat_name = feature_names[feat_idx]
        x_vals    = X_test_original.iloc[:, feat_idx].values.astype(float)
        y_vals    = shap_values_test[:, feat_idx]
        is_binary = feat_name in binary_features

        forced       = feat_name in FORCE_JITTER_FEATURES
        auto_int_ok  = _is_integer_discrete(x_vals)
        is_int_discrete = (not is_binary) and (forced or auto_int_ok)

        if is_binary:
            jitter = _jitter_rng.uniform(-BINARY_JITTER_WIDTH, BINARY_JITTER_WIDTH, size=len(x_vals))
            x_plot = x_vals + jitter
        elif is_int_discrete:
            jitter = _jitter_rng.uniform(-INT_DISCRETE_JITTER_WIDTH,
                                         INT_DISCRETE_JITTER_WIDTH, size=len(x_vals))
            x_plot = x_vals + jitter
        else:
            x_plot = x_vals

        sc = ax.scatter(
            x_plot, y_vals,
            c=x_vals, cmap=SHAP_CMAP,
            s=12, alpha=0.6, edgecolors='none'
        )
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.set_xlabel(display_names[feat_idx], fontsize=int(12*FONT_SCALE), fontweight='bold')
        ax.set_ylabel('SHAP Value', fontsize=int(12*FONT_SCALE), fontweight='bold')

        if is_binary:
            ax.set_xlim(*BINARY_XLIM)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['0', '1'])
        elif is_int_discrete:
            uniq = np.sort(np.unique(np.round(x_vals)).astype(int))
            ax.set_xticks(uniq)
            ax.set_xticklabels([str(v) for v in uniq])
            ax.set_xlim(uniq.min() - 0.6, uniq.max() + 0.6)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.2)
        ax.tick_params(axis='both', labelsize=int(10*FONT_SCALE))
        for lb in ax.get_xticklabels() + ax.get_yticklabels():
            lb.set_fontweight('bold')
        cbar = plt.colorbar(sc, ax=ax, shrink=0.8)
        cbar.set_label('Feature Value', fontsize=int(9*FONT_SCALE), fontweight='bold')

    for idx in range(n_features, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    save_shap_fig(f'图4_SHAP散点图_全部{n_features}个特征.png')
    plt.close()

    # ============================================================
    # 图5 / 图6 共用：把特征值预格式化为字符串
    # ============================================================
    INTEGER_DISPLAY_RAW  = list(set(binary_features) | {'CCI'})
    INTEGER_DISPLAY_DISP = {DISPLAY_NAME_MAP.get(n, n) for n in INTEGER_DISPLAY_RAW}

    X_test_formatted_str = X_test_original.copy()
    X_test_formatted_str.columns = display_names

    _str_data = pd.DataFrame(index=X_test_formatted_str.index)
    for col in X_test_formatted_str.columns:
        if col in INTEGER_DISPLAY_DISP:
            _str_data[col] = X_test_formatted_str[col].apply(
                lambda v: str(int(round(float(v))))
            )
        else:
            _str_data[col] = X_test_formatted_str[col].apply(
                lambda v: f"{float(v):.1f}"
            )

    X_test_formatted_display = _str_data

    shap_explanation_test = shap.Explanation(
        values=shap_values_test,
        base_values=np.full(shap_values_test.shape[0], base_value),
        data=_str_data.values,
        feature_names=display_names
    )

    # ============================================================
    # 图5: SHAP 瀑布图
    # ============================================================
    print("  绘制SHAP瀑布图（典型样本）...")

    high_risk_idx = int(np.argmax(y_proba_test))
    low_risk_idx  = int(np.argmin(y_proba_test))
    mid_idx       = int(np.argmin(np.abs(y_proba_test - 0.5)))

    sample_cases = [
        (high_risk_idx, '高风险患者', 'High_Risk'),
        (low_risk_idx,  '低风险患者', 'Low_Risk'),
        (mid_idx,       '边界患者',   'Borderline')
    ]

    for sample_idx, cn_label, en_label in sample_cases:
        plt.figure(figsize=FIG_WATERFALL, facecolor='white')
        shap.plots.waterfall(shap_explanation_test[sample_idx],
                             max_display=n_features, show=False)
        ax = plt.gca()
        ax.set_title('')
        ax.tick_params(axis='y', labelsize=int(13*FONT_SCALE))
        ax.tick_params(axis='x', labelsize=int(12*FONT_SCALE))
        for lb in ax.get_yticklabels():
            lb.set_fontweight('bold')
        for lb in ax.get_xticklabels():
            lb.set_fontweight('bold')
        plt.gcf().set_size_inches(FIG_WATERFALL)
        plt.tight_layout()
        save_shap_fig(f'图5_SHAP瀑布图_{cn_label}.png')
        plt.close()

    # ============================================================
    # 图6: SHAP Force Plot —— 顶部标签重排
    # ============================================================
    # 关键修复（本版新增）：
    #   ❶ 暴力清理 ax.texts 中所有 Unicode 箭头字符
    #   ❷ 暴力清理 ax.patches 中所有 FancyArrowPatch / FancyArrow（SHAP 在
    #      f(x) 数值下方画的小红蓝双箭头就是这种类型，不是文本！）
    #   ❸ 用 annotate 重绘整齐的 higher → ← lower 顶部标签
    #   ❹ SHAP 默认的 base value 灰色细指示线保留不动

    def _is_arrow_char(c):
        """判断单个字符是不是任意 Unicode 箭头。"""
        o = ord(c)
        return (0x2190 <= o <= 0x21FF) or (0x27F0 <= o <= 0x27FF) or \
               (0x2900 <= o <= 0x297F) or (0x2B00 <= o <= 0x2BFF)


    def _redraw_force_plot_top(fig, fx_value, base_value):
        if not fig.axes:
            return
        ax   = fig.axes[0]
        xlim = ax.get_xlim()
        xspan = xlim[1] - xlim[0]

        # ★ 第 0 步：暴力清理所有包含 Unicode 箭头字符的 text
        for t in list(ax.texts):
            s = t.get_text()
            if any(_is_arrow_char(c) for c in s):
                t.set_visible(False)

        # ★ 第 0.5 步（关键新增）：清理 ax.patches 中所有 FancyArrowPatch / FancyArrow
        # SHAP 在 f(x) 数值下方画的"小红蓝双箭头"就是 FancyArrowPatch 对象，
        # 它不在 ax.texts 里，所以前面的字符过滤抓不到它。
        # force plot 主体的楔形是 Polygon/Rectangle，不是箭头 patch，所以删它不会误伤。
        for p in list(ax.patches):
            if isinstance(p, (FancyArrowPatch, FancyArrow)):
                p.set_visible(False)

        fx_x_pos   = None
        base_x_pos = None

        # 1) 收集并隐藏 SHAP 默认的顶部文字标签
        for t in list(ax.texts):
            if not t.get_visible():
                continue
            s = t.get_text().strip()
            s_clean = (s.replace('−', '-').replace('\u2212', '-')
                        .replace(',', '').replace(' ', ''))
            x_pos = t.get_position()[0]

            if s in ('higher', 'lower', 'f(x)'):
                t.set_visible(False)
                if s == 'f(x)' and fx_x_pos is None:
                    fx_x_pos = x_pos
                continue

            if 'base value' in s.lower():
                t.set_visible(False)
                if base_x_pos is None:
                    base_x_pos = x_pos
                continue

            try:
                v = float(s_clean)
                if abs(v - fx_value) < 0.5:
                    t.set_visible(False)
                    if fx_x_pos is None:
                        fx_x_pos = x_pos
                    continue
                if abs(v - base_value) < 0.3:
                    t.set_visible(False)
                    if base_x_pos is None:
                        base_x_pos = x_pos
                    continue
            except (ValueError, TypeError):
                pass

        if fx_x_pos   is None: fx_x_pos   = fx_value
        if base_x_pos is None: base_x_pos = base_value

        # 2) 重绘
        trans = blended_transform_factory(ax.transData, fig.transFigure)

        color_higher = '#FF0051'
        color_lower  = '#1E88E5'

        text_offset = xspan * 0.045
        arr_inner   = xspan * 0.008
        arr_outer   = xspan * 0.025

        # higher 文字
        ax.text(fx_x_pos - text_offset, 0.94, 'higher',
                color=color_higher, fontsize=13, fontweight='bold',
                ha='right', va='center', transform=trans, clip_on=False, zorder=10)

        # 红色 → 箭头
        ax.annotate('',
                    xy=(fx_x_pos - arr_inner, 0.94),
                    xytext=(fx_x_pos - arr_outer, 0.94),
                    xycoords=trans, textcoords=trans,
                    arrowprops=dict(arrowstyle='->', color=color_higher, lw=1.8),
                    annotation_clip=False, zorder=10)

        # 蓝色 ← 箭头
        ax.annotate('',
                    xy=(fx_x_pos + arr_inner, 0.94),
                    xytext=(fx_x_pos + arr_outer, 0.94),
                    xycoords=trans, textcoords=trans,
                    arrowprops=dict(arrowstyle='->', color=color_lower, lw=1.8),
                    annotation_clip=False, zorder=10)

        # lower 文字
        ax.text(fx_x_pos + text_offset, 0.94, 'lower',
                color=color_lower, fontsize=13, fontweight='bold',
                ha='left', va='center', transform=trans, clip_on=False, zorder=10)

        # f(x) 标签
        ax.text(fx_x_pos, 0.86, 'f(x)',
                color='#444444', fontsize=11, style='italic',
                ha='center', va='center', transform=trans, clip_on=False, zorder=10)

        # f(x) 数值
        fx_text = ax.text(fx_x_pos, 0.78, f'{fx_value:.2f}',
                          color='#000000', fontsize=14, fontweight='bold',
                          ha='center', va='center',
                          transform=trans, clip_on=False, zorder=11)
        fx_text.set_bbox(dict(facecolor='white', edgecolor='white', pad=2.5))

        # base value 区域
        ax.text(base_x_pos, 0.86, 'base value',
                color='#444444', fontsize=10,
                ha='center', va='center', transform=trans, clip_on=False, zorder=10)
        bv_text = ax.text(base_x_pos, 0.78, f'{base_value:.2f}',
                          color='#000000', fontsize=11, fontweight='bold',
                          ha='center', va='center',
                          transform=trans, clip_on=False, zorder=11)
        bv_text.set_bbox(dict(facecolor='white', edgecolor='white', pad=2.0))


    print("  绘制SHAP Force Plot...")

    for sample_idx, cn_label, en_label in sample_cases:
        fig = shap.plots.force(shap_explanation_test[sample_idx],
                               show=False, matplotlib=True,
                               figsize=FIG_FORCE,
                               text_rotation=0)
        plt.title('')
        fig.subplots_adjust(top=0.62, bottom=0.32, left=0.04, right=0.96)
        fx_value_i = float(base_value + shap_values_test[sample_idx].sum())
        _redraw_force_plot_top(fig, fx_value_i, base_value)
        save_shap_fig(f'图6_SHAP力图_{cn_label}.png')
        plt.close()

    # ============================================================
    # 图7: SHAP 交互作用图
    # ============================================================
    print("  绘制SHAP交互作用图（Top 5特征）...")

    top5_n   = min(5, n_features)
    top5_idx = np.argsort(mean_abs_shap)[::-1][:top5_n]

    pairs = []
    for i in range(len(top5_idx)):
        for j in range(i + 1, len(top5_idx)):
            pairs.append((top5_idx[i], top5_idx[j]))

    n_pairs = len(pairs)
    p_cols  = min(5, n_pairs)
    p_rows  = math.ceil(n_pairs / p_cols)
    fig, axes = plt.subplots(p_rows, p_cols,
                             figsize=(6 * p_cols, 5 * p_rows), facecolor='white')
    axes = np.array(axes).flatten()

    for idx, (i, j) in enumerate(pairs):
        ax = axes[idx]
        feat_name_i = feature_names[i]
        x_vals      = X_test_original.iloc[:, i].values.astype(float)
        y_vals      = shap_values_test[:, i]
        c_vals      = X_test_original.iloc[:, j].values
        is_binary_i = feat_name_i in binary_features
        is_int_discrete_i = (not is_binary_i) and (
            (feat_name_i in FORCE_JITTER_FEATURES) or _is_integer_discrete(x_vals)
        )

        if is_binary_i:
            jitter = _jitter_rng.uniform(-BINARY_JITTER_WIDTH, BINARY_JITTER_WIDTH, size=len(x_vals))
            x_plot = x_vals + jitter
        elif is_int_discrete_i:
            jitter = _jitter_rng.uniform(-INT_DISCRETE_JITTER_WIDTH,
                                         INT_DISCRETE_JITTER_WIDTH, size=len(x_vals))
            x_plot = x_vals + jitter
        else:
            x_plot = x_vals

        sc = ax.scatter(
            x_plot, y_vals,
            c=c_vals, cmap=SHAP_CMAP,
            s=12, alpha=0.6, edgecolors='none'
        )
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.set_xlabel(display_names[i], fontsize=int(12*FONT_SCALE), fontweight='bold')
        ax.set_ylabel('SHAP Value',     fontsize=int(11*FONT_SCALE), fontweight='bold')

        if is_binary_i:
            ax.set_xlim(*BINARY_XLIM)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['0', '1'])
        elif is_int_discrete_i:
            uniq = np.sort(np.unique(np.round(x_vals)).astype(int))
            ax.set_xticks(uniq)
            ax.set_xticklabels([str(v) for v in uniq])
            ax.set_xlim(uniq.min() - 0.6, uniq.max() + 0.6)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.2)
        ax.tick_params(axis='both', labelsize=int(10*FONT_SCALE))
        for lb in ax.get_xticklabels() + ax.get_yticklabels():
            lb.set_fontweight('bold')
        cbar = plt.colorbar(sc, ax=ax, shrink=0.8)
        cbar.set_label(display_names[j], fontsize=int(10*FONT_SCALE), fontweight='bold')

    for idx in range(n_pairs, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    save_shap_fig('图7_SHAP交互作用图_前5个重要特征.png')
    plt.close()

    # ============================================================
    # 图8: 肺炎组 vs 无肺炎组 SHAP 特征重要性对比
    # ============================================================
    print("  绘制肺炎组vs无肺炎组SHAP特征重要性对比...")

    pneu_mask   = (y_test == 1).values
    nopneu_mask = (y_test == 0).values

    mean_shap_pneu   = np.abs(shap_values_test[pneu_mask]).mean(axis=0)
    mean_shap_nopneu = np.abs(shap_values_test[nopneu_mask]).mean(axis=0)

    compare_df = pd.DataFrame({
        'Feature':      display_names,
        'Pneumonia':    mean_shap_pneu,
        'No_Pneumonia': mean_shap_nopneu
    })
    compare_df['Diff'] = compare_df['Pneumonia'] - compare_df['No_Pneumonia']
    compare_df = compare_df.sort_values('Diff', ascending=True)

    fig, ax = plt.subplots(figsize=FIG_SINGLE, facecolor='white')
    y_pos = np.arange(len(compare_df))
    width = 0.35

    ax.barh(y_pos - width/2, compare_df['Pneumonia'].values, width,
            color=COLOR_WARM,  alpha=0.85, label='Pneumonia',    edgecolor='white')
    ax.barh(y_pos + width/2, compare_df['No_Pneumonia'].values, width,
            color=COLOR_PRIMARY, alpha=0.85, label='No Pneumonia', edgecolor='white')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(compare_df['Feature'].values, fontsize=int(13*FONT_SCALE), fontweight='bold')
    ax.set_xlabel('Mean |SHAP Value|', fontsize=int(14*FONT_SCALE), fontweight='bold')
    ax.legend(fontsize=int(13*FONT_SCALE), loc='lower right', prop={'weight': 'bold'})
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.tick_params(axis='x', labelsize=int(12*FONT_SCALE))
    for lb in ax.get_xticklabels():
        lb.set_fontweight('bold')

    plt.tight_layout()
    save_shap_fig('图8_SHAP特征重要性_肺炎组vs无肺炎组对比.png')
    plt.close()

    # ============================================================
    # 图9: SHAP 热力图
    # ============================================================
    print("  绘制SHAP热力图...")

    n_samples        = len(y_proba_test)
    n_show           = min(25, n_samples // 2)
    importance_order = np.argsort(mean_abs_shap)[::-1]
    top_high         = np.argsort(y_proba_test)[::-1][:n_show]
    top_low          = np.argsort(y_proba_test)[:n_show]
    sample_indices   = np.concatenate([top_high, top_low])

    shap_subset = shap_values_test[sample_indices][:, importance_order]
    feat_labels = [display_names[i] for i in importance_order]

    fig, ax = plt.subplots(figsize=FIG_HEATMAP, facecolor='white')
    vmax = np.percentile(np.abs(shap_subset), 95)
    im   = ax.imshow(shap_subset, cmap=SHAP_CMAP, aspect='auto',
                     vmin=-vmax, vmax=vmax, interpolation='nearest')

    ax.set_xticks(range(len(feat_labels)))
    ax.set_xticklabels(feat_labels, rotation=45, ha='right', fontsize=int(12*FONT_SCALE), fontweight='bold')
    ax.set_ylabel(f'Samples (Top {n_show} High-Risk → Top {n_show} Low-Risk)',
                  fontsize=int(13*FONT_SCALE), fontweight='bold')
    ax.tick_params(axis='y', labelsize=int(11*FONT_SCALE))

    divider_y = n_show - 0.5
    ax.axhline(y=divider_y, color='white', linewidth=2, linestyle='--')
    ax.text(-0.5, n_show / 2,          'High Risk', rotation=90, va='center', ha='right',
            fontsize=int(12*FONT_SCALE), fontweight='bold', color=COLOR_WARM)
    ax.text(-0.5, n_show + n_show / 2,  'Low Risk',  rotation=90, va='center', ha='right',
            fontsize=int(12*FONT_SCALE), fontweight='bold', color=COLOR_PRIMARY)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('SHAP Value', fontsize=int(13*FONT_SCALE), fontweight='bold')

    plt.tight_layout()
    save_shap_fig('图9_SHAP热力图_高低风险样本对比.png')
    plt.close()

    # ============================================================
    # 图10: SHAP 决策图
    # ============================================================
    print("  绘制SHAP决策图（全部测试集样本决策路径）...")

    decision_idx = np.arange(len(y_proba_test))
    highlight_positions = [int(high_risk_idx), int(low_risk_idx), int(mid_idx)]

    plt.figure(figsize=(11, 8), facecolor='white')

    shap.decision_plot(
        base_value,
        shap_values_test[decision_idx],
        features=X_test_formatted_display.iloc[decision_idx].values,
        feature_names=display_names,
        feature_order='importance',
        highlight=highlight_positions if highlight_positions else None,
        plot_color=SHAP_CMAP,
        link='identity',
        auto_size_plot=False,
        show=False,
    )

    ax = plt.gca()
    ax.set_title('')
    ax.set_xlabel('Model Output Value', fontsize=int(13*FONT_SCALE), fontweight='bold')
    ax.tick_params(axis='both', labelsize=int(11*FONT_SCALE))
    for lb in ax.get_xticklabels() + ax.get_yticklabels():
        lb.set_fontweight('bold')

    plt.tight_layout()
    save_shap_fig('图10_SHAP决策图_样本决策路径.png')
    plt.close()

    # ============================================================
    # 图10b: SHAP 决策图 — 概率空间版
    # ============================================================
    print("  绘制SHAP决策图（概率空间版）...")

    plt.figure(figsize=(11, 8), facecolor='white')
    shap.decision_plot(
        base_value,
        shap_values_test[decision_idx],
        features=X_test_formatted_display.iloc[decision_idx].values,
        feature_names=display_names,
        feature_order='importance',
        highlight=highlight_positions if highlight_positions else None,
        plot_color=SHAP_CMAP,
        link='logit',
        auto_size_plot=False,
        show=False,
    )
    ax = plt.gca()
    ax.set_title('')
    ax.set_xlabel('Predicted Probability of Pulmonary Infection',
                  fontsize=int(13*FONT_SCALE), fontweight='bold')
    ax.tick_params(axis='both', labelsize=int(11*FONT_SCALE))
    for lb in ax.get_xticklabels() + ax.get_yticklabels():
        lb.set_fontweight('bold')

    plt.tight_layout()
    save_shap_fig('图10b_SHAP决策图_概率空间.png')
    plt.close()

    # ============================================================
    # 图11: 相关矩阵
    # ============================================================
    MATRIX_FS_DIAG_LABEL  = 17
    MATRIX_FS_Y_LABEL     = 18
    MATRIX_FS_STAR        = 16
    MATRIX_FS_CBAR_TICK   = 17
    MATRIX_FS_CBAR_LABEL  = 17
    MATRIX_Y_TICK_PAD     = -3
    CORR_CBAR_BBOX        = [0.22, 0.2, 0.56, 0.022]
    CORR_LEGEND_Y         = 0.16
    SHAP_CBAR_MAIN_BBOX   = [0.14, 0.2, 0.34, 0.022]
    SHAP_CBAR_INTER_BBOX  = [0.54, 0.2, 0.34, 0.022]
    SHAP_CBAR_LABEL_Y     = 0.16

    print("\n📈 绘制相关矩阵（下三角，参考期刊风格）...")


    def _calc_corr_pvals(df):
        cols = list(df.columns)
        n = len(cols)
        corr  = df.corr(method='pearson').values.astype(float)
        pvals = np.ones((n, n), dtype=float)
        for i in range(n):
            for j in range(n):
                if i == j:
                    pvals[i, j] = 0.0
                elif i < j:
                    try:
                        _, p = pearsonr(df.iloc[:, i].values, df.iloc[:, j].values)
                    except Exception:
                        p = 1.0
                    pvals[i, j] = p
                    pvals[j, i] = p
        return corr, pvals


    def _sig_mark(p):
        if p <= 0.001: return '***'
        if p <= 0.01:  return '**'
        if p <= 0.05:  return '*'
        return ''


    def _scale_size(val, vmin, vmax, min_size=0.30, max_size=0.88):
        if vmax <= vmin:
            return (min_size + max_size) / 2
        r = (val - vmin) / (vmax - vmin)
        r = max(0.0, min(1.0, r))
        return min_size + r * (max_size - min_size)


    def plot_corr_triangle(corr, pvals, labels, save_name, scheme_cmap=None):
        n = len(labels)
        cmap = plt.get_cmap('RdYlBu') if scheme_cmap is None else scheme_cmap
        norm = Normalize(vmin=-1.0, vmax=1.0)

        fig, ax = plt.subplots(figsize=(11, 11), facecolor='white')

        CELL_SIZE   = 0.88
        FRAME_COLOR = '#CCCCCC'
        FRAME_LW    = 0.8

        for i in range(n):
            for j in range(n):
                if j > i:
                    continue

                frame_x0 = j - CELL_SIZE / 2
                frame_y0 = -i - CELL_SIZE / 2
                frame = Rectangle((frame_x0, frame_y0), CELL_SIZE, CELL_SIZE,
                                  facecolor='none', edgecolor=FRAME_COLOR, linewidth=FRAME_LW)
                ax.add_patch(frame)

                r    = corr[i, j]
                p    = pvals[i, j]
                size = _scale_size(abs(r), 0.0, 1.0)
                color = cmap(norm(r))

                x0 = j - size / 2
                y0 = -i - size / 2
                rect = Rectangle((x0, y0), size, size,
                                 facecolor=color, edgecolor='white', linewidth=1.0)
                ax.add_patch(rect)

                mark = _sig_mark(p) if i != j else '***'
                if mark:
                    lightness = (color[0] * 299 + color[1] * 587 + color[2] * 114) / 1000
                    txt_color = 'white' if lightness < 0.55 else '#2C3E50'
                    ax.text(j, -i, mark, ha='center', va='center',
                            fontsize=MATRIX_FS_STAR, fontweight='bold', color=txt_color)

        for i, lab in enumerate(labels):
            x_text = i + 0.15
            y_text = -i + CELL_SIZE / 2 + 0.10
            ax.text(x_text, y_text, lab, rotation=45, ha='center', va='bottom',
                    fontsize=MATRIX_FS_DIAG_LABEL, fontweight='bold')

        ax.set_xlim(-1.2, n + 0.5)
        ax.set_ylim(-n - 0.5, 1.2)
        ax.set_aspect('equal')
        ax.set_xticks([])

        ax.set_yticks([-i for i in range(n)])
        ax.set_yticklabels(labels, fontsize=MATRIX_FS_Y_LABEL, fontweight='bold')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='y', length=0, pad=MATRIX_Y_TICK_PAD)

        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cax = fig.add_axes(CORR_CBAR_BBOX)
        cbar = plt.colorbar(sm, cax=cax, orientation='horizontal')
        cbar.set_ticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        cbar.ax.tick_params(labelsize=MATRIX_FS_CBAR_TICK)
        for lb in cbar.ax.get_xticklabels():
            lb.set_fontweight('bold')

        fig.text(0.5, CORR_LEGEND_Y,
                 '* p≤0.05     ** p≤0.01     *** p≤0.001',
                 ha='center', va='center',
                 fontsize=MATRIX_FS_CBAR_LABEL, fontweight='bold', color='#2C3E50')

        plt.subplots_adjust(left=0.10, right=0.96, top=0.92, bottom=0.26)
        save_shap_fig(save_name)
        plt.close()


    X_full_for_corr        = X_full_original.copy()
    X_full_for_corr.columns = display_names
    corr_mat, pval_mat     = _calc_corr_pvals(X_full_for_corr)

    plot_corr_triangle(corr_mat, pval_mat, display_names,
                       save_name='图11_相关矩阵_下三角_显著性.png')

    pd.DataFrame(corr_mat, index=display_names, columns=display_names).to_csv(
        os.path.join(output_path, '相关矩阵_Pearson.csv'), encoding='utf-8-sig')
    pd.DataFrame(pval_mat, index=display_names, columns=display_names).to_csv(
        os.path.join(output_path, '相关矩阵_P值.csv'), encoding='utf-8-sig')

    # ============================================================
    # 图12: SHAP 主效应 + 交互效应矩阵
    # ============================================================
    print("\n📈 计算 SHAP 交互值并绘制主+交互效应矩阵 ...")


    def compute_shap_interaction(raw_model, X_df, model_key):
        TREE_OK = {'RF', 'GBDT', 'DT', 'LightGBM', 'XGBoost'}
        if model_key not in TREE_OK:
            raise RuntimeError(f"{model_key} 不是树模型，shap_interaction_values 暂不支持。")
        expl = shap.TreeExplainer(raw_model)
        sv_int = expl.shap_interaction_values(X_df)
        if isinstance(sv_int, list):
            sv_int = sv_int[1]
        sv_int = np.asarray(sv_int)
        if sv_int.ndim == 4:
            sv_int = sv_int[..., 1]
        return sv_int


    TREE_OK_SET = {'RF', 'GBDT', 'DT', 'LightGBM', 'XGBoost'}
    if RUN_SHAP_INTERACTIONS and MODEL_KEY in TREE_OK_SET:
        try:
            sv_int_test = compute_shap_interaction(raw_model, X_test_df, MODEL_KEY)
        except Exception as _e_int:
            sv_int_test = None
            print(f"  ⚠️ 交互值计算失败（{_e_int}），跳过图12，继续输出后续CSV")
    if RUN_SHAP_INTERACTIONS and MODEL_KEY in TREE_OK_SET and sv_int_test is not None:
        abs_int_mean = np.abs(sv_int_test).mean(axis=0)

        main_effect = np.diag(abs_int_mean).copy()
        inter_effect = abs_int_mean.copy() * 2.0
        np.fill_diagonal(inter_effect, 0.0)

        main_vmin,  main_vmax  = float(main_effect.min()),  float(main_effect.max())
        inter_vals = inter_effect[np.tril_indices_from(inter_effect, k=-1)]
        inter_vmin = float(inter_vals.min()) if len(inter_vals) else 0.0
        inter_vmax = float(inter_vals.max()) if len(inter_vals) else 1.0
        if main_vmax  <= main_vmin:  main_vmax  = main_vmin  + 1e-6
        if inter_vmax <= inter_vmin: inter_vmax = inter_vmin + 1e-6

        cmap_main  = plt.get_cmap('RdYlBu_r')
        cmap_inter = plt.get_cmap('YlGnBu')
        norm_main  = Normalize(vmin=main_vmin,  vmax=main_vmax)
        norm_inter = Normalize(vmin=inter_vmin, vmax=inter_vmax)

        n = len(display_names)
        fig, ax = plt.subplots(figsize=(11, 11), facecolor='white')

        CELL_SIZE   = 0.88
        FRAME_COLOR = '#CCCCCC'
        FRAME_LW    = 0.8

        # ---- 图12 专用排版微调（保留原配色） ----
        SHAP12_XLIM_LEFT   = -0.82   # 减少左侧空白，让左侧变量名更贴近主图
        SHAP12_XLIM_RIGHT  = n + 0.95
        # 最后一行中心在 -(n-1)，不是 -n；原先写成 -n-0.45 会白白多留约1个单元格高度。
        SHAP12_YLIM_BOTTOM = -(n - 1) - CELL_SIZE / 2 - 0.08
        SHAP12_YLIM_TOP    = 1.42
        SHAP12_Y_TICK_PAD  = -10     # 左侧变量名再向主图靠近一点

        # 主图继续下移；colorbar 同时上移，二者主动靠近。
        SHAP12_SUBPLOTS = dict(left=0.10, right=0.965, top=0.895, bottom=0.145)
        SHAP12_CBAR_MAIN_BBOX  = [0.14, 0.078, 0.34, 0.024]
        SHAP12_CBAR_INTER_BBOX = [0.54, 0.078, 0.34, 0.024]

        # 斜标签：让“首字母/起始位置”大致位于对应对角方块的居中上方。
        # Intubation 额外右移，避免与主图重叠。
        SHAP12_LABEL_DX = {
            'NEU': 0.03,
            'Intubation': 0.18,
            'MV': 0.07,
            'LDH': 0.06,
            'LYM': 0.05,
            'BUN': 0.06,
            'CCI': 0.06,
            'FIB': 0.07,
            'Surgery': 0.09,
            'Diuretics': 0.11,
            'TCO2': 0.11,
        }
        SHAP12_LABEL_DY = {
            'NEU': 0.11,
            'Intubation': 0.13,
            'MV': 0.12,
            'LDH': 0.12,
            'LYM': 0.12,
            'BUN': 0.12,
            'CCI': 0.12,
            'FIB': 0.12,
            'Surgery': 0.12,
            'Diuretics': 0.12,
            'TCO2': 0.12,
        }

        for i in range(n):
            for j in range(n):
                if j > i:
                    continue

                frame_x0 = j - CELL_SIZE / 2
                frame_y0 = -i - CELL_SIZE / 2
                frame = Rectangle((frame_x0, frame_y0), CELL_SIZE, CELL_SIZE,
                                  facecolor='none', edgecolor=FRAME_COLOR, linewidth=FRAME_LW)
                ax.add_patch(frame)

                is_diag = (i == j)

                if is_diag:
                    val    = main_effect[i]
                    size   = _scale_size(val, main_vmin,  main_vmax)
                    color  = cmap_main(norm_main(val))
                else:
                    val    = inter_effect[i, j]
                    size   = _scale_size(val, inter_vmin, inter_vmax)
                    color  = cmap_inter(norm_inter(val))

                x0 = j - size / 2
                y0 = -i - size / 2
                rect = Rectangle((x0, y0), size, size,
                                 facecolor=color, edgecolor='white', linewidth=1.0)
                ax.add_patch(rect)

        for i, lab in enumerate(display_names):
            x_text = i + SHAP12_LABEL_DX.get(lab, 0.06)
            y_text = -i + CELL_SIZE / 2 + SHAP12_LABEL_DY.get(lab, 0.12)
            ax.text(x_text, y_text, lab,
                    rotation=45,
                    ha='left', va='bottom',
                    rotation_mode='anchor',
                    fontsize=MATRIX_FS_DIAG_LABEL, fontweight='bold')

        ax.set_xlim(SHAP12_XLIM_LEFT, SHAP12_XLIM_RIGHT)
        ax.set_ylim(SHAP12_YLIM_BOTTOM, SHAP12_YLIM_TOP)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([-i for i in range(n)])
        ax.set_yticklabels(display_names, fontsize=MATRIX_FS_Y_LABEL, fontweight='bold')
        for s in ['top', 'right', 'bottom', 'left']:
            ax.spines[s].set_visible(False)
        ax.tick_params(axis='y', length=0, pad=SHAP12_Y_TICK_PAD)

        sm_main  = ScalarMappable(norm=norm_main,  cmap=cmap_main);  sm_main.set_array([])
        sm_inter = ScalarMappable(norm=norm_inter, cmap=cmap_inter); sm_inter.set_array([])

        cax1 = fig.add_axes(SHAP12_CBAR_MAIN_BBOX)
        cb1  = plt.colorbar(sm_main, cax=cax1, orientation='horizontal')
        cb1.ax.tick_params(labelsize=MATRIX_FS_CBAR_TICK, pad=3)
        for lb in cb1.ax.get_xticklabels():
            lb.set_fontweight('bold')
        # 标题放在色条上方，刻度仍在下方：彻底避免标题与刻度数字重叠。
        cb1.ax.set_title('Main Effect |SHAP|',
                         fontsize=MATRIX_FS_CBAR_LABEL,
                         fontweight='bold', pad=7)

        cax2 = fig.add_axes(SHAP12_CBAR_INTER_BBOX)
        cb2  = plt.colorbar(sm_inter, cax=cax2, orientation='horizontal')
        cb2.ax.tick_params(labelsize=MATRIX_FS_CBAR_TICK, pad=3)
        for lb in cb2.ax.get_xticklabels():
            lb.set_fontweight('bold')
        cb2.ax.set_title('Interaction Effect |SHAP|',
                         fontsize=MATRIX_FS_CBAR_LABEL,
                         fontweight='bold', pad=7)

        plt.subplots_adjust(**SHAP12_SUBPLOTS)
        save_shap_fig('图12_SHAP主效应与交互效应矩阵.png')
        plt.close()

        pd.DataFrame(abs_int_mean, index=display_names, columns=display_names).to_csv(
            os.path.join(output_path, 'SHAP交互值矩阵_测试集_绝对值均值.csv'),
            encoding='utf-8-sig')
        pd.DataFrame({'Feature': display_names, 'Main_Effect_|SHAP|': main_effect}).to_csv(
            os.path.join(output_path, 'SHAP主效应_测试集.csv'),
            index=False, encoding='utf-8-sig')

        print(f"  ✅ 图12 完成（主效应范围 [{main_vmin:.3f}, {main_vmax:.3f}]，"
              f"交互效应范围 [{inter_vmin:.3f}, {inter_vmax:.3f}]）")
    else:
        print(f"  ⚠️ {MODEL_KEY} 不是树模型，跳过图12")

    # ============================================================
    # 保存 SHAP 数值数据
    # ============================================================
    print("\n📊 保存SHAP值数据...")

    shap_df_test = pd.DataFrame(shap_values_test, columns=display_names)
    shap_df_test.insert(0, ID_COL, test_ids.values)
    shap_df_test['Predicted_Prob'] = y_proba_test
    shap_df_test['Actual']         = y_test.values
    shap_df_test.to_csv(os.path.join(output_path, 'SHAP值_测试集.csv'),
                        index=False, encoding='utf-8-sig')

    importance_summary = pd.DataFrame({
        'Feature':       display_names,
        'Original_Name': feature_names,
        'Mean_|SHAP|':   np.abs(shap_values_test).mean(axis=0),
        'Std_SHAP':      shap_values_test.std(axis=0),
        'Max_|SHAP|':    np.abs(shap_values_test).max(axis=0)
    }).sort_values('Mean_|SHAP|', ascending=False)
    importance_summary.to_csv(os.path.join(output_path, 'SHAP特征重要性汇总.csv'),
                              index=False, encoding='utf-8-sig')

    print("  ✅ 数据已保存")

    print(f"\n{'=' * 70}")
    print(f"✅ SHAP可解释性分析完成！模型: {MODEL_KEY}")
    print("=" * 70)
    print("\n🎉 分析完成！")
