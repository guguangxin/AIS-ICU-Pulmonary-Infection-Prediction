# -*- coding: utf-8 -*-
"""
Created on Thu May  7 18:54:02 2026

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
HISTORICAL EXPLORATORY BSA-INCLUSIVE ANALYSIS
急性脑梗肺炎预测 - SHAP可解释性分析（LightGBM最优模型）
============================================================
"""

import warnings
warnings.filterwarnings('ignore')
import argparse
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle, FancyArrowPatch, FancyArrow
from matplotlib.cm import ScalarMappable
from matplotlib.transforms import blended_transform_factory
import shap
import joblib
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

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

_SCHEME_NAMES = {
    1:'深海青绿', 2:'暮色紫蓝', 3:'森林翡翠', 4:'极夜蓝金',
    5:'珊瑚海蓝', 6:'莫兰迪雾色', 7:'青瓷朱砂', 8:'薄荷日落', 9:'紫水晶金',
}
print(f"  🎨 当前配色方案: {SCHEME} — {_SCHEME_NAMES[SCHEME]}")

# ============================================================
# Public-repository path configuration
# ============================================================
_parser = argparse.ArgumentParser(
    description="Historical exploratory BSA-inclusive LightGBM SHAP reconstruction."
)
_parser.add_argument(
    "--data-file",
    default=os.environ.get("AIS_ICU_BSA10_DATA_FILE"),
    help="Restricted analysis dataset CSV. Alternatively set AIS_ICU_BSA10_DATA_FILE.",
)
_parser.add_argument(
    "--model-dir",
    default=os.environ.get("AIS_ICU_BSA10_MODEL_DIR", "outputs/historical_bsa_inclusive"),
    help="Directory containing the historical model/scaler artifacts from historical_BSA_10predictor_models.py.",
)
_parser.add_argument(
    "--output",
    default=os.environ.get("AIS_ICU_BSA10_SHAP_OUTPUT", "outputs/historical_bsa_shap"),
    help="Local SHAP output directory (default: outputs/historical_bsa_shap).",
)
_parser.add_argument(
    "--export-patient-level",
    action="store_true",
    help="Explicitly export patient-level SHAP values locally. Do not commit them publicly.",
)
_args = _parser.parse_args()

if not _args.data_file:
    raise SystemExit(
        "A restricted input dataset is required. Use --data-file PATH or set AIS_ICU_BSA10_DATA_FILE."
    )

data_path = os.path.abspath(os.path.expanduser(_args.model_dir))
ckpt_dir = os.path.join(data_path, "checkpoints")
data_file = os.path.abspath(os.path.expanduser(_args.data_file))
output_path = os.path.abspath(os.path.expanduser(_args.output))
EXPORT_PATIENT_LEVEL = bool(_args.export_patient_level)
os.makedirs(output_path, exist_ok=True)


def save_fig(filename):
    plt.savefig(os.path.join(output_path, filename),
                dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  📊 已保存: {filename}")


# ============================================================
# 模型与特征配置
# ============================================================
MODEL_KEY = 'LightGBM'
OUTCOME = 'Pulmonary_infection'

FEATURES_USED = [
    'Broad_spectrum_antibiotics',
    'NEUT_abs',
    'Intubation_tracheotomy',
    'Mechanical_ventilation',
    'LDH',
    'LYMPH_abs',
    'BUN',
    'CCI',
    'FIB',
    'Surgery',
]

# ============================================================
# 加载 pkl 文件
# ============================================================
print("=" * 70)
print(f"🔍 SHAP可解释性分析 — {MODEL_KEY}（仅测试集）")
print("=" * 70)

feature_names = joblib.load(os.path.join(data_path, '特征名称列表.pkl'))
scaler        = joblib.load(os.path.join(data_path, '标准化器.pkl'))

model = None
load_source = None
raw_model = None
raw_source = None

for cand in ['已训练模型字典.pkl', 'best_models_calibrated.pkl', 'best_models.pkl']:
    cand_p = os.path.join(data_path, cand)
    if os.path.exists(cand_p):
        try:
            models_dict = joblib.load(cand_p)
            if MODEL_KEY in models_dict:
                model = models_dict[MODEL_KEY]
                load_source = cand
                break
        except Exception as e:
            print(f"  ⚠️ {cand} 读取失败: {e}")

if model is None:
    ckpt_p = os.path.join(ckpt_dir, f'ckpt_{MODEL_KEY}.pkl')
    if os.path.exists(ckpt_p):
        ckpt = joblib.load(ckpt_p)
        model = ckpt.get('model')
        load_source = f'checkpoints/ckpt_{MODEL_KEY}.pkl'

if model is None:
    raise FileNotFoundError(f"❌ 找不到模型 {MODEL_KEY}。")

from sklearn.calibration import CalibratedClassifierCV
is_calibrated = isinstance(model, CalibratedClassifierCV)

if is_calibrated:
    print(f"  ℹ️ 检测到 CalibratedClassifierCV 包装，正在加载未校准的基模型用于 SHAP 分析...")

    raw_p = os.path.join(data_path, 'raw_models_uncalibrated.pkl')
    if os.path.exists(raw_p):
        try:
            raw_models = joblib.load(raw_p)
            if MODEL_KEY in raw_models:
                raw_model = raw_models[MODEL_KEY]
                raw_source = 'raw_models_uncalibrated.pkl'
        except Exception as e:
            print(f"  ⚠️ raw_models_uncalibrated.pkl 读取失败: {e}")

    if raw_model is None:
        ckpt_p = os.path.join(ckpt_dir, f'ckpt_{MODEL_KEY}.pkl')
        if os.path.exists(ckpt_p):
            ckpt = joblib.load(ckpt_p)
            raw_model = ckpt.get('raw_model')
            if raw_model is not None:
                raw_source = f'checkpoints/ckpt_{MODEL_KEY}.pkl [raw_model]'

    if raw_model is None:
        try:
            first_cc = model.calibrated_classifiers_[0]
            raw_model = getattr(first_cc, 'estimator', None) or getattr(first_cc, 'base_estimator', None)
            if raw_model is not None:
                raw_source = 'CalibratedClassifierCV.calibrated_classifiers_[0].estimator'
                print(f"  ⚠️ 未找到独立保存的基模型，使用 CalibratedClassifierCV 内第一折的基模型作为近似")
        except Exception as e:
            print(f"  ⚠️ 从 CalibratedClassifierCV 内部提取基模型失败: {e}")

    if raw_model is None:
        raise RuntimeError(f"❌ 找不到 {MODEL_KEY} 的未校准基模型。")
else:
    raw_model = model
    raw_source = '(same as loaded model — not calibrated)'

print(f"  ✅ 已加载最终模型: {MODEL_KEY}  (来源: {load_source})")
print(f"     校准状态      : {'已校准 (CalibratedClassifierCV)' if is_calibrated else '未校准 (基模型)'}")
print(f"  ✅ SHAP用基模型  : (来源: {raw_source})")
print(f"  特征名({len(feature_names)}): {feature_names}")

# ============================================================
# 重建测试集
# ============================================================
print("\n📂 从原始数据重建测试集...")

data_raw = pd.read_csv(data_file, encoding='utf-8-sig')
data_raw.columns = data_raw.columns.str.replace('\ufeff', '').str.strip()

if 'Study_row_id' in data_raw.columns:
    _study_row_id_all = data_raw['Study_row_id'].copy()
else:
    _study_row_id_all = pd.Series(np.arange(1, len(data_raw) + 1), index=data_raw.index, name='Study_row_id')

missing_cols = [c for c in FEATURES_USED + [OUTCOME] if c not in data_raw.columns]
if missing_cols:
    raise KeyError(f"❌ 原始数据中缺失列: {missing_cols}")

data_raw = data_raw[[OUTCOME] + FEATURES_USED].copy()

if data_raw[OUTCOME].dtype == object:
    data_raw[OUTCOME] = data_raw[OUTCOME].map(
        {'No':0,'Yes':1,'no':0,'yes':1,'0':0,'1':1,0:0,1:1}
    )

X_raw = data_raw[FEATURES_USED].copy()
y     = data_raw[OUTCOME].astype(int)

binary_features     = []
continuous_features = []
for col in X_raw.columns:
    if X_raw[col].nunique() <= 5 or X_raw[col].dtype == object:
        binary_features.append(col)
    else:
        continuous_features.append(col)

for col in continuous_features:
    if X_raw[col].isna().any():
        X_raw[col] = X_raw[col].fillna(X_raw[col].median())
for col in binary_features:
    if X_raw[col].isna().any():
        X_raw[col] = X_raw[col].fillna(X_raw[col].mode()[0])

_, X_test_raw, _, y_test = train_test_split(
    X_raw, y, test_size=0.3, random_state=42, stratify=y
)

existing_continuous = [f for f in continuous_features if f in X_test_raw.columns]
X_test_s = X_test_raw.copy()
if existing_continuous:
    X_test_s[existing_continuous] = scaler.transform(X_test_raw[existing_continuous])

X_test_s = X_test_s[feature_names]
X_test_scaled = X_test_s.values

X_test_df       = pd.DataFrame(X_test_scaled, columns=feature_names)
X_test_original = X_test_df.copy()
if existing_continuous:
    X_test_original[existing_continuous] = scaler.inverse_transform(
        X_test_df[existing_continuous]
    )

y_test = y_test.reset_index(drop=True)

n_features = len(feature_names)
BAR_COLORS = BAR_COLORS_BASE[:n_features]

print(f"  测试集: {X_test_df.shape[0]} 样本, {n_features} 特征")
print(f"  特征列表: {feature_names}")
print(f"  连续变量 ({len(existing_continuous)}个): {existing_continuous}")
print(f"  二分类变量 ({len(binary_features)}个): {binary_features}")
print(f"  ✅ 测试集重建完成")

X_full_original = X_raw[feature_names].copy().reset_index(drop=True)
y_full          = y.reset_index(drop=True)
print(f"  全量数据（训练+测试，用于相关矩阵）: {X_full_original.shape[0]} 样本")

# ─── 特征显示名称映射 ───
DISPLAY_NAME_MAP = {
    'Broad_spectrum_antibiotics': 'BSA',
    'NEUT_abs':                   'NEU',
    'Intubation_tracheotomy':     'Intubation',
    'Mechanical_ventilation':     'MV',
    'LYMPH_abs':                  'LYM',
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
y_proba_test  = model.predict_proba(X_test_scaled)[:, 1]

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
save_fig('图1_SHAP蜂群图_特征对模型输出的影响.png')
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
save_fig('图2_SHAP特征重要性排序.png')
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
save_fig('图3_SHAP小提琴图_特征值分布.png')
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
save_fig(f'图4_SHAP散点图_全部{n_features}个特征.png')
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
    save_fig(f'图5_SHAP瀑布图_{cn_label}.png')
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
    save_fig(f'图6_SHAP力图_{cn_label}.png')
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
save_fig('图7_SHAP交互作用图_前5个重要特征.png')
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
save_fig('图8_SHAP特征重要性_肺炎组vs无肺炎组对比.png')
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
save_fig('图9_SHAP热力图_高低风险样本对比.png')
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
save_fig('图10_SHAP决策图_样本决策路径.png')
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
save_fig('图10b_SHAP决策图_概率空间.png')
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
    save_fig(save_name)
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
if MODEL_KEY in TREE_OK_SET:
    sv_int_test = compute_shap_interaction(raw_model, X_test_df, MODEL_KEY)
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
        x_text = i + 0.15
        y_text = -i + CELL_SIZE / 2 + 0.10
        ax.text(x_text, y_text, lab, rotation=45, ha='center', va='bottom',
                fontsize=MATRIX_FS_DIAG_LABEL, fontweight='bold')

    ax.set_xlim(-1.2, n + 0.5)
    ax.set_ylim(-n - 0.5, 1.2)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([-i for i in range(n)])
    ax.set_yticklabels(display_names, fontsize=MATRIX_FS_Y_LABEL, fontweight='bold')
    for s in ['top', 'right', 'bottom', 'left']:
        ax.spines[s].set_visible(False)
    ax.tick_params(axis='y', length=0, pad=MATRIX_Y_TICK_PAD)

    sm_main  = ScalarMappable(norm=norm_main,  cmap=cmap_main);  sm_main.set_array([])
    sm_inter = ScalarMappable(norm=norm_inter, cmap=cmap_inter); sm_inter.set_array([])

    cax1 = fig.add_axes(SHAP_CBAR_MAIN_BBOX)
    cb1  = plt.colorbar(sm_main,  cax=cax1, orientation='horizontal')
    cb1.ax.tick_params(labelsize=MATRIX_FS_CBAR_TICK)
    for lb in cb1.ax.get_xticklabels(): lb.set_fontweight('bold')
    fig.text(SHAP_CBAR_MAIN_BBOX[0] + SHAP_CBAR_MAIN_BBOX[2] / 2, SHAP_CBAR_LABEL_Y,
             'Main Effect |SHAP|',
             ha='center', va='center',
             fontsize=MATRIX_FS_CBAR_LABEL, fontweight='bold')

    cax2 = fig.add_axes(SHAP_CBAR_INTER_BBOX)
    cb2  = plt.colorbar(sm_inter, cax=cax2, orientation='horizontal')
    cb2.ax.tick_params(labelsize=MATRIX_FS_CBAR_TICK)
    for lb in cb2.ax.get_xticklabels(): lb.set_fontweight('bold')
    fig.text(SHAP_CBAR_INTER_BBOX[0] + SHAP_CBAR_INTER_BBOX[2] / 2, SHAP_CBAR_LABEL_Y,
             'Interaction Effect |SHAP|',
             ha='center', va='center',
             fontsize=MATRIX_FS_CBAR_LABEL, fontweight='bold')

    plt.subplots_adjust(left=0.10, right=0.96, top=0.92, bottom=0.26)
    save_fig('图12_SHAP主效应与交互效应矩阵.png')
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

if EXPORT_PATIENT_LEVEL:
    shap_df_test = pd.DataFrame(shap_values_test, columns=display_names)
    shap_df_test.insert(0, 'Study_row_id', _study_row_id_all.loc[X_test_raw.index].to_numpy())
    shap_df_test['Predicted_Prob'] = y_proba_test
    shap_df_test['Actual']         = y_test.values
    shap_df_test.to_csv(os.path.join(output_path, 'restricted_SHAP_values_test.csv'),
                        index=False, encoding='utf-8-sig')
else:
    print("  🔒 Patient-level SHAP-value export skipped (use --export-patient-level only for local restricted QA).")

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