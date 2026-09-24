import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore')

FEATURES = [
    'NEUT_abs', 'Intubation_tracheotomy', 'Mechanical_ventilation',
    'LDH', 'LYMPH_abs', 'BUN', 'CCI', 'FIB', 'Surgery', 'Diuretics', 'CO2'
]


def calibration_intercept_slope(y_true, y_proba, eps=1e-6):
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(y_proba, dtype=float), eps, 1 - eps)
    lp = logit(p)

    # slope: y ~ intercept + slope * logit(p)
    try:
        lr = LogisticRegression(penalty=None, solver='lbfgs', max_iter=2000)
    except TypeError:
        lr = LogisticRegression(C=1e12, solver='lbfgs', max_iter=2000)
    try:
        lr.fit(lp.reshape(-1, 1), y)
    except Exception:
        lr = LogisticRegression(C=1e12, solver='lbfgs', max_iter=2000)
        lr.fit(lp.reshape(-1, 1), y)
    slope = float(lr.coef_[0, 0])

    # intercept: slope fixed at 1
    def neg_ll(b):
        eta = lp + b[0]
        log_p = -np.logaddexp(0, -eta)
        log_1mp = -np.logaddexp(0, eta)
        return -(y * log_p + (1 - y) * log_1mp).sum()

    intercept = float(minimize(neg_ll, [0.0], method='L-BFGS-B').x[0])
    return intercept, slope


def auc_bootstrap_ci(y, p, n_boot=5000, seed=20260923):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        vals.append(roc_auc_score(yy, p[idx]))
    return np.quantile(vals, [0.025, 0.975])


def prepare_xy(train_df, test_df):
    continuous = [
        c for c in FEATURES
        if train_df[c].nunique(dropna=True) > 5 and train_df[c].dtype != object
    ]
    Xtr = train_df[FEATURES].copy()
    Xte = test_df[FEATURES].copy()
    scaler = MinMaxScaler()
    Xtr.loc[:, continuous] = scaler.fit_transform(Xtr[continuous])
    Xte.loc[:, continuous] = scaler.transform(Xte[continuous])
    ytr = train_df['Pulmonary_infection'].astype(int).to_numpy()
    yte = test_df['Pulmonary_infection'].astype(int).to_numpy()
    return Xtr.to_numpy(), Xte.to_numpy(), ytr, yte


def fit_fixed_spec(train_df, test_df):
    Xtr, Xte, ytr, yte = prepare_xy(train_df, test_df)
    models = {
        'LR': LogisticRegression(
            random_state=42, max_iter=3000, C=0.2493730,
            penalty='l2', solver='lbfgs'
        ),
        'GBDT': GradientBoostingClassifier(
            random_state=42, n_estimators=194, max_depth=5,
            learning_rate=0.01, min_samples_split=30,
            min_samples_leaf=20, subsample=0.7, max_features='log2'
        ),
    }
    results = {}
    for name, base in models.items():
        try:
            cal = CalibratedClassifierCV(estimator=base, method='sigmoid', cv=10, n_jobs=1)
        except TypeError:
            cal = CalibratedClassifierCV(base_estimator=base, method='sigmoid', cv=10, n_jobs=1)
        cal.fit(Xtr, ytr)
        p = cal.predict_proba(Xte)[:, 1]
        results[name] = (yte, p)
    return results


def metric_row(name, y, p, train_n, train_events, n_boot=5000):
    ci = auc_bootstrap_ci(y, p, n_boot=n_boot)
    cal_i, cal_s = calibration_intercept_slope(y, p)
    return {
        'Model': name,
        'Train_N': int(train_n),
        'Train_events': int(train_events),
        'Test_N': int(len(y)),
        'Test_events': int(np.sum(y)),
        'AUC': float(roc_auc_score(y, p)),
        'AUC_CI_low': float(ci[0]),
        'AUC_CI_high': float(ci[1]),
        'AP': float(average_precision_score(y, p)),
        'Brier': float(brier_score_loss(y, p)),
        'Cal_intercept': float(cal_i),
        'Cal_slope': float(cal_s),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--analysis', default='BSAfree_feature_selection_input_exactsplit.csv')
    ap.add_argument('--timing', default='data5.csv')
    ap.add_argument('--outdir', default='day3_sensitivity_output')
    ap.add_argument('--bootstrap', type=int, default=5000)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    a = pd.read_csv(args.analysis)
    t = pd.read_csv(args.timing)

    # Basic identity checks
    assert len(a) == 3368 and len(t) == 3368
    assert a['Study_row_id'].is_unique and t['ID'].is_unique

    for c in ['ICU_admit_datetime', 'Pneumonia_datetime', 'ICU_discharge_datetime']:
        t[c] = pd.to_datetime(t[c], errors='coerce')

    # Merge only the timing-derived day-3 event flag into the locked analysis-ready matrix
    t['elapsed_h'] = (t['Pneumonia_datetime'] - t['ICU_admit_datetime']).dt.total_seconds() / 3600.0
    t['day3_event'] = (
        t['Outcome'].eq(1)
        & t['elapsed_h'].gt(48)
        & t['elapsed_h'].lt(72)
    )

    # Verify all positive events are strictly post-landmark and before ICU exit
    pos = t.loc[t['Outcome'].eq(1)].copy()
    assert pos['Pneumonia_datetime'].notna().all()
    assert (pos['elapsed_h'] > 48).all()
    assert (pos['Pneumonia_datetime'] < pos['ICU_discharge_datetime']).all()

    # Event timing summary
    pos['index_group'] = np.select(
        [
            pos['elapsed_h'].gt(48) & pos['elapsed_h'].lt(72),
            pos['elapsed_h'].ge(72) & pos['elapsed_h'].lt(96),
            pos['elapsed_h'].ge(96),
        ],
        ['ICU day 3', 'ICU day 4', 'ICU day 5+'],
        default='Other'
    )
    rows = []
    for grp in ['ICU day 3', 'ICU day 4', 'ICU day 5+']:
        n = int((pos['index_group'] == grp).sum())
        rows.append({
            'Item': 'Pneumonia index timing', 'Category': grp, 'N': n,
            'Percent_of_1357_events': 100.0 * n / len(pos),
            'Event_count': np.nan, 'Event_percent_within_stratum': np.nan
        })

    rem = t['ICU_LOS_days'] - 2
    for label, mask in [
        ('Exactly 0 recorded remaining ICU days', rem.eq(0)),
        ('0-1 recorded remaining ICU days', rem.ge(0) & rem.le(1)),
    ]:
        ev = int((mask & t['Outcome'].eq(1)).sum())
        rows.append({
            'Item': 'Observation opportunity', 'Category': label,
            'N': int(mask.sum()), 'Percent_of_1357_events': np.nan,
            'Event_count': ev,
            'Event_percent_within_stratum': 100.0 * ev / int(mask.sum())
        })

    temporal = pd.DataFrame(rows)
    temporal.to_csv(outdir / 'temporal_separation_summary.csv', index=False)

    # Locked analysis-ready matrix and split checks
    a = a.merge(
        t[['ID', 'day3_event']],
        left_on='Study_row_id', right_on='ID', validate='one_to_one'
    )
    tr = a[a['Primary_split'].str.lower().eq('train')].copy()
    te = a[a['Primary_split'].str.lower().eq('test')].copy()
    assert len(tr) == 2357 and int(tr['Pulmonary_infection'].sum()) == 950
    assert len(te) == 1011 and int(te['Pulmonary_infection'].sum()) == 407
    assert not tr[FEATURES].isna().any().any()
    assert not te[FEATURES].isna().any().any()

    # Primary reproduction (sanity check)
    primary = fit_fixed_spec(tr, te)
    primary_rows = []
    for name, (y, p) in primary.items():
        ci, cs = calibration_intercept_slope(y, p)
        primary_rows.append({
            'Model': name, 'Scenario': 'Primary reproduction',
            'Train_N': len(tr), 'Train_events': int(tr['Pulmonary_infection'].sum()),
            'Test_N': len(te), 'Test_events': int(te['Pulmonary_infection'].sum()),
            'AUC': roc_auc_score(y, p), 'AP': average_precision_score(y, p),
            'Brier': brier_score_loss(y, p), 'Cal_intercept': ci, 'Cal_slope': cs
        })

    # Reviewer-requested sensitivity: remove day-3 positive events from both original Train and Test,
    # retain the 11-predictor specification and previously selected hyperparameters,
    # then refit scaler + model + 10-fold Platt calibration on the reduced Train partition.
    tr2 = tr.loc[~tr['day3_event']].copy()
    te2 = te.loc[~te['day3_event']].copy()
    assert len(tr2) == 1990 and int(tr2['Pulmonary_infection'].sum()) == 583
    assert len(te2) == 852 and int(te2['Pulmonary_infection'].sum()) == 248

    sens = fit_fixed_spec(tr2, te2)
    sens_rows = []
    for name, (y, p) in sens.items():
        sens_rows.append(metric_row(
            name, y, p, len(tr2), int(tr2['Pulmonary_infection'].sum()),
            n_boot=args.bootstrap
        ))
    sens_df = pd.DataFrame(sens_rows)
    sens_df.to_csv(outdir / 'day3_exclusion_sensitivity_summary.csv', index=False)

    # Optional locked-prediction subset check: original primary model predictions, only test day-3 events removed.
    keep = ~te['day3_event'].to_numpy()
    locked_rows = []
    for name, (y, p) in primary.items():
        yy, pp = y[keep], p[keep]
        ci, cs = calibration_intercept_slope(yy, pp)
        locked_rows.append({
            'Model': name, 'Scenario': 'Primary predictions; day-3-positive test cases removed',
            'Train_N': len(tr), 'Train_events': int(tr['Pulmonary_infection'].sum()),
            'Test_N': len(yy), 'Test_events': int(yy.sum()),
            'AUC': roc_auc_score(yy, pp), 'AP': average_precision_score(yy, pp),
            'Brier': brier_score_loss(yy, pp), 'Cal_intercept': ci, 'Cal_slope': cs
        })

    compare = pd.DataFrame(primary_rows + locked_rows)
    compare.to_csv(outdir / 'primary_and_locked_subset_check.csv', index=False)

    print('\nTemporal summary:')
    print(temporal.to_string(index=False))
    print('\nDay-3 exclusion refit sensitivity:')
    print(sens_df.to_string(index=False))
    print('\nPrimary/locked subset checks:')
    print(compare.to_string(index=False))


if __name__ == '__main__':
    main()
