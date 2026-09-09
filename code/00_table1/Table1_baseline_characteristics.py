#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Table 1 generator for the AIS-ICU pulmonary-infection study.

Purpose
-------
Reproduce the descriptive baseline/candidate-predictor table used in the revised
manuscript from the completed analysis-ready cohort.

Statistics
----------
- Continuous variables: median (IQR); two-sided Mann-Whitney U test.
- Categorical variables: n (%); Pearson chi-square with Yates continuity
  correction for 2x2 tables, except Fisher's exact test when any expected cell
  count is <5.
- All-zero categorical variables: P reported as NA.
- P values are descriptive/unadjusted.

Important data note
-------------------
The analysis-ready revision archive is already imputed/winsorized under the
original preprocessing workflow. This script summarizes that completed archive;
it does not reconstruct the original raw-data preprocessing.

Example
-------
python Table1_baseline_characteristics.py \
    --input /path/to/restricted/analysis_ready.csv \
    --output-dir ./outputs/table1
"""

from __future__ import annotations

import argparse
from pathlib import Path
import math
import numpy as np
import pandas as pd
from scipy import stats

OUTCOME = "Pulmonary_infection"
ID_COL = "Study_row_id"
EXPECTED_N = 3368
EXPECTED_EVENTS = 1357

# Exact current Table 1 order.
VARIABLES = [
    {"name": "Age", "label": "Age", "type": "continuous", "digits": 0},
    {
        "name": "Sex", "label": "Sex", "type": "categorical",
        "levels": [(1, "Male"), (2, "Female")],
    },
    {
        "name": "Admission_type", "label": "Admission type", "type": "categorical",
        "levels": [(1, "Elective"), (2, "Non-elective")],
    },
    {
        "name": "CCI", "label": "Charlson Comorbidity Index (CCI)",
        "type": "continuous", "digits": 0,
    },
    {
        "name": "Diabetes_mellitus", "label": "Diabetes mellitus (DM)",
        "type": "categorical", "levels": [(0, "No"), (1, "Yes")],
    },
    {
        "name": "Heart_disease", "label": "Heart disease", "type": "categorical",
        "levels": [(0, "No"), (1, "Yes")],
    },
    {
        "name": "Surgery", "label": "Surgery", "type": "categorical",
        "levels": [(0, "No"), (1, "Yes")],
    },
    {
        "name": "Mechanical_ventilation", "label": "Mechanical ventilation (MV)",
        "type": "categorical", "levels": [(0, "No"), (1, "Yes")],
    },
    {
        "name": "Intubation_tracheotomy",
        "label": "Endotracheal intubation/tracheotomy",
        "type": "categorical", "levels": [(0, "No"), (1, "Yes")],
    },
    {
        "name": "Central_venous_catheter", "label": "Central venous catheter (CVC)",
        "type": "categorical", "levels": [(0, "No"), (1, "Yes")],
    },
    {
        "name": "CRRT", "label": "Continuous renal replacement therapy (CRRT)",
        "type": "categorical", "levels": [(0, "No"), (1, "Yes")],
    },
    {
        "name": "Immunosuppressants", "label": "Immunosuppressants",
        "type": "categorical", "levels": [(0, "No"), (1, "Yes")],
    },
    {
        "name": "Anticholinergics", "label": "Anticholinergic agents",
        "type": "categorical", "levels": [(0, "No"), (1, "Yes")],
    },
    {
        "name": "Anticholinesterases", "label": "Cholinesterase inhibitors",
        "type": "categorical", "levels": [(0, "No"), (1, "Yes")],
    },
    {
        "name": "Benzodiazepines", "label": "Benzodiazepines",
        "type": "categorical", "levels": [(0, "No"), (1, "Yes")],
    },
    {
        "name": "Non_benzodiazepines", "label": "Non-benzodiazepine sedatives",
        "type": "categorical", "levels": [(0, "No"), (1, "Yes")],
    },
    {
        "name": "Antipsychotics", "label": "Antipsychotics",
        "type": "categorical", "levels": [(0, "No"), (1, "Yes")],
    },
    {
        "name": "Vasoactive_agents", "label": "Vasoactive agents",
        "type": "categorical", "levels": [(0, "No"), (1, "Yes")],
    },
    {
        "name": "Broad_spectrum_antibiotics", "label": "Broad-spectrum antibiotics (BSA)",
        "type": "categorical", "levels": [(0, "No"), (1, "Yes")],
    },
    {
        "name": "Diuretics", "label": "Diuretics", "type": "categorical",
        "levels": [(0, "No"), (1, "Yes")],
    },
    {
        "name": "WBC", "label": "White blood cell count (WBC), x10^9/L",
        "type": "continuous", "digits": 2,
    },
    {
        "name": "NEUT_abs", "label": "Absolute neutrophil count (NEU), x10^9/L",
        "type": "continuous", "digits": 2,
    },
    {
        "name": "LYMPH_abs", "label": "Absolute lymphocyte count (LYM), x10^9/L",
        "type": "continuous", "digits": 2,
    },
    {
        "name": "HGB", "label": "Hemoglobin (Hb), g/L",
        "type": "continuous", "digits": 2,
    },
    {
        "name": "RBC", "label": "Red blood cell count (RBC), x10^12/L",
        "type": "continuous", "digits": 2,
    },
    {
        "name": "PLT", "label": "Platelet count (PLT), x10^9/L",
        "type": "continuous", "digits": 2,
    },
    {
        "name": "INR", "label": "International normalized ratio (INR)",
        "type": "continuous", "digits": 2,
    },
    {
        "name": "FIB", "label": "Fibrinogen (FIB), g/L",
        "type": "continuous", "digits": 2,
    },
    {
        "name": "ALB", "label": "Albumin (ALB), g/L",
        "type": "continuous", "digits": 2,
    },
    {
        "name": "CREA", "label": "Serum creatinine (SCr), umol/L",
        "type": "continuous", "digits": 2,
    },
    {
        "name": "BUN", "label": "Blood urea nitrogen (BUN), mmol/L",
        "type": "continuous", "digits": 2,
    },
    {
        "name": "ALT", "label": "Alanine aminotransferase (ALT), U/L",
        "type": "continuous", "digits": 2,
    },
    {
        "name": "LDH", "label": "Lactate dehydrogenase (LDH), U/L",
        "type": "continuous", "digits": 2,
    },
    {
        "name": "HbA1c", "label": "Glycated hemoglobin (HbA1c), %",
        "type": "continuous", "digits": 2,
    },
    {
        "name": "NA", "label": "Serum sodium (Na), mmol/L",
        "type": "continuous", "digits": 2,
    },
    {
        "name": "K", "label": "Serum potassium (K), mmol/L",
        "type": "continuous", "digits": 2,
    },
    {
        "name": "CL", "label": "Serum chloride (Cl), mmol/L",
        "type": "continuous", "digits": 2,
    },
    {
        "name": "CO2", "label": "Total carbon dioxide (TCO2), mmol/L",
        "type": "continuous", "digits": 2,
    },
]


def read_csv_flexible(path: Path) -> pd.DataFrame:
    last_error = None
    for enc in ("utf-8-sig", "utf-8", "gb18030", "gbk", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise last_error or RuntimeError(f"Unable to read {path}")


def normalize_outcome(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        y = pd.to_numeric(s, errors="coerce")
    else:
        mapping = {
            "0": 0, "0.0": 0, "no": 0, "No": 0, "NO": 0,
            "1": 1, "1.0": 1, "yes": 1, "Yes": 1, "YES": 1,
        }
        y = s.astype(str).str.strip().map(mapping)
    if y.isna().any() or not set(y.unique()).issubset({0, 1}):
        raise ValueError(f"{OUTCOME} must be complete binary 0/1.")
    return y.astype(int)


def fmt_number(x: float, digits: int) -> str:
    if not np.isfinite(x):
        return "NA"
    if digits == 0:
        return str(int(round(x)))
    return f"{x:.{digits}f}"


def fmt_median_iqr(x: pd.Series, digits: int) -> str:
    v = pd.to_numeric(x, errors="coerce").dropna()
    if v.empty:
        return "NA"
    q1, med, q3 = np.percentile(v.to_numpy(dtype=float), [25, 50, 75])
    return f"{fmt_number(med, digits)} ({fmt_number(q1, digits)}, {fmt_number(q3, digits)})"


def fmt_n_pct(n: int, denominator: int) -> str:
    pct = 100.0 * n / denominator if denominator else np.nan
    return f"{n} ({pct:.1f}%)"


def format_p(p: float | None) -> str:
    if p is None or not np.isfinite(p):
        return "NA"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def continuous_test(df: pd.DataFrame, var: str) -> tuple[float, str, float]:
    g0 = pd.to_numeric(df.loc[df[OUTCOME] == 0, var], errors="coerce").dropna()
    g1 = pd.to_numeric(df.loc[df[OUTCOME] == 1, var], errors="coerce").dropna()
    result = stats.mannwhitneyu(g0, g1, alternative="two-sided", method="auto")
    return float(result.pvalue), "Mann-Whitney U", float(result.statistic)


def categorical_test(df: pd.DataFrame, var: str, levels: list[tuple[object, str]]) -> tuple[float | None, str, float | None, float | None]:
    # Explicit level order makes the result auditable and includes zero-count levels.
    tab = np.array([
        [int(((df[var] == level) & (df[OUTCOME] == 0)).sum()),
         int(((df[var] == level) & (df[OUTCOME] == 1)).sum())]
        for level, _ in levels
    ], dtype=int)

    # If only one level is present in the whole cohort, no between-group test is defined.
    if np.count_nonzero(tab.sum(axis=1)) <= 1:
        return None, "Not applicable (zero variance)", None, None

    # Remove structurally empty levels before testing.
    active = tab[tab.sum(axis=1) > 0]
    chi2, p_chi, _, expected = stats.chi2_contingency(active, correction=(active.shape == (2, 2)))
    min_expected = float(expected.min())

    if active.shape == (2, 2) and min_expected < 5:
        odds_ratio, p_fisher = stats.fisher_exact(active, alternative="two-sided")
        return float(p_fisher), "Fisher exact", float(odds_ratio), min_expected

    return float(p_chi), "Chi-square (Yates correction for 2x2)", float(chi2), min_expected


def validate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.astype(str).str.replace("\ufeff", "", regex=False).str.strip()

    required = {OUTCOME, ID_COL, *(v["name"] for v in VARIABLES)}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df[OUTCOME] = normalize_outcome(df[OUTCOME])

    if len(df) != EXPECTED_N:
        raise ValueError(f"Expected n={EXPECTED_N}, observed n={len(df)}")
    if int(df[OUTCOME].sum()) != EXPECTED_EVENTS:
        raise ValueError(f"Expected events={EXPECTED_EVENTS}, observed={int(df[OUTCOME].sum())}")
    if df[ID_COL].duplicated().any():
        raise ValueError(f"Duplicate {ID_COL} values detected.")

    return df


def build_table(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_total = len(df)
    n0 = int((df[OUTCOME] == 0).sum())
    n1 = int((df[OUTCOME] == 1).sum())

    display_rows: list[dict] = []
    audit_rows: list[dict] = []

    for spec in VARIABLES:
        var = spec["name"]
        label = spec["label"]

        if spec["type"] == "continuous":
            digits = int(spec["digits"])
            p, test, stat = continuous_test(df, var)
            display_rows.append({
                "Variable": label,
                f"Total (n={n_total})": fmt_median_iqr(df[var], digits),
                f"No pulmonary infection (n={n0})": fmt_median_iqr(df.loc[df[OUTCOME] == 0, var], digits),
                f"Pulmonary infection (n={n1})": fmt_median_iqr(df.loc[df[OUTCOME] == 1, var], digits),
                "P value": format_p(p),
            })
            audit_rows.append({
                "Variable": var, "Display_label": label, "Type": "continuous",
                "Test": test, "Statistic": stat, "P_value_raw": p,
                "Min_expected_count": np.nan,
            })
            continue

        levels = list(spec["levels"])
        p, test, stat, min_expected = categorical_test(df, var, levels)
        display_rows.append({
            "Variable": label,
            f"Total (n={n_total})": "",
            f"No pulmonary infection (n={n0})": "",
            f"Pulmonary infection (n={n1})": "",
            "P value": format_p(p),
        })
        for level, level_label in levels:
            nt = int((df[var] == level).sum())
            n_no = int(((df[var] == level) & (df[OUTCOME] == 0)).sum())
            n_yes = int(((df[var] == level) & (df[OUTCOME] == 1)).sum())
            display_rows.append({
                "Variable": f"  {level_label}",
                f"Total (n={n_total})": fmt_n_pct(nt, n_total),
                f"No pulmonary infection (n={n0})": fmt_n_pct(n_no, n0),
                f"Pulmonary infection (n={n1})": fmt_n_pct(n_yes, n1),
                "P value": "",
            })
        audit_rows.append({
            "Variable": var, "Display_label": label, "Type": "categorical",
            "Test": test, "Statistic": stat, "P_value_raw": p,
            "Min_expected_count": min_expected,
        })

    return pd.DataFrame(display_rows), pd.DataFrame(audit_rows)


def markdown_table(df: pd.DataFrame) -> str:
    # Avoid an optional tabulate dependency.
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        vals = [str(row[c]).replace("|", "\\|") for c in cols]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate revised manuscript Table 1.")
    parser.add_argument("--input", required=True, type=Path, help="Restricted analysis-ready CSV.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/table1"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = validate(read_csv_flexible(args.input))
    table, audit = build_table(df)

    table_csv = args.output_dir / "Table1_baseline_characteristics.csv"
    audit_csv = args.output_dir / "Table1_statistical_audit.csv"
    table_md = args.output_dir / "Table1_baseline_characteristics.md"

    table.to_csv(table_csv, index=False, encoding="utf-8-sig")
    audit.to_csv(audit_csv, index=False, encoding="utf-8-sig")

    note = (
        "\n\nNote. Continuous variables are presented as median (IQR), and categorical variables as n (%). "
        "Continuous variables were compared using two-sided Mann-Whitney U tests. Categorical variables "
        "were compared using chi-square tests with Yates continuity correction for 2x2 tables; Fisher's "
        "exact test was used when an expected cell count was <5. P values are unadjusted and descriptive. "
        "NA indicates that no hypothesis test was defined because the variable had zero variance.\n"
    )
    table_md.write_text(
        "# Table 1. Baseline and first-48-hour clinical characteristics\n\n"
        + markdown_table(table)
        + note,
        encoding="utf-8",
    )

    # Auditable manuscript checks.
    checks = audit.set_index("Variable")
    assert abs(float(checks.loc["K", "P_value_raw"]) - 0.4077368338693862) < 1e-12
    assert format_p(float(checks.loc["K", "P_value_raw"])) == "0.408"
    assert format_p(float(checks.loc["Heart_disease", "P_value_raw"])) == "0.328"
    assert format_p(float(checks.loc["CRRT", "P_value_raw"])) == "0.078"
    assert format_p(float(checks.loc["Anticholinesterases", "P_value_raw"])) == "0.403"

    print(f"Validated cohort: n={len(df)}, events={int(df[OUTCOME].sum())}")
    print(f"Saved: {table_csv}")
    print(f"Saved: {audit_csv}")
    print(f"Saved: {table_md}")
    print("Key audit check: potassium Mann-Whitney P=0.4077368 -> displays as 0.408")


if __name__ == "__main__":
    main()
