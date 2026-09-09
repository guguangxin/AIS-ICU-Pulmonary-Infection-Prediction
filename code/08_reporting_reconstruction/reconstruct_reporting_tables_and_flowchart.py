#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reconstruct aggregate reporting tables and Supplementary Figure S1.

This module intentionally uses only version-controlled aggregate revision metadata.
It does not contain or require participant-level records.

Covered outputs:
- Supplementary Tables S1, S3, S4, S16, S17, S24, S29, S31
- Supplementary Figure S1

Other manuscript tables/figures are generated from the analysis modules listed in
`docs/table_figure_code_map.md`.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import pandas as pd


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", s).strip("_")


def markdown_table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        vals = [str(row[c]).replace("|", "\\|") for c in df.columns]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_tables(meta: dict, out_dir: Path) -> None:
    for key, item in meta["tables"].items():
        df = pd.DataFrame(item["rows"], columns=item["columns"])
        stem = f"Table_{key}_{slug(item['title'])[:70]}"
        csv_path = out_dir / f"{stem}.csv"
        md_path = out_dir / f"{stem}.md"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        md = f"# Table {key}. {item['title']}\n\n{markdown_table(df)}\n\nNote. {item['note']}\n"
        md_path.write_text(md, encoding="utf-8")
        print("Saved:", csv_path)
        print("Saved:", md_path)


def draw_flowchart(meta: dict, out_dir: Path) -> None:
    info = meta["figure_S1"]
    stages = info["stages"]
    fig, ax = plt.subplots(figsize=(11, 14))
    ax.set_axis_off()
    y_positions = [0.93 - i * 0.16 for i in range(len(stages))]
    box_w, box_h = 0.60, 0.085
    x = 0.08

    for i, (stage, y) in enumerate(zip(stages, y_positions)):
        box = FancyBboxPatch(
            (x, y - box_h / 2), box_w, box_h,
            boxstyle="round,pad=0.012,rounding_size=0.01",
            linewidth=1.3, facecolor="white"
        )
        ax.add_patch(box)
        ax.text(
            x + box_w / 2, y,
            f"{stage['label']}\n(n={stage['remaining']:,})",
            ha="center", va="center", fontsize=10
        )

        if i > 0:
            prev_y = y_positions[i - 1]
            ax.annotate(
                "", xy=(x + box_w / 2, y + box_h / 2),
                xytext=(x + box_w / 2, prev_y - box_h / 2),
                arrowprops=dict(arrowstyle="->", linewidth=1.2)
            )
            if stage["excluded"] is not None:
                ax.text(
                    0.74, (prev_y + y) / 2,
                    f"Excluded: {stage['excluded_label']}\n(n={stage['excluded']:,})",
                    ha="left", va="center", fontsize=9
                )

    ax.text(0.5, 0.995, f"Figure S1. {info['title']}", ha="center", va="top", fontsize=13, fontweight="bold")
    ax.text(0.08, 0.02, info["note"], ha="left", va="bottom", fontsize=8, wrap=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    path = out_dir / "Figure_S1_patient_inclusion_flowchart.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path(__file__).with_name("reporting_metadata.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/reporting_reconstruction"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    meta = json.loads(args.metadata.read_text(encoding="utf-8"))
    write_tables(meta, args.output_dir)
    draw_flowchart(meta, args.output_dir)


if __name__ == "__main__":
    main()
