#!/usr/bin/env python3
"""Plot main mean±std curves or independent single-Seed ablation curves."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

PANELS = (
    ("ordinary_mean_return", "Ordinary mean return", False),
    ("ordinary_success_rate", "Ordinary success rate", False),
    ("ordinary_safe_success_rate", "Ordinary safe-success rate", False),
    ("ordinary_mean_cost", "Ordinary mean cost: log(1+C)", True),
    ("ordinary_collision_rate", "Ordinary collision rate", False),
    ("hybrid_intervention_rate", "Hybrid intervention rate", False),
)
LABELS = {
    "flow_bc": "Flow-BC",
    "gaussian": "Gaussian Guardian-DSAC",
    "flow_q_only": "Flow-Q-only",
    "full_flow": "Guardian-Flow-DSAC",
    "flow_no_p": "Full Flow without P",
    "flow_no_correction": "Full Flow without correction",
}
COLORS = {
    "flow_bc": "#7f7f7f",
    "gaussian": "#1f77b4",
    "flow_q_only": "#ff7f0e",
    "full_flow": "#2ca02c",
    "flow_no_p": "#d62728",
    "flow_no_correction": "#9467bd",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with args.input.expanduser().resolve().open(newline="", encoding="utf-8") as stream:
        all_rows = list(csv.DictReader(stream))
    rows = [row for row in all_rows if row["row_type"] == "aggregate"]
    if not rows:
        rows = [row for row in all_rows if row["row_type"] == "run"]
    if not rows:
        raise ValueError("learning curve CSV contains no aggregate rows")
    methods = [
        method for method in LABELS if any(row["method"] == method for row in rows)
    ]
    figure, axes = plt.subplots(3, 2, figsize=(12.0, 12.5), constrained_layout=True)
    for axis, (metric, title, log_cost) in zip(axes.flat, PANELS):
        for method in methods:
            selected = sorted(
                (row for row in rows if row["method"] == method),
                key=lambda row: int(row["step"]),
            )
            x = np.asarray([int(row["step"]) for row in selected], np.float64)
            mean = np.asarray([float(row[metric]) for row in selected], np.float64)
            std = np.asarray(
                [float(row.get(metric + "_std") or 0.0) for row in selected],
                np.float64,
            )
            if log_cost:
                lower = np.log1p(np.maximum(mean - std, 0.0))
                upper = np.log1p(np.maximum(mean + std, 0.0))
                mean = np.log1p(np.maximum(mean, 0.0))
            else:
                lower, upper = mean - std, mean + std
            axis.plot(
                x,
                mean,
                marker="o",
                markersize=3,
                linewidth=1.8,
                label=LABELS[method],
                color=COLORS[method],
            )
            if np.any(std > 0.0):
                axis.fill_between(x, lower, upper, alpha=0.18, color=COLORS[method])
        axis.set_title(title)
        axis.set_xlabel("Environment steps")
        axis.grid(True, alpha=0.25)
        axis.ticklabel_format(axis="x", style="sci", scilimits=(6, 6))
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="upper center", ncol=max(len(labels), 1), frameon=False
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, format="pdf", bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
