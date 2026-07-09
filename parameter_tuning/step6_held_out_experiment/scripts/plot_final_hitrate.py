#!/usr/bin/env python3
"""Plot Hit Rate(%) vs Time (s) for step6 held-out experiment modes.

Uses the same styling and series layout as
``results/step4_some_other_baselines_and_optimals/conclusion/plot_hitrate_vs_time.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

STEP6_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = STEP6_DIR.parents[1] / "results" / "step6_held_out_experiment"
PLOTS_DIR = STEP6_DIR / "plots"

EWMA_ALPHA = 0.05
LINE_WIDTH = 1.6
AVG_LINE_WIDTH = 0.8
AVG_DASH_PATTERN = (0, (8, 2))

MODES = [
    {
        "dir": "mode_reactive",
        "label": "Reactive",
        "color": "#FF5500",
    },
    {
        "dir": "mode_speculative",
        "label": "Speculative-Only(RL)",
        "color": "#00CC44",
    },
    {
        "dir": "mode_speculativereactive",
        "label": "Speculative+Reactive(RL)",
        "color": "#0055FF",
        "linewidth": LINE_WIDTH * 2,
        "avg_linewidth": AVG_LINE_WIDTH * 2,
        "zorder": 5,
    },
    {
        "dir": "mode_heuristicspeculativereactive",
        "label": "Speculative+Reactive (Heuristic)",
        "color": "#CC00FF",
    },
    {
        "dir": "mode_reactiveoptimal",
        "label": "Reactive Optimal",
        "color": "#00BFBF",
    },
    {
        "dir": "mode_speculativereactiveoptimal",
        "label": "Speculative+Reactive Optimal",
        "color": "#FFAA00",
    },
]

plt.rcParams.update(
    {
        "font.size": 18,
        "axes.titlesize": 20,
        "axes.titleweight": "bold",
        "axes.labelsize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 14,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "black",
        "grid.color": "#dddddd",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def ewma(series: pd.Series, alpha: float = EWMA_ALPHA) -> pd.Series:
    return series.ewm(alpha=alpha, adjust=False).mean()


def load_series(csv_path: Path) -> tuple[pd.DataFrame, float]:
    df = pd.read_csv(csv_path)
    if "second_bucket" in df.columns:
        time_col = "second_bucket"
        hr_col = "hitrate"
    else:
        time_col = "lti_start_time"
        hr_col = "hit_rate"
    raw_hr = df[hr_col].astype(float)
    out = pd.DataFrame(
        {
            "time": df[time_col].astype(float),
            "hit_rate": ewma(raw_hr),
        }
    )
    return out, float(raw_hr.mean())


def load_avg_hitrate(mode_dir: Path, csv_mean: float) -> float:
    summary_path = mode_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        return float(summary["average_hitrate_per_lti"])
    return csv_mean


def plot_hitrate_vs_time(data_root: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10.0, 5.5))

    for mode in MODES:
        mode_dir = data_root / mode["dir"]
        csv_path = mode_dir / "lti_metrics.csv"
        if not csv_path.exists():
            print(f"SKIP missing: {csv_path}")
            continue

        series, csv_mean = load_series(csv_path)
        avg_hr = load_avg_hitrate(mode_dir, csv_mean)
        legend_label = f"{mode['label']} ({avg_hr:.2f}%)"
        line_w = mode.get("linewidth", LINE_WIDTH)
        avg_line_w = mode.get("avg_linewidth", AVG_LINE_WIDTH)
        zorder = mode.get("zorder", 3)
        ax.plot(
            series["time"],
            series["hit_rate"],
            color=mode["color"],
            linewidth=line_w,
            label=legend_label,
            zorder=zorder,
        )
        ax.axhline(
            avg_hr,
            color=mode["color"],
            linestyle=AVG_DASH_PATTERN,
            linewidth=avg_line_w,
            zorder=zorder - 1,
        )
        print(f"{mode['label']}: avg_per_lti={avg_hr:.2f}%  csv_mean={csv_mean:.2f}%")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Hit Rate(%)")
    ax.set_xlim(left=200)
    ax.set_ylim(10, 98)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", alpha=0.6, linestyle="-", linewidth=0.6)
    ax.grid(False, axis="x")
    ax.legend(loc="lower right", frameon=True, framealpha=0.95, edgecolor="#cccccc")

    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.10, top=0.92)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out_base = PLOTS_DIR / "final_hitrate"
    fig.savefig(out_base.with_suffix(".png"), dpi=180, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    return out_base


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Root directory containing aggregated mode_* folders",
    )
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    out_base = plot_hitrate_vs_time(data_root)
    print(f"\nWrote {out_base}.png")
    print(f"Wrote {out_base}.pdf")


if __name__ == "__main__":
    main()
