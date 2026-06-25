#!/usr/bin/env python3
"""Plot hit rate vs aging factor with reactive baseline per table size."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt

STEP0_DIR = Path(__file__).resolve().parents[1]
ROOT = STEP0_DIR.parents[2] / "results" / "agingfactor_tablesize_experiment_data"
OUTPUT_DIR = STEP0_DIR / "plots"

MODES = ["speculative", "speculativereactive"]
ALGORITHMS = ["bandit", "ppo", "dqn"]
ORDERINGS = ["trace", "source", "destination"]
TABLESIZES = [30, 50, 70, 90, 110]

# Distinct colors for each table size (solid speculative + dotted reactive share color)
TABLESIZE_COLORS = {
    30: "#1f77b4",
    50: "#ff7f0e",
    70: "#2ca02c",
    90: "#d62728",
    110: "#9467bd",
}


def read_hit_rate(summary_path: Path) -> float | None:
    if not summary_path.exists():
        return None
    with open(summary_path) as f:
        data = json.load(f)
    return data.get("average_hitrate_per_lti")


def load_speculative_series(
    root: Path, mode: str, algorithm: str, ordering: str, tablesize: int
) -> list[tuple[float, float]]:
    base = root / f"mode_{mode}" / f"algorithm_{algorithm}" / f"ordering_{ordering}" / f"tablesize_{tablesize}"
    if not base.exists():
        return []

    points = []
    for aging_dir in sorted(base.glob("agingfactor_*")):
        match = re.search(r"agingfactor_([\d.]+)$", aging_dir.name)
        if not match:
            continue
        hit_rate = read_hit_rate(aging_dir / "summary.json")
        if hit_rate is not None:
            points.append((float(match.group(1)), hit_rate))
    return sorted(points)


def load_reactive_hit_rate(root: Path, tablesize: int) -> float | None:
    return read_hit_rate(root / "mode_reactive" / f"tablesize_{tablesize}" / "summary.json")


def plot_one(root: Path, mode: str, algorithm: str, ordering: str, output_dir: Path) -> Path | None:
    fig, ax = plt.subplots(figsize=(10, 6))
    has_data = False

    for tablesize in TABLESIZES:
        color = TABLESIZE_COLORS[tablesize]
        series = load_speculative_series(root, mode, algorithm, ordering, tablesize)
        reactive_hr = load_reactive_hit_rate(root, tablesize)

        if series:
            xs, ys = zip(*series)
            ax.plot(
                xs,
                ys,
                color=color,
                linestyle="-",
                linewidth=2,
                marker="o",
                markersize=4,
                label=f"tablesize {tablesize} ({mode})",
            )
            has_data = True

        if reactive_hr is not None and series:
            ax.axhline(
                reactive_hr,
                color=color,
                linestyle=":",
                linewidth=2,
                label=f"tablesize {tablesize} (reactive)",
            )
            has_data = True

    if not has_data:
        plt.close(fig)
        return None

    ax.set_xlabel("Aging factor")
    ax.set_ylabel("Hit rate (%)")
    ax.set_title(f"{mode} / ordering_{ordering} / {algorithm}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, ncol=2)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{mode}_ordering_{ordering}_{algorithm}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def generate_all(root: Path, output_dir: Path) -> list[Path]:
    written = []
    for mode in MODES:
        for ordering in ORDERINGS:
            for algorithm in ALGORITHMS:
                path = plot_one(root, mode, algorithm, ordering, output_dir)
                if path:
                    written.append(path)
                    print(f"  saved {path.relative_to(root)}")
                else:
                    print(f"  skipped {mode}/ordering_{ordering}/{algorithm} (no data)")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    root = args.root.resolve()
    output_dir = args.output_dir.resolve()
    print(f"Generating plots from {root}")
    written = generate_all(root, output_dir)
    print(f"\nDone: {len(written)} plots in {output_dir}")


if __name__ == "__main__":
    main()
