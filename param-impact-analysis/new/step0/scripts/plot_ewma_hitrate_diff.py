#!/usr/bin/env python3
"""Plot EWMA(reactive) - EWMA(speculative) hit rate difference at a tablesize."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

STEP0_DIR = Path(__file__).resolve().parents[1]
ROOT = STEP0_DIR.parents[2] / "results" / "agingfactor_tablesize_experiment_data"
OUTPUT_DIR = STEP0_DIR / "plots"


def ewma(series: pd.Series, alpha: float) -> pd.Series:
    return series.ewm(alpha=alpha, adjust=False).mean()


def find_best_config(root: Path, mode: str, tablesize: int) -> tuple[Path, float, dict]:
    best_hr = None
    best_dir = None
    best_meta = {}

    pattern = re.compile(
        rf"mode_{mode}/algorithm_(?P<algorithm>[^/]+)/ordering_(?P<ordering>[^/]+)/tablesize_{tablesize}/agingfactor_(?P<agingfactor>[^/]+)$"
    )

    for summary_path in (root / f"mode_{mode}").rglob(f"tablesize_{tablesize}/agingfactor_*/summary.json"):
        with open(summary_path) as f:
            hr = json.load(f)["average_hitrate_per_lti"]
        if best_hr is None or hr > best_hr:
            best_hr = hr
            best_dir = summary_path.parent
            m = pattern.search(str(best_dir.relative_to(root)))
            if m:
                best_meta = m.groupdict()

    if best_dir is None:
        raise FileNotFoundError(f"No {mode} results for tablesize {tablesize}")

    return best_dir, best_hr, best_meta


def load_ewma(lti_path: Path, alpha: float) -> pd.DataFrame:
    df = pd.read_csv(lti_path)
    df["ewma_hitrate"] = ewma(df["hit_rate"], alpha)
    return df


def plot_diff(tablesize: int, alpha: float, start_time: float, output_dir: Path) -> Path:
    sp_dir, sp_hr, sp_meta = find_best_config(ROOT, "speculative", tablesize)

    reactive_path = ROOT / "mode_reactive" / f"tablesize_{tablesize}" / "lti_metrics.csv"
    with open(reactive_path.parent / "summary.json") as f:
        re_hr = json.load(f)["average_hitrate_per_lti"]

    sp = load_ewma(sp_dir / "lti_metrics.csv", alpha)
    reactive = load_ewma(reactive_path, alpha)

    diff = reactive["ewma_hitrate"].values - sp["ewma_hitrate"].values
    avg_diff = re_hr - sp_hr
    time = reactive["lti_start_time"]
    mask = time >= start_time

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(time[mask], diff[mask], color="#9467bd", linewidth=2)
    ax.axhline(avg_diff, color="#9467bd", linestyle=":", linewidth=2)
    ax.axhline(0, color="black", linestyle="-", linewidth=0.8, alpha=0.4)

    ax.set_xlabel("Simulation time (s)")
    ax.set_ylabel("Hit rate EWMA diff (%)")
    ax.set_xlim(start_time, time.iloc[-1])
    ax.set_title(
        f"Reactive − speculative EWMA (α={alpha}), tablesize={tablesize}\n"
        f"({sp_meta['algorithm']}, {sp_meta['ordering']}, af={sp_meta['agingfactor']})"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"tablesize_{tablesize}_reactive_minus_speculative_ewma.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)

    print(f"tablesize={tablesize}, α={alpha}:")
    print(f"  speculative: {sp_dir.relative_to(ROOT)} ({sp_hr:.2f}%)")
    print(f"  reactive:    {re_hr:.2f}%")
    print(f"  avg diff (reactive − speculative): {avg_diff:.2f} pp")
    print(f"  saved: {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tablesize", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--start-time", type=float, default=10.0, help="Plot from this simulation time (s)")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    plot_diff(args.tablesize, args.alpha, args.start_time, args.output_dir.resolve())


if __name__ == "__main__":
    main()
