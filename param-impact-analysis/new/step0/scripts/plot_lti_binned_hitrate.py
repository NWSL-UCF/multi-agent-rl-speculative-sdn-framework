#!/usr/bin/env python3
"""Plot hit rate averaged over fixed-size LTI windows for best modes vs reactive."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STEP0_DIR = Path(__file__).resolve().parents[1]
ROOT = STEP0_DIR.parents[2] / "results" / "agingfactor_tablesize_experiment_data"
OUTPUT_DIR = STEP0_DIR / "plots"


def find_best_config(root: Path, mode: str, tablesize: int) -> tuple[Path, float, dict]:
    """Return (agingfactor_dir, hitrate, metadata) for best average_hitrate_per_lti at tablesize."""
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


def binned_hitrate(df: pd.DataFrame, window: int, last_bin_start: int = 180) -> pd.DataFrame:
    """Average hit_rate over fixed-size LTI windows; final bin spans last_bin_start through end."""
    rows = []
    t = int(df["lti_start_time"].iloc[0])

    while t + window - 1 < last_bin_start:
        chunk = df[(df["lti_start_time"] >= t) & (df["lti_start_time"] <= t + window - 1)]
        rows.append(
            {
                "lti_start": chunk["lti_start_time"].iloc[0],
                "lti_end": chunk["lti_start_time"].iloc[-1],
                "x": (chunk["lti_start_time"].iloc[0] + chunk["lti_start_time"].iloc[-1]) / 2,
                "hit_rate": chunk["hit_rate"].mean(),
                "n_ltis": len(chunk),
            }
        )
        t += window

    chunk = df[df["lti_start_time"] >= last_bin_start]
    if not chunk.empty:
        rows.append(
            {
                "lti_start": chunk["lti_start_time"].iloc[0],
                "lti_end": chunk["lti_start_time"].iloc[-1],
                "x": (chunk["lti_start_time"].iloc[0] + chunk["lti_start_time"].iloc[-1]) / 2,
                "hit_rate": chunk["hit_rate"].mean(),
                "n_ltis": len(chunk),
            }
        )

    return pd.DataFrame(rows)


def plot_tablesize(tablesize: int, window: int, last_bin_start: int, output_dir: Path) -> Path:
    sp_dir, sp_hr, sp_meta = find_best_config(ROOT, "speculative", tablesize)
    sr_dir, sr_hr, sr_meta = find_best_config(ROOT, "speculativereactive", tablesize)

    reactive_path = ROOT / "mode_reactive" / f"tablesize_{tablesize}" / "lti_metrics.csv"
    with open(ROOT / "mode_reactive" / f"tablesize_{tablesize}" / "summary.json") as f:
        re_hr = json.load(f)["average_hitrate_per_lti"]

    sp = binned_hitrate(pd.read_csv(sp_dir / "lti_metrics.csv"), window, last_bin_start)
    sr = binned_hitrate(pd.read_csv(sr_dir / "lti_metrics.csv"), window, last_bin_start)
    reactive = binned_hitrate(pd.read_csv(reactive_path), window, last_bin_start)

    colors = {"sr": "#2ca02c", "sp": "#ff7f0e", "re": "#1f77b4"}

    n_bins = len(sr)
    x = np.arange(n_bins)
    width = 0.25
    bin_labels = [f"{int(r['lti_start'])}–{int(r['lti_end'])}" for _, r in sr.iterrows()]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - width, sr["hit_rate"], width, color=colors["sr"])
    ax.bar(x, sp["hit_rate"], width, color=colors["sp"])
    ax.bar(x + width, reactive["hit_rate"], width, color=colors["re"])

    ax.axhline(sr_hr, color=colors["sr"], linestyle=":", linewidth=1.5)
    ax.axhline(sp_hr, color=colors["sp"], linestyle=":", linewidth=1.5)
    ax.axhline(re_hr, color=colors["re"], linestyle=":", linewidth=1.5)

    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, rotation=45, ha="right", fontsize=7)
    ax.set_xlabel(f"LTI window ({window} LTIs per group; last group from {last_bin_start})")
    ax.set_ylabel("Hit rate (%)")
    ax.set_title(f"Hit rate avg over {window} LTIs, tablesize={tablesize}")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"tablesize_{tablesize}_bin{window}_lti_hitrate.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)

    print(f"tablesize={tablesize}, window={window} LTIs:")
    print(f"  best speculative:         {sp_dir.relative_to(ROOT)} ({sp_hr:.2f}%)")
    print(f"  best speculativereactive: {sr_dir.relative_to(ROOT)} ({sr_hr:.2f}%)")
    print(f"  reactive avg:             {re_hr:.2f}%")
    print(f"  bins: {len(sr)} points per series")
    print(f"  saved: {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tablesize", type=int, default=50)
    parser.add_argument("--window", type=int, default=20, help="Number of LTIs per averaged data point")
    parser.add_argument(
        "--last-bin-start",
        type=int,
        default=180,
        help="LTI start time for the final bin (through end of run)",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    plot_tablesize(args.tablesize, args.window, args.last_bin_start, args.output_dir.resolve())


if __name__ == "__main__":
    main()
