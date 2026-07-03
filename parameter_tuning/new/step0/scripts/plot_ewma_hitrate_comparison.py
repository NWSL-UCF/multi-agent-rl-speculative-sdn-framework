#!/usr/bin/env python3
"""Plot EWMA hit rate for best speculative, speculativereactive, and reactive at same tablesize."""

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


def load_ewma(lti_path: Path, alpha: float) -> pd.DataFrame:
    df = pd.read_csv(lti_path)
    df["ewma_hitrate"] = ewma(df["hit_rate"], alpha)
    return df


def plot_tablesize(tablesize: int, alpha: float, output_dir: Path) -> Path:
    sp_dir, sp_hr, sp_meta = find_best_config(ROOT, "speculative", tablesize)
    sr_dir, sr_hr, sr_meta = find_best_config(ROOT, "speculativereactive", tablesize)

    reactive_path = ROOT / "mode_reactive" / f"tablesize_{tablesize}" / "lti_metrics.csv"
    with open(ROOT / "mode_reactive" / f"tablesize_{tablesize}" / "summary.json") as f:
        re_hr = json.load(f)["average_hitrate_per_lti"]

    sp = load_ewma(sp_dir / "lti_metrics.csv", alpha)
    sr = load_ewma(sr_dir / "lti_metrics.csv", alpha)
    reactive = load_ewma(reactive_path, alpha)

    sp_label = (
        f"speculative EWMA ({sp_meta['algorithm']}, {sp_meta['ordering']}, af={sp_meta['agingfactor']})"
    )
    sr_label = (
        f"speculativereactive EWMA ({sr_meta['algorithm']}, {sr_meta['ordering']}, af={sr_meta['agingfactor']})"
    )
    re_label = "reactive EWMA"

    colors = {"sr": "#2ca02c", "sp": "#ff7f0e", "re": "#1f77b4"}

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(sr["lti_start_time"], sr["ewma_hitrate"], color=colors["sr"], linewidth=2, label=sr_label)
    ax.plot(sp["lti_start_time"], sp["ewma_hitrate"], color=colors["sp"], linewidth=2, label=sp_label)
    ax.plot(reactive["lti_start_time"], reactive["ewma_hitrate"], color=colors["re"], linewidth=2, label=re_label)

    ax.axhline(sr_hr, color=colors["sr"], linestyle=":", linewidth=2, label=f"speculativereactive avg ({sr_hr:.1f}%)")
    ax.axhline(sp_hr, color=colors["sp"], linestyle=":", linewidth=2, label=f"speculative avg ({sp_hr:.1f}%)")
    ax.axhline(re_hr, color=colors["re"], linestyle=":", linewidth=2, label=f"reactive avg ({re_hr:.1f}%)")

    ax.set_xlabel("Simulation time (s)")
    ax.set_ylabel("Hit rate EWMA (%)")
    ax.set_title(f"Hit rate EWMA (α={alpha}), tablesize={tablesize}")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"tablesize_{tablesize}_best_modes_ewma.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)

    print(f"tablesize={tablesize}:")
    print(f"  best speculative:     {sp_dir.relative_to(ROOT)} ({sp_hr:.2f}% vs reactive {re_hr:.2f}%)")
    print(f"  best speculativereactive: {sr_dir.relative_to(ROOT)} ({sr_hr:.2f}%)")
    print(f"  saved: {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tablesize", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=0.04)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    plot_tablesize(args.tablesize, args.alpha, args.output_dir.resolve())


if __name__ == "__main__":
    main()
