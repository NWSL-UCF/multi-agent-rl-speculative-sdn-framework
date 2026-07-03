#!/usr/bin/env python3
"""Plot best hit rate vs aging factor (max over all algo x ordering) with reactive baseline."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

STEP0_DIR = Path(__file__).resolve().parents[1]
ROOT = STEP0_DIR.parents[2] / "results" / "agingfactor_tablesize_experiment_data"
OUTPUT_DIR = STEP0_DIR / "plots"

MODES = ["speculative", "speculativereactive"]
TABLESIZES = [30, 50, 70, 90, 110]

AGING_RANGES: dict[str, list[float]] = {
    "low": [0.75, 0.80, 0.85, 0.90, 0.95, 0.99],
    "high": [0.99, 0.991, 0.993, 0.995, 0.997, 0.999],
}

PANEL_TITLES = {
    "low": "Low aging factors",
    "high": "High aging factors",
}

# Colorblind-friendly palette (IBM Design Library)
TABLESIZE_COLORS = {
    30: "#648FFF",
    50: "#FFB000",
    70: "#785EF0",
    90: "#DC267F",
    110: "#FE6100",
}

STAR_MARKERSIZE = 9

plt.rcParams.update(
    {
        "font.size": 13,
        "axes.titlesize": 15,
        "axes.titleweight": "bold",
        "axes.labelsize": 14,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#cccccc",
        "grid.color": "#dddddd",
    }
)


def read_hit_rate(summary_path: Path) -> float | None:
    if not summary_path.exists():
        return None
    with open(summary_path) as f:
        return json.load(f).get("average_hitrate_per_lti")


def load_best_series(root: Path, mode: str, tablesize: int) -> list[tuple[float, float]]:
    by_af: dict[float, float] = {}
    for summary_path in (root / f"mode_{mode}").rglob(
        f"tablesize_{tablesize}/agingfactor_*/summary.json"
    ):
        match = re.search(r"agingfactor_([\d.]+)/summary\.json$", str(summary_path))
        if not match:
            continue
        af = float(match.group(1))
        hit_rate = read_hit_rate(summary_path)
        if hit_rate is None:
            continue
        by_af[af] = max(by_af.get(af, float("-inf")), hit_rate)
    return sorted(by_af.items())


def filter_series(series: list[tuple[float, float]], aging_factors: list[float]) -> list[tuple[float, float]]:
    allowed = set(aging_factors)
    return [(af, hr) for af, hr in series if af in allowed]


def load_reactive_hit_rate(root: Path, tablesize: int) -> float | None:
    return read_hit_rate(root / "mode_reactive" / f"tablesize_{tablesize}" / "summary.json")


def compute_shared_ylim(root: Path, mode: str, ranges: dict[str, list[float]]) -> tuple[float, float]:
    values: list[float] = []
    all_afs = {af for afs in ranges.values() for af in afs}
    for tablesize in TABLESIZES:
        series = filter_series(load_best_series(root, mode, tablesize), list(all_afs))
        values.extend(hr for _, hr in series)
        reactive_hr = load_reactive_hit_rate(root, tablesize)
        if reactive_hr is not None:
            values.append(reactive_hr)
    if not values:
        return 0, 100
    ymin, ymax = min(values), max(values)
    pad = (ymax - ymin) * 0.05 or 1
    return max(0, ymin - pad), min(100, ymax + pad)


def compute_global_peaks(
    root: Path, mode: str, ranges: dict[str, list[float]]
) -> dict[int, tuple[str, float, float]]:
    """Per tablesize: (range_name, aging_factor, hit_rate) at global max across all ranges."""
    peaks: dict[int, tuple[str, float, float]] = {}
    for tablesize in TABLESIZES:
        best_hr = float("-inf")
        best_range: str | None = None
        best_af: float | None = None
        for range_name, aging_factors in ranges.items():
            series = filter_series(load_best_series(root, mode, tablesize), aging_factors)
            for af, hr in series:
                if hr > best_hr:
                    best_hr = hr
                    best_range = range_name
                    best_af = af
        if best_range is not None and best_af is not None:
            peaks[tablesize] = (best_range, best_af, best_hr)
    return peaks


def draw_panel(
    ax: plt.Axes,
    root: Path,
    mode: str,
    aging_factors: list[float],
    range_name: str,
    global_peaks: dict[int, tuple[str, float, float]],
    ylim: tuple[float, float],
    panel_label: str,
    panel_title: str,
    show_ylabel: bool = True,
) -> bool:
    has_data = False
    x_positions = list(range(len(aging_factors)))
    af_to_x = {af: i for i, af in enumerate(aging_factors)}

    for tablesize in TABLESIZES:
        color = TABLESIZE_COLORS[tablesize]
        series = filter_series(load_best_series(root, mode, tablesize), aging_factors)
        reactive_hr = load_reactive_hit_rate(root, tablesize)
        if not series:
            continue

        xs = [af_to_x[af] for af, _ in series]
        ys = [hr for _, hr in series]
        has_data = True

        if reactive_hr is not None:
            ax.axhline(reactive_hr, color=color, linestyle="--", linewidth=1.8, alpha=0.85, zorder=1)
            ax.fill_between(
                xs,
                ys,
                reactive_hr,
                where=[y >= reactive_hr for y in ys],
                color=color,
                alpha=0.12,
                interpolate=True,
                zorder=2,
            )

        ax.plot(
            xs,
            ys,
            color=color,
            linestyle="-",
            linewidth=0.8,
            marker="o",
            markersize=7,
            markeredgecolor="white",
            markeredgewidth=1.2,
            zorder=3,
        )

        if tablesize in global_peaks:
            peak_range, peak_af, peak_hr = global_peaks[tablesize]
            if peak_range == range_name and peak_af in af_to_x:
                peak_x = af_to_x[peak_af]
                ax.plot(
                    peak_x,
                    peak_hr,
                    marker="*",
                    markersize=STAR_MARKERSIZE,
                    color=color,
                    linestyle="none",
                    zorder=5,
                )

    if not has_data:
        return False

    ax.set_title(f"{panel_label}  {panel_title}", loc="left", pad=10)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(af) for af in aging_factors], rotation=0)
    ax.set_xlabel("Aging Factor")
    if show_ylabel:
        ax.set_ylabel("Hit Rate(%)")
    ax.set_ylim(ylim)
    ax.grid(True, alpha=0.6, linestyle="-", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return True


def make_legend(fig: plt.Figure) -> None:
    tablesize_handles = [
        Line2D(
            [0], [0], color=TABLESIZE_COLORS[ts], linewidth=1.2, marker="o", markersize=4, label=str(ts)
        )
        for ts in TABLESIZES
    ]
    style_handles = [
        Line2D([0], [0], color="#666666", linewidth=0.8, marker="o", markersize=4, label="Speculative+Reactive Best"),
        Line2D([0], [0], color="#666666", linewidth=1.2, linestyle="--", label="Reactive"),
        mpatches.Patch(facecolor="#888888", alpha=0.25, label="Gain"),
        Line2D([0], [0], color="#666666", linewidth=0, marker="*", markersize=STAR_MARKERSIZE, label="Max"),
    ]
    fig.legend(
        handles=tablesize_handles + style_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.06),
        ncol=9,
        frameon=False,
        fontsize=13,
        handlelength=1.4,
        handletextpad=0.35,
        columnspacing=0.9,
        borderaxespad=0.0,
    )


def plot_combined(
    root: Path,
    mode: str,
    output_dir: Path,
    ranges: dict[str, list[float]],
    ylim: tuple[float, float],
    global_peaks: dict[int, tuple[str, float, float]],
) -> tuple[Path, Path] | None:
    range_items = list(ranges.items())
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    labels = ["(a)", "(b)"]

    for ax, (range_name, aging_factors), panel_label in zip(axes, range_items, labels):
        ok = draw_panel(
            ax,
            root,
            mode,
            aging_factors,
            range_name,
            global_peaks,
            ylim,
            panel_label,
            PANEL_TITLES[range_name],
            show_ylabel=(ax is axes[0]),
        )
        if not ok:
            plt.close(fig)
            return None

    make_legend(fig)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.16, wspace=0.12)

    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / f"{mode}_best_over_algo_ordering"
    png_path = base.with_suffix(".png")
    pdf_path = base.with_suffix(".pdf")
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def plot_single(
    root: Path,
    mode: str,
    output_dir: Path,
    aging_factors: list[float],
    range_name: str,
    ylim: tuple[float, float],
    global_peaks: dict[int, tuple[str, float, float]],
) -> Path | None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ok = draw_panel(
        ax,
        root,
        mode,
        aging_factors,
        range_name,
        global_peaks,
        ylim,
        "",
        PANEL_TITLES[range_name],
    )
    if not ok:
        plt.close(fig)
        return None

    make_legend(fig)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.16)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{mode}_best_over_algo_ordering_{range_name}.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--mode", choices=[*MODES, "all"], default="speculativereactive")
    parser.add_argument(
        "--range",
        choices=[*AGING_RANGES.keys(), "all"],
        default="all",
    )
    parser.add_argument(
        "--separate",
        action="store_true",
        help="Also write individual low/high panel PNGs",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    output_dir = args.output_dir.resolve()
    modes = MODES if args.mode == "all" else [args.mode]
    ranges = AGING_RANGES if args.range == "all" else {args.range: AGING_RANGES[args.range]}

    for mode in modes:
        ylim = compute_shared_ylim(root, mode, ranges)
        global_peaks = compute_global_peaks(root, mode, ranges)
        print(f"{mode}: shared y-axis {ylim[0]:.1f}–{ylim[1]:.1f}%")
        for ts, (rng, af, hr) in sorted(global_peaks.items()):
            print(f"  tablesize {ts}: global max {hr:.2f}% at af={af} ({rng})")

        if args.range == "all" and len(ranges) == 2:
            paths = plot_combined(root, mode, output_dir, ranges, ylim, global_peaks)
            if paths:
                png_path, pdf_path = paths
                print(f"saved {png_path}")
                print(f"saved {pdf_path}")
        elif len(ranges) == 1:
            range_name, aging_factors = next(iter(ranges.items()))
            path = plot_single(root, mode, output_dir, aging_factors, range_name, ylim, global_peaks)
            if path:
                print(f"saved {path}")

        if args.separate:
            for range_name, aging_factors in ranges.items():
                path = plot_single(root, mode, output_dir, aging_factors, range_name, ylim, global_peaks)
                if path:
                    print(f"saved {path}")
                else:
                    print(f"skipped {mode}/{range_name} (no data)")


if __name__ == "__main__":
    main()
