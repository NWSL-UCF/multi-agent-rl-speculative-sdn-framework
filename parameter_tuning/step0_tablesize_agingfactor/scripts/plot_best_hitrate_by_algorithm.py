#!/usr/bin/env python3
"""Plot best speculativereactive hit rate vs aging factor, one panel per algorithm."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

STEP0_DIR = Path(__file__).resolve().parents[1]
ROOT = STEP0_DIR.parents[2] / "results" / "agingfactor_tablesize_experiment_data"
OUTPUT_DIR = STEP0_DIR / "plots"

ALGORITHMS = ["bandit", "ppo", "dqn"]

ALGORITHM_LABELS = {
    "bandit": "Bandit",
    "ppo": "PPO",
    "dqn": "DQN",
}
TABLESIZES = [30, 50, 70, 90, 110]
AGING_FACTORS = [0.75, 0.80, 0.85, 0.90, 0.95, 0.99, 0.991, 0.993, 0.995, 0.997, 0.999]

ORDERING_MARKERS = {
    "source": "o",
    "destination": "s",
    "trace": "^",
}

ORDERING_LABELS = {
    "source": "Source",
    "destination": "Destination",
    "trace": "Trace",
}

TABLESIZE_COLORS = {
    30: "#648FFF",
    50: "#FFB000",
    70: "#785EF0",
    90: "#DC267F",
    110: "#FE6100",
}

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


def load_best_series_for_algo(
    root: Path, algorithm: str, tablesize: int
) -> list[tuple[float, float, str]]:
    """Best hit rate per aging factor and the ordering that achieved it."""
    by_af: dict[float, tuple[float, str]] = {}
    base = root / "mode_speculativereactive" / f"algorithm_{algorithm}"
    for summary_path in base.rglob(f"tablesize_{tablesize}/agingfactor_*/summary.json"):
        af_match = re.search(r"agingfactor_([\d.]+)/summary\.json$", str(summary_path))
        ordering_match = re.search(r"ordering_(source|destination|trace)/", str(summary_path))
        if not af_match or not ordering_match:
            continue
        af = float(af_match.group(1))
        ordering = ordering_match.group(1)
        hit_rate = read_hit_rate(summary_path)
        if hit_rate is None:
            continue
        prev_hr = by_af.get(af, (float("-inf"), ordering))[0]
        if hit_rate > prev_hr:
            by_af[af] = (hit_rate, ordering)
    return sorted((af, hr, ordering) for af, (hr, ordering) in by_af.items())


def load_reactive_hit_rate(root: Path, tablesize: int) -> float | None:
    return read_hit_rate(root / "mode_reactive" / f"tablesize_{tablesize}" / "summary.json")


def compute_shared_ylim(root: Path) -> tuple[float, float]:
    values: list[float] = []
    for algorithm in ALGORITHMS:
        for tablesize in TABLESIZES:
            series = [(af, hr, ordering) for af, hr, ordering in load_best_series_for_algo(root, algorithm, tablesize) if af in AGING_FACTORS]
            values.extend(hr for _, hr, _ in series)
            reactive_hr = load_reactive_hit_rate(root, tablesize)
            if reactive_hr is not None:
                values.append(reactive_hr)
    if not values:
        return 0, 100
    ymin, ymax = min(values), max(values)
    pad = (ymax - ymin) * 0.05 or 1
    return max(0, ymin - pad), min(100, ymax + pad)


def draw_algo_panel(
    ax: plt.Axes,
    root: Path,
    algorithm: str,
    ylim: tuple[float, float],
    *,
    show_ylabel: bool = True,
) -> None:
    x_positions = list(range(len(AGING_FACTORS)))
    af_to_x = {af: i for i, af in enumerate(AGING_FACTORS)}

    for tablesize in TABLESIZES:
        color = TABLESIZE_COLORS[tablesize]
        series = [
            (af, hr, ordering)
            for af, hr, ordering in load_best_series_for_algo(root, algorithm, tablesize)
            if af in AGING_FACTORS
        ]
        reactive_hr = load_reactive_hit_rate(root, tablesize)
        if not series:
            continue

        xs = [af_to_x[af] for af, _, _ in series]
        ys = [hr for _, hr, _ in series]

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

        ax.plot(xs, ys, color=color, linestyle="-", linewidth=0.55, zorder=3)
        for x, y, ordering in zip(xs, ys, (o for _, _, o in series)):
            ax.plot(
                x,
                y,
                linestyle="none",
                marker=ORDERING_MARKERS[ordering],
                color=color,
                markersize=7,
                markeredgecolor="white",
                markeredgewidth=1.2,
                zorder=4,
            )

    ax.set_title(ALGORITHM_LABELS[algorithm], loc="center", pad=10)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(af) for af in AGING_FACTORS], rotation=45, ha="right")
    ax.set_xlabel("Aging Factor")
    if show_ylabel:
        ax.set_ylabel("Hit Rate(%)")
        ax.tick_params(axis="y", labelleft=True)
    else:
        ax.tick_params(axis="y", labelleft=False)
    ax.set_ylim(ylim)
    ax.grid(True, alpha=0.6, linestyle="-", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_boxed_legend(
    fig: plt.Figure,
    handles: list,
    x_center: float,
    y_top: float,
    ncol: int,
    caption: str,
) -> None:
    legend = fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(x_center, y_top),
        bbox_transform=fig.transFigure,
        ncol=ncol,
        frameon=True,
        fancybox=False,
        edgecolor="#cccccc",
        fontsize=13,
        handlelength=1.4,
        handletextpad=0.35,
        columnspacing=0.9,
        borderaxespad=0.4,
    )
    fig.add_artist(legend)
    fig.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig.transFigure.inverted())
    fig.text(
        (bbox.x0 + bbox.x1) / 2,
        bbox.y0 - 0.012,
        caption,
        transform=fig.transFigure,
        ha="center",
        va="top",
        fontsize=12,
        fontweight="bold",
        clip_on=False,
    )


def make_legend(fig: plt.Figure) -> None:
    tablesize_handles = [
        Line2D([0], [0], color=TABLESIZE_COLORS[ts], linewidth=1.2, label=str(ts))
        for ts in TABLESIZES
    ]
    ordering_handles = [
        Line2D(
            [0],
            [0],
            color="#666666",
            linewidth=0,
            marker=ORDERING_MARKERS[ordering],
            markersize=7,
            label=ORDERING_LABELS[ordering],
        )
        for ordering in ("source", "destination", "trace")
    ]
    mode_handles = [
        Line2D([0], [0], color="#666666", linewidth=0.55, label="Speculative+Reactive Best"),
        Line2D([0], [0], color="#666666", linewidth=1.2, linestyle="--", label="Reactive"),
    ]

    y_top = 0.20
    add_boxed_legend(fig, tablesize_handles, 0.19, y_top, ncol=5, caption="SFT Size")
    add_boxed_legend(
        fig,
        ordering_handles,
        0.50,
        y_top,
        ncol=3,
        caption="Flow Ordering in Controller Table",
    )
    add_boxed_legend(fig, mode_handles, 0.81, y_top, ncol=2, caption="Mode")


def plot_by_algorithm(root: Path, output_dir: Path) -> tuple[Path, Path]:
    ylim = compute_shared_ylim(root)
    fig, axes = plt.subplots(1, 3, figsize=(14, 6), sharey=True)

    for ax, algorithm in zip(axes, ALGORITHMS):
        draw_algo_panel(ax, root, algorithm, ylim, show_ylabel=(algorithm == ALGORITHMS[0]))

    make_legend(fig)
    fig.subplots_adjust(left=0.075, right=0.98, bottom=0.36, top=0.95, wspace=0.08)

    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / "speculativereactive_best_by_algorithm"
    png_path = base.with_suffix(".png")
    pdf_path = base.with_suffix(".pdf")
    fig.savefig(png_path, dpi=180, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    return png_path, pdf_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    root = args.root.resolve()
    output_dir = args.output_dir.resolve()
    ylim = compute_shared_ylim(root)
    print(f"shared y-axis {ylim[0]:.1f}–{ylim[1]:.1f}%")
    png_path, pdf_path = plot_by_algorithm(root, output_dir)
    print(f"saved {png_path}")
    print(f"saved {pdf_path}")


if __name__ == "__main__":
    main()
