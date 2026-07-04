#!/usr/bin/env python3
"""Three-panel resource overhead figure across reactive/bandit/DQN/PPO.

Left:   total CPU-seconds bar chart
Middle: per-LTI average RSS distribution (box-and-whisker)
Right:  peak RSS bar chart
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

BASE = Path(__file__).resolve().parent
PLOTS_DIR = BASE / "plots"

SERIES = [
    {"subdir": "reactive", "label": "Reactive", "color": "#FF5500"},
    {"subdir": "speculativereactive_bandit", "label": "Bandit", "color": "#0055FF"},
    {"subdir": "speculativereactive_ppo", "label": "PPO", "color": "#9467BD"},
    {"subdir": "speculativereactive_dqn", "label": "DQN", "color": "#2CA02C"},
]

plt.rcParams.update(
    {
        "font.size": 18,
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


def style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", alpha=0.6, linestyle="-", linewidth=0.6)
    ax.grid(False, axis="x")


def load_resource_csv(subdir: str) -> pd.DataFrame | None:
    csv_path = BASE / subdir / "lti_resource_metrics.csv"
    if not csv_path.exists():
        print(f"SKIP missing: {csv_path}")
        return None
    return pd.read_csv(csv_path).sort_values("lti_number").reset_index(drop=True)


def draw_box(ax: plt.Axes, data, labels, colors, ylabel, title, ylim=None):
    positions = [i * 0.5 for i in range(len(data))]
    bp = ax.boxplot(
        data,
        positions=positions,
        patch_artist=True,
        widths=0.3,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.6},
        whiskerprops={"color": "black", "linewidth": 1.0},
        capprops={"color": "black", "linewidth": 1.0},
        boxprops={"linewidth": 1.0, "edgecolor": "black"},
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.set_xticks(positions)
    ax.set_xticklabels(
        ["Reac." if lbl == "Reactive" else lbl for lbl in labels], rotation=90
    )
    ax.set_xlim(-0.4, (positions[-1] if positions else 0) + 0.4)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    style_axes(ax)


def plot_three_panel() -> Path:
    fig, (ax_rss, ax_peak, ax_cpu) = plt.subplots(
        1, 3, figsize=(9.0, 3.6), gridspec_kw={"width_ratios": [1, 1, 1]}
    )

    labels, colors = [], []
    rss_data = []
    cpu_seconds = []
    peak_values = []

    for series in SERIES:
        df = load_resource_csv(series["subdir"])
        if df is None:
            continue
        labels.append(series["label"])
        colors.append(series["color"])
        rss_data.append(df["avg_rss_mb"].astype(float).values)
        cpu_seconds.append(float(df["cpu_time_s_delta"].astype(float).sum()))
        peak_values.append(float(df["peak_rss_mb"].max()))
        print(
            f"{series['label']}: cpu_seconds={cpu_seconds[-1]:.1f} s, "
            f"rss_med={pd.Series(rss_data[-1]).median():.0f} MB, "
            f"peak_rss={peak_values[-1]:.0f} MB"
        )

    hatches = ["//", "\\\\", "xx", ".."]
    bar_labels = ["Reac." if lbl == "Reactive" else lbl for lbl in labels]

    def draw_bar(ax, values, ylabel, title, fmt, ylim=None):
        x_pos = [i * 0.5 for i in range(len(labels))]
        bars = ax.bar(x_pos, values, color=colors, width=0.3,
                      edgecolor="black", linewidth=0.8, alpha=0.9)
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(bar_labels, rotation=90)
        ax.set_xlim(-0.4, (x_pos[-1] if x_pos else 0) + 0.4)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        if ylim is not None:
            ax.set_ylim(*ylim)
        else:
            ax.set_ylim(0, max(values) * 1.18 if values else 1)
        style_axes(ax)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    fmt.format(value), ha="center", va="bottom", fontsize=13)

    draw_box(ax_rss, rss_data, labels, colors,
             "Avg RSS (MB)", "RSS", ylim=(500, 880))
    draw_bar(ax_peak, peak_values, "Peak RSS (MB)", "Peak RSS", "{:.0f}", ylim=(500, 880))
    draw_bar(ax_cpu, cpu_seconds, "Total CPU Time (s)", "Total CPU Time", "{:.1f}")

    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.14, top=0.9, wspace=0.55)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out_base = PLOTS_DIR / "resource_overhead_three_panel"
    fig.savefig(out_base.with_suffix(".png"), dpi=180, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    return out_base


def main() -> None:
    out_base = plot_three_panel()
    print(f"\nWrote {out_base}.png")
    print(f"Wrote {out_base}.pdf")


if __name__ == "__main__":
    main()
