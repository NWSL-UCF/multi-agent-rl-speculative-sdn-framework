#!/usr/bin/env python3
"""Per-packet latency (EWMA) vs total packets arrived."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DEBUG_DIR = Path(__file__).resolve().parent
PLOTS_DIR = DEBUG_DIR / "plots"

EWMA_ALPHA_LATENCY = 0.001

SERIES = [
    {
        "subdir": "reactive",
        "label": "Reactive",
        "legend_label": "Reac.",
        "color": "#FF5500",
    },
    {
        "subdir": "speculativereactive",
        "label": "Speculative+Reactive",
        "legend_label": "Spec.+Reac.",
        "color": "#0055FF",
    },
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


def load_per_packet_metrics(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.sort_values("id").reset_index(drop=True)
    df["packets_arrived"] = df.index + 1
    df["latency_ms"] = df["total_delay"] * 1000.0
    return df


def ewma(series: pd.Series, alpha: float) -> pd.Series:
    return series.ewm(alpha=alpha, adjust=False).mean()


def style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", alpha=0.6, linestyle="-", linewidth=0.6)
    ax.grid(False, axis="x")


def plot_per_packet_latency() -> Path:
    fig, ax = plt.subplots(figsize=(5.0, 3.5))

    for series in SERIES:
        csv_path = DEBUG_DIR / series["subdir"] / "per_packet_metrics.csv"
        if not csv_path.exists():
            print(f"SKIP missing: {csv_path}")
            continue

        df = load_per_packet_metrics(csv_path)
        df["latency_ms_ewma"] = ewma(df["latency_ms"], EWMA_ALPHA_LATENCY)
        avg_latency_ms = float(df["latency_ms"].mean())

        ax.plot(
            df["packets_arrived"],
            df["latency_ms_ewma"],
            color=series["color"],
            linewidth=1.6,
            label=f"{series['legend_label']} ({avg_latency_ms:.2f} ms)",
        )

        print(
            f"{series['label']}: packets={len(df)}, avg_latency={avg_latency_ms:.4f} ms"
        )

    ax.set_xlabel("Total Packets Arrived")
    ax.set_ylabel("Per-Packet Latency (ms)")
    ax.set_xlim(left=0)
    ax.set_ylim(3, 9)
    style_axes(ax)
    ax.legend(loc="upper right", frameon=True, framealpha=0.95, edgecolor="#cccccc")

    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.12, top=0.95)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out_base = PLOTS_DIR / "per_packet_latency_vs_total_packets_arrived"
    fig.savefig(out_base.with_suffix(".png"), dpi=180, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    return out_base


def main() -> None:
    out_base = plot_per_packet_latency()
    print(f"\nWrote {out_base}.png")
    print(f"Wrote {out_base}.pdf")


if __name__ == "__main__":
    main()
