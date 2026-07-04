#!/usr/bin/env python3
"""Plot hidden_layers param-impact: grouped Fixed vs Interpolated bars per NFA."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from plot_param_impact import RESULTS_ROOT, make_panel_grid

STEP3_DIR = Path(__file__).resolve().parents[1]
PLOTS_DIR = STEP3_DIR / "plots"
HIDDEN_LAYERS_PLOTS_DIR = PLOTS_DIR / "hidden_layers"

HIDDEN_LAYER_VALUES = [1, 2, 3]
VARIANTS = ("fixed", "interpolated")
VARIANT_LABELS = {"fixed": "Fixed", "interpolated": "Interpolated"}
VARIANT_COLORS = {"fixed": "#E8A317", "interpolated": "#4CAF50"}
BAR_HATCH = "////"
BAR_EDGE = "black"
BAR_EDGEWIDTH = 0.8
BAR_WIDTH_INCHES = 0.11
GROUP_SPACING = 0.14
X_PAD = 0.07

# Bar panels only need room for 3 tight bar groups, so they are narrower than the
# line-plot panels while keeping the shared (ternary-matched) panel height.
HL_PANEL_WIDTH_IN = 1.55

PANELS = (
    {
        "objective": "speculativereactive_hitrate",
        "metric_key": "average_hitrate_per_lti",
        "ylabel": "Hit Rate (%)",
        "scale": 1.0,
        "ylim": (72, 76),
    },
    {
        "objective": "speculation_efficiency",
        "metric_key": "average_speculation_efficiency_per_lti",
        "ylabel": "Spec. Eff.",
        "scale": 1.0,
        "ylim": (2.8, 4.5),
    },
)

NFA_HITRATE_YLIM: dict[int, tuple[float, float]] = {}


def panel_for_nfa(panel: dict, nfa: int) -> dict:
    if panel["objective"] == "speculativereactive_hitrate" and nfa in NFA_HITRATE_YLIM:
        return {**panel, "ylim": NFA_HITRATE_YLIM[nfa]}
    return panel


X_LABEL = r"Hidden Layers, $n_h$"

plt.rcParams.update(
    {
        "font.size": 14,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 12,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "black",
        "grid.color": "#cccccc",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def summary_path(nfa: int, variant: str, hidden_layers: int, objective: str) -> Path:
    path = (
        RESULTS_ROOT
        / "hidden_layers"
        / objective
        / f"numberofFlowsPerAgent_{nfa}"
        / variant
        / f"hidden_layers_{hidden_layers}"
        / "summary.json"
    )
    if path.exists():
        return path

    if objective == "speculation_efficiency":
        fallback = (
            RESULTS_ROOT
            / "hidden_layers"
            / "speculativereactive_hitrate"
            / f"numberofFlowsPerAgent_{nfa}"
            / variant
            / f"hidden_layers_{hidden_layers}"
            / "summary.json"
        )
        if fallback.exists():
            return fallback

    raise FileNotFoundError(f"Missing summary: {path}")


def load_metric(nfa: int, variant: str, hidden_layers: int, *, metric_key: str, objective: str) -> float:
    summary = json.loads(summary_path(nfa, variant, hidden_layers, objective).read_text())
    return float(summary[metric_key])


def style_axes_spines(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def x_positions() -> np.ndarray:
    return np.arange(len(HIDDEN_LAYER_VALUES), dtype=float) * GROUP_SPACING


def x_limits() -> tuple[float, float]:
    x = x_positions()
    return float(x[0] - X_PAD), float(x[-1] + X_PAD)


def bar_width_data(ax: plt.Axes, width_inches: float) -> float:
    """Convert a physical bar width to data units for the current axes size."""
    bbox = ax.get_window_extent().transformed(ax.figure.dpi_scale_trans.inverted())
    x0, x1 = ax.get_xlim()
    if bbox.width <= 0:
        return width_inches
    return width_inches * (x1 - x0) / bbox.width


def prepare_panel_axes(
    ax: plt.Axes,
    panel: dict,
    *,
    show_ylabel: bool,
    title: str | None = None,
) -> None:
    x = x_positions()
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in HIDDEN_LAYER_VALUES])
    ax.set_xlim(*x_limits())
    ax.set_xlabel(X_LABEL)
    if show_ylabel:
        ax.set_ylabel(panel["ylabel"])
    ax.set_ylim(*panel["ylim"])
    ax.grid(True, axis="y", linestyle="--", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    style_axes_spines(ax)
    if title is not None:
        ax.set_title(title, fontsize=13, pad=6)


def plot_grouped_bars(
    ax: plt.Axes,
    nfa: int,
    panel: dict,
    *,
    show_legend: bool,
    bar_width: float,
) -> dict:
    x = x_positions()
    series_data: dict[str, list[float]] = {}

    for offset, variant in zip((-bar_width / 2, bar_width / 2), VARIANTS):
        raw_values = [
            load_metric(
                nfa,
                variant,
                hidden_layers,
                metric_key=panel["metric_key"],
                objective=panel["objective"],
            )
            for hidden_layers in HIDDEN_LAYER_VALUES
        ]
        series_data[variant] = raw_values
        plot_values = [value * panel["scale"] for value in raw_values]
        ax.bar(
            x + offset,
            plot_values,
            bar_width,
            label=VARIANT_LABELS[variant],
            color=VARIANT_COLORS[variant],
            hatch=BAR_HATCH,
            edgecolor=BAR_EDGE,
            linewidth=BAR_EDGEWIDTH,
            zorder=3,
        )

    if show_legend:
        ax.legend(loc="upper left", frameon=False)

    return {
        "hidden_layers": HIDDEN_LAYER_VALUES,
        "fixed": series_data["fixed"],
        "interpolated": series_data["interpolated"],
    }


NFA_VALUES = (5, 8)


def plot_hidden_layers(nfa: int) -> Path:
    data: dict = {"numberofFlowsPerAgent": nfa, "panels": {}}

    fig, axes = make_panel_grid(2, panel_width=HL_PANEL_WIDTH_IN)
    panel_cfgs = [panel_for_nfa(panel, nfa) for panel in PANELS]
    for ax, panel_cfg in zip(axes, panel_cfgs):
        prepare_panel_axes(ax, panel_cfg, show_ylabel=True)

    fig.canvas.draw()

    for ax, panel, panel_cfg, show_legend in zip(axes, PANELS, panel_cfgs, (True, False)):
        bar_width = bar_width_data(ax, BAR_WIDTH_INCHES)
        data["panels"][panel["objective"]] = plot_grouped_bars(
            ax,
            nfa,
            panel_cfg,
            show_legend=show_legend,
            bar_width=bar_width,
        )

    HIDDEN_LAYERS_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out_base = HIDDEN_LAYERS_PLOTS_DIR / f"hidden_layers_nfa{nfa}_fixed_vs_interpolated"
    fig.savefig(out_base.with_suffix(".png"), dpi=180, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

    (HIDDEN_LAYERS_PLOTS_DIR / f"hidden_layers_nfa{nfa}_fixed_vs_interpolated_data.json").write_text(
        json.dumps(data, indent=2)
    )
    return out_base


def plot_hidden_layers_combined() -> Path:
    data: dict = {"numberofFlowsPerAgent": list(NFA_VALUES), "series": {}}

    fig, axes = make_panel_grid(4, panel_width=HL_PANEL_WIDTH_IN)
    plot_specs = [
        (PANELS[0], NFA_VALUES[0], 0),
        (PANELS[0], NFA_VALUES[1], 1),
        (PANELS[1], NFA_VALUES[0], 2),
        (PANELS[1], NFA_VALUES[1], 3),
    ]
    for panel, nfa, col in plot_specs:
        panel_cfg = panel_for_nfa(panel, nfa)
        prepare_panel_axes(
            axes[col],
            panel_cfg,
            show_ylabel=col in (0, 2),
            title=f"$U$ = {nfa}",
        )

    fig.canvas.draw()

    for panel, nfa, col in plot_specs:
        panel_cfg = panel_for_nfa(panel, nfa)
        key = f"nfa{nfa}_{panel['objective']}"
        bar_width = bar_width_data(axes[col], BAR_WIDTH_INCHES)
        data["series"][key] = plot_grouped_bars(
            axes[col],
            nfa,
            panel_cfg,
            show_legend=col == 0,
            bar_width=bar_width,
        )

    HIDDEN_LAYERS_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out_base = HIDDEN_LAYERS_PLOTS_DIR / "hidden_layers_fixed_vs_interpolated"
    fig.savefig(out_base.with_suffix(".png"), dpi=180, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

    (HIDDEN_LAYERS_PLOTS_DIR / "hidden_layers_fixed_vs_interpolated_data.json").write_text(json.dumps(data, indent=2))
    return out_base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot hidden_layers Fixed vs Interpolated bars.")
    parser.add_argument(
        "--nfa",
        type=int,
        default=None,
        help="Generate only this numberofFlowsPerAgent figure (5 or 8). Default: combined only.",
    )
    parser.add_argument(
        "--combined-only",
        action="store_true",
        help="Generate only the combined 4-panel figure.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.nfa is None:
        out_base = plot_hidden_layers_combined()
        print(f"Wrote {out_base}.pdf/.png")
        print(f"Wrote {HIDDEN_LAYERS_PLOTS_DIR / 'hidden_layers_fixed_vs_interpolated_data.json'}")
        return

    out_base = plot_hidden_layers(args.nfa)
    print(f"Wrote {out_base}.pdf/.png")
    print(f"Wrote {HIDDEN_LAYERS_PLOTS_DIR / f'hidden_layers_nfa{args.nfa}_fixed_vs_interpolated_data.json'}")

    if not args.combined_only:
        combined_base = plot_hidden_layers_combined()
        print(f"Wrote {combined_base}.pdf/.png")
        print(f"Wrote {HIDDEN_LAYERS_PLOTS_DIR / 'hidden_layers_fixed_vs_interpolated_data.json'}")


if __name__ == "__main__":
    main()
