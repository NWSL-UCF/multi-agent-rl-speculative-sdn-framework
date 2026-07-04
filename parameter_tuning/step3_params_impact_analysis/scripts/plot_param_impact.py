#!/usr/bin/env python3
"""Plot step-3 param-impact: three side-by-side panels with data-driven polynomial trends."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

STEP3_DIR = Path(__file__).resolve().parents[1]


def _resolve_results_root() -> Path:
    """Locate the step3 param-impact results across known layouts."""
    candidates = [
        STEP3_DIR.parents[1] / "results" / "step3_param_impact_analysis",
        STEP3_DIR.parents[2] / "results" / "step3_param_impact",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


RESULTS_ROOT = _resolve_results_root()
DIRECTORY_JSON = STEP3_DIR / "directory.json"
PLOTS_DIR = STEP3_DIR / "plots"


def param_plots_dir(param: str) -> Path:
    path = PLOTS_DIR / param
    path.mkdir(parents=True, exist_ok=True)
    return path


# Each panel is sized to match a panel of
# step2_second_tuning_using_ternary_search/plots/ternary_search_best_trajectories.pdf
# (figsize=(11.0, 4.1), 1x3 grid, left=0.08/right=0.98/bottom=0.14/top=0.86, wspace=0.06).
PANEL_WIDTH_IN = 3.1731
PANEL_HEIGHT_IN = 2.9520

# Insets (inches) around the panel grid. bbox_inches="tight" trims any slack,
# so these only need to be generous enough that labels never clip; the panel
# box size and inter-panel gap below are what fix the per-panel dimensions.
GRID_MARGIN_LEFT_IN = 0.62
GRID_MARGIN_RIGHT_IN = 0.12
GRID_MARGIN_BOTTOM_IN = 1.05
GRID_MARGIN_TOP_IN = 0.48
PANEL_GAP_IN = 0.72


def make_panel_grid(
    n_panels: int,
    *,
    sharex: bool = False,
    sharey: bool = False,
    panel_width: float = PANEL_WIDTH_IN,
    panel_height: float = PANEL_HEIGHT_IN,
    panel_gap: float = PANEL_GAP_IN,
):
    """Create a 1xN subplot grid whose every panel is exactly panel_width x panel_height."""
    fig_w = (
        GRID_MARGIN_LEFT_IN
        + n_panels * panel_width
        + (n_panels - 1) * panel_gap
        + GRID_MARGIN_RIGHT_IN
    )
    fig_h = GRID_MARGIN_BOTTOM_IN + panel_height + GRID_MARGIN_TOP_IN
    fig, axes = plt.subplots(1, n_panels, figsize=(fig_w, fig_h), sharex=sharex, sharey=sharey)
    fig.subplots_adjust(
        left=GRID_MARGIN_LEFT_IN / fig_w,
        right=1.0 - GRID_MARGIN_RIGHT_IN / fig_w,
        bottom=GRID_MARGIN_BOTTOM_IN / fig_h,
        top=1.0 - GRID_MARGIN_TOP_IN / fig_h,
        wspace=panel_gap / panel_width,
    )
    return fig, axes

PANEL_ORDER = (
    "speculative_hitrate",
    "speculativereactive_hitrate",
    "speculation_efficiency",
)

MARKER = "o"
BEST_STAR_COLOR = "#FF0000"
STAR_MARKERSIZE = 12
ANNOT_FONTSIZE = 12

PANEL_BASE_TITLES = {
    "speculative_hitrate": "Spec.",
    "speculativereactive_hitrate": "Spec.+Reac.",
    "speculation_efficiency": "Spec. Eff.",
}

ALGORITHM_LABELS = {
    "dqn": "DQN",
    "ppo": "PPO",
    "bandit": "Bandit",
}

SERIES = {
    "speculative_hitrate": {
        "color": "#00C853",
        "ylabel": "Hit Rate(%)",
    },
    "speculativereactive_hitrate": {
        "color": "#0066FF",
        "ylabel": "Hit Rate(%)",
    },
    "speculation_efficiency": {
        "color": "#FF6600",
        "ylabel": "Spec. Eff.",
    },
}

PARAM_XLABELS = {
    "numberofFlowsPerAgent": r"Flows per Agent, $U$",
    "gamma": r"Discount Factor, $\gamma$",
    "rewardAgingFactor": r"Reward Aging Factor, $\alpha_r$",
    "spatialReward": r"Spatial Reward Decay, $\beta$",
    "LR": r"Learning Rate, $\alpha_{dqn}$",
    "LFUTimeInterval": r"LFU Time Interval, $T_{LFU}$",
    "LTI": r"Learning Time Interval, $T_{LTI}$",
}

plt.rcParams.update(
    {
        "font.size": 14,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "black",
        "grid.color": "#dddddd",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def load_directory() -> dict:
    return json.loads(DIRECTORY_JSON.read_text())


def standard_params(directory: dict | None = None) -> list[str]:
    directory = directory or load_directory()
    return [name for name in directory if name != "hidden_layers"]


def format_algorithm_label(algorithm: str) -> str:
    return ALGORITHM_LABELS.get(algorithm.lower(), algorithm.upper())


def panel_title(objective: str, algorithm: str) -> str:
    return f"{PANEL_BASE_TITLES[objective]} ({format_algorithm_label(algorithm)})"


def load_algorithm_from_args(param: str, objective: str) -> str:
    objective_root = RESULTS_ROOT / param / objective
    args_files = sorted(objective_root.rglob("args.json"))
    if not args_files:
        raise FileNotFoundError(f"No args.json found under {objective_root}")

    algorithms = {
        str(json.loads(args_path.read_text())["algorithm"]).lower() for args_path in args_files
    }
    if len(algorithms) != 1:
        raise ValueError(f"Inconsistent algorithms for {param}/{objective}: {sorted(algorithms)}")

    return algorithms.pop()


def load_param_values(param: str, directory: dict | None = None) -> list[float]:
    directory = directory or load_directory()
    return [float(v) for v in directory[param]["varies"]]


def value_folder_name(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:g}"


def load_series(param: str, objective: str, param_values: list[float]) -> tuple[np.ndarray, np.ndarray]:
    xs: list[float] = []
    ys: list[float] = []
    param_root = RESULTS_ROOT / param / objective

    for value in param_values:
        summary_path = param_root / value_folder_name(value) / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing aggregated summary: {summary_path}")

        summary = json.loads(summary_path.read_text())
        xs.append(value)
        if objective == "speculation_efficiency":
            ys.append(float(summary["average_speculation_efficiency_per_lti"]))
        else:
            ys.append(float(summary["average_hitrate_per_lti"]))

    return np.array(xs, dtype=float), np.array(ys, dtype=float)


def select_poly_degree(x: np.ndarray, y: np.ndarray, *, max_degree: int | None = None) -> int:
    n = len(x)
    if n < 2:
        return 0

    upper = min(max_degree or 4, n - 1)
    best_degree = 1
    best_bic = float("inf")

    for degree in range(1, upper + 1):
        coeffs = np.polyfit(x, y, degree)
        residuals = y - np.polyval(coeffs, x)
        rss = float(np.sum(residuals**2))
        k = degree + 1
        bic = n * np.log(rss / n) + k * np.log(n) if rss > 0 else -float("inf")
        if bic < best_bic:
            best_bic = bic
            best_degree = degree

    return best_degree


def poly_trend(
    x: np.ndarray,
    y: np.ndarray,
    *,
    degree: int | None = None,
    points: int = 200,
) -> tuple[np.ndarray, np.ndarray, int]:
    fit_degree = select_poly_degree(x, y) if degree is None else degree
    coeffs = np.polyfit(x, y, fit_degree)
    x_line = np.linspace(float(x.min()), float(x.max()), points)
    y_line = np.polyval(coeffs, x_line)
    return x_line, y_line, fit_degree


def fmt_param_label(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:g}"


def xlim_padding(param_values: list[float]) -> float:
    span = max(param_values) - min(param_values)
    if span <= 0:
        return 0.5
    return max(span * 0.02, 0.02)


def style_axes_spines(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)


MAX_XTICKS = 6
MAX_TICK_PROXIMITY = 0.08  # drop non-max ticks within this fraction of the x-range from max


def select_xtick_values(param_values: list[float], *, max_ticks: int = MAX_XTICKS) -> list[float]:
    """Pick readable x ticks, always showing max and dropping labels too close to it."""
    values = list(param_values)
    n = len(values)
    if n == 0:
        return []
    if n == 1:
        return values

    min_val, max_val = values[0], values[-1]
    span = max_val - min_val
    min_sep = span * MAX_TICK_PROXIMITY if span > 0 else 0.0

    if n <= max_ticks:
        candidates = values
    else:
        indices = sorted(set(int(i) for i in np.linspace(0, n - 1, max_ticks, dtype=int)))
        candidates = [values[i] for i in indices]

    candidate_set = set(candidates)
    candidate_set.add(min_val)
    candidate_set.add(max_val)

    filtered: list[float] = []
    for tick in sorted(candidate_set):
        if tick == max_val or max_val - tick >= min_sep:
            filtered.append(tick)

    if min_val not in filtered:
        filtered.insert(0, min_val)
    if max_val not in filtered:
        filtered.append(max_val)

    return sorted(set(filtered))


def style_xaxis(
    ax: plt.Axes,
    param: str,
    param_values: list[float],
    *,
    show_xlabel: bool,
) -> None:
    pad = xlim_padding(param_values)
    ax.set_xlim(min(param_values) - pad, max(param_values) + pad)
    tick_values = select_xtick_values(param_values)
    ax.set_xticks(tick_values)
    ax.set_xticklabels([fmt_param_label(v) for v in tick_values], rotation=45, ha="right")
    if show_xlabel:
        ax.set_xlabel(PARAM_XLABELS.get(param, param))
    ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)


ANNOT_CORNERS: dict[str, str | set[str]] = {
    "LFUTimeInterval": "all",
    "LTI": "all",
    "LR": {"speculative_hitrate"},
}


def annotation_corner(param: str, objective: str) -> str:
    placement = ANNOT_CORNERS.get(param)
    if placement == "all" or (isinstance(placement, set) and objective in placement):
        return "top_right"
    return "top_left"


def annotate_value_range(ax: plt.Axes, y: np.ndarray, *, corner: str = "top_left") -> dict[str, float]:
    low_val = float(np.min(y))
    high_val = float(np.max(y))
    delta = high_val - low_val
    text = f"Lower: {low_val:.2f}\nUpper: {high_val:.2f}\nΔ: {delta:.2f}"
    x_pos, ha = (0.97, "right") if corner == "top_right" else (0.03, "left")
    ax.text(
        x_pos,
        0.97,
        text,
        transform=ax.transAxes,
        va="top",
        ha=ha,
        fontsize=ANNOT_FONTSIZE,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "none",
            "linewidth": 0,
            "alpha": 0.92,
        },
        zorder=6,
    )
    return {
        "lowest_value": low_val,
        "highest_value": high_val,
        "difference": delta,
    }


def plot_series_on_axes(
    ax: plt.Axes,
    param: str,
    objective: str,
    style: dict,
    param_values: list[float],
    *,
    star_color: str = BEST_STAR_COLOR,
    star_edgecolor: str = "none",
    star_edgewidth: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    x, y = load_series(param, objective, param_values)
    x_line, y_line, degree = poly_trend(x, y)
    best_idx = int(np.argmax(y))

    ax.scatter(
        x,
        y,
        color=style["color"],
        marker=MARKER,
        s=42,
        edgecolors="none",
        zorder=3,
    )
    ax.plot(
        x_line,
        y_line,
        color=style["color"],
        linestyle=(0, (6, 3)),
        linewidth=1.4,
        alpha=0.75,
        zorder=2,
    )
    ax.plot(
        x[best_idx],
        y[best_idx],
        marker="*",
        markersize=STAR_MARKERSIZE,
        color=star_color,
        markeredgecolor=star_edgecolor,
        markeredgewidth=star_edgewidth,
        linestyle="none",
        zorder=5,
    )
    return x, y, degree, best_idx


# The two hit-rate objectives are drawn together in one dual-y-axis panel.
MERGED_OBJECTIVES = ("speculative_hitrate", "speculativereactive_hitrate")
MERGED_YLABELS = {
    "speculative_hitrate": "Spec. Hit Rate (%)",
    "speculativereactive_hitrate": "Spec.+Reac. Hit Rate (%)",
}


def _series_record(objective: str, algorithm: str, x: np.ndarray, y: np.ndarray, degree: int, best_idx: int) -> dict:
    return {
        "algorithm": algorithm,
        "title": panel_title(objective, algorithm),
        "param_values": x.tolist(),
        "value": y.tolist(),
        "trend_degree": degree,
        "best_param_value": float(x[best_idx]),
        "best_value": float(y[best_idx]),
        "lowest_value": float(np.min(y)),
        "highest_value": float(np.max(y)),
        "difference": float(np.max(y) - np.min(y)),
    }


def _pad_axis_ylim(ax: plt.Axes, y: np.ndarray, *, frac: float = 0.12) -> None:
    lo, hi = float(np.min(y)), float(np.max(y))
    pad = max((hi - lo) * frac, 0.05)
    ax.set_ylim(lo - pad, hi + pad)


def plot_merged_hitrate_panel(
    ax: plt.Axes,
    param: str,
    param_values: list[float],
    data: dict,
) -> None:
    """Draw both hit-rate objectives on one panel using a twin (dual) y-axis."""
    left_obj, right_obj = MERGED_OBJECTIVES
    ax_right = ax.twinx()
    target_axes = {left_obj: ax, right_obj: ax_right}
    legend_handles: list[Line2D] = []

    for objective in MERGED_OBJECTIVES:
        style = SERIES[objective]
        color = style["color"]
        target = target_axes[objective]
        algorithm = load_algorithm_from_args(param, objective)
        x, y, degree, best_idx = plot_series_on_axes(
            target,
            param,
            objective,
            style,
            param_values,
            star_color=color,
            star_edgecolor="white",
            star_edgewidth=0.9,
        )
        _pad_axis_ylim(target, y)
        record = _series_record(objective, algorithm, x, y, degree, best_idx)
        data["series"][objective] = record

        target.set_ylabel(MERGED_YLABELS[objective], color=color)
        target.tick_params(axis="y", colors=color)
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                marker=MARKER,
                markersize=7,
                linestyle=(0, (6, 3)),
                linewidth=1.4,
                label=f"{record['title']}  (Δ={record['difference']:.2f}%)",
            )
        )
    ax.spines["top"].set_visible(False)
    ax_right.spines["top"].set_visible(False)
    ax.spines["left"].set_color(SERIES[left_obj]["color"])
    ax.spines["right"].set_visible(False)
    ax_right.spines["right"].set_color(SERIES[right_obj]["color"])
    ax_right.spines["left"].set_visible(False)

    style_xaxis(ax, param, param_values, show_xlabel=True)
    ax.grid(False)  # style_xaxis added both-axis grid; keep only vertical guides below
    ax.grid(True, axis="x", linestyle=":", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)

    ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        frameon=False,
        fontsize=11,
        handlelength=1.6,
        handletextpad=0.4,
        borderaxespad=0.2,
    )


def _top_delta_legend(ax: plt.Axes, *, title: str, color: str, delta: float, is_hitrate: bool) -> None:
    delta_suffix = "%" if is_hitrate else ""
    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=color,
                marker=MARKER,
                markersize=7,
                linestyle=(0, (6, 3)),
                linewidth=1.4,
                label=f"{title}  (Δ={delta:.2f}{delta_suffix})",
            )
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        frameon=False,
        fontsize=11,
        handlelength=1.6,
        handletextpad=0.4,
        borderaxespad=0.2,
    )


def plot_param(param: str, *, directory: dict | None = None) -> Path:
    directory = directory or load_directory()
    param_values = load_param_values(param, directory)
    data: dict[str, dict] = {"param": param, "series": {}}

    fig, axes = make_panel_grid(2, sharex=True, panel_gap=1.6)

    # Left panel: merged Spec. and Spec.+Reac. hit rate on a dual y-axis.
    plot_merged_hitrate_panel(axes[0], param, param_values, data)

    # Right panel: speculation efficiency.
    eff_objective = "speculation_efficiency"
    eff_style = SERIES[eff_objective]
    eff_color = eff_style["color"]
    eff_algorithm = load_algorithm_from_args(param, eff_objective)
    eff_title = panel_title(eff_objective, eff_algorithm)
    x, y, degree, best_idx = plot_series_on_axes(
        axes[1],
        param,
        eff_objective,
        eff_style,
        param_values,
        star_color=eff_color,
        star_edgecolor="white",
        star_edgewidth=0.9,
    )
    record = _series_record(eff_objective, eff_algorithm, x, y, degree, best_idx)
    data["series"][eff_objective] = record
    _pad_axis_ylim(axes[1], y)
    _top_delta_legend(axes[1], title=eff_title, color=eff_color, delta=record["difference"], is_hitrate=False)
    axes[1].set_ylabel(eff_style["ylabel"])
    style_xaxis(axes[1], param, param_values, show_xlabel=True)
    style_axes_spines(axes[1])

    out_dir = param_plots_dir(param)
    out_base = out_dir / f"{param}_hitrate_spec_eff"
    fig.savefig(out_base.with_suffix(".png"), dpi=180, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

    (out_dir / f"{param}_hitrate_spec_eff_data.json").write_text(json.dumps(data, indent=2))
    return out_base


def print_param_summary(param: str, data: dict) -> None:
    out_dir = param_plots_dir(param)
    print(f"Wrote {out_dir / f'{param}_hitrate_spec_eff'}.pdf/.png")
    print(f"Wrote {out_dir / f'{param}_hitrate_spec_eff_data.json'}")
    for objective in PANEL_ORDER:
        series = data["series"][objective]
        print(
            f"  {series['title']}: degree-{series['trend_degree']} trend, "
            f"best {param}={series['best_param_value']:g} ({series['best_value']:.3f}), "
            f"Δ={series['difference']:.3f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot step-3 param-impact figures.")
    parser.add_argument(
        "--param",
        action="append",
        dest="params",
        help="Parameter to plot (repeatable). Default: all standard params.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    directory = load_directory()
    params = args.params or standard_params(directory)

    for param in params:
        if param not in directory or param == "hidden_layers":
            raise SystemExit(f"Unknown or unsupported param: {param}")
        out_base = plot_param(param, directory=directory)
        data = json.loads((param_plots_dir(param) / f"{param}_hitrate_spec_eff_data.json").read_text())
        print_param_summary(param, data)
        print()


if __name__ == "__main__":
    main()
