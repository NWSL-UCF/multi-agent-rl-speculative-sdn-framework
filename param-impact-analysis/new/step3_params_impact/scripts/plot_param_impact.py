#!/usr/bin/env python3
"""Plot step-3 param-impact: three side-by-side panels with data-driven polynomial trends."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

STEP3_DIR = Path(__file__).resolve().parents[1]
RESULTS_ROOT = STEP3_DIR.parents[2] / "results" / "step3_param_impact"
DIRECTORY_JSON = STEP3_DIR / "directory.json"
PLOTS_DIR = STEP3_DIR / "plots"


def param_plots_dir(param: str) -> Path:
    path = PLOTS_DIR / param
    path.mkdir(parents=True, exist_ok=True)
    return path

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
        color=BEST_STAR_COLOR,
        linestyle="none",
        zorder=5,
    )
    return x, y, degree, best_idx


def plot_param(param: str, *, directory: dict | None = None) -> Path:
    directory = directory or load_directory()
    param_values = load_param_values(param, directory)
    data: dict[str, dict] = {"param": param, "series": {}}

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharex=True)

    for ax, objective in zip(axes, PANEL_ORDER):
        style = SERIES[objective]
        algorithm = load_algorithm_from_args(param, objective)
        title = panel_title(objective, algorithm)
        x, y, degree, best_idx = plot_series_on_axes(ax, param, objective, style, param_values)
        annotate_value_range(ax, y, corner=annotation_corner(param, objective))
        data["series"][objective] = {
            "algorithm": algorithm,
            "title": title,
            "param_values": x.tolist(),
            "value": y.tolist(),
            "trend_degree": degree,
            "best_param_value": float(x[best_idx]),
            "best_value": float(y[best_idx]),
            "lowest_value": float(np.min(y)),
            "highest_value": float(np.max(y)),
            "difference": float(np.max(y) - np.min(y)),
        }
        ax.set_title(title, fontsize=14, fontweight="bold", pad=8)
        ax.set_ylabel(style["ylabel"])
        style_xaxis(ax, param, param_values, show_xlabel=True)
        style_axes_spines(ax)

    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.22, top=0.88, wspace=0.20)
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
