#!/usr/bin/env python3
"""Plot ternary-search optimization trajectories for the best run per objective."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

STEP2_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = STEP2_DIR.parents[2]
RESULTS_DIR = REPO_ROOT / "results" / "step2_second_tuning"
PLOTS_DIR = STEP2_DIR / "plots"

sys.path.insert(0, str(REPO_ROOT / "param-impact-analysis" / "new" / "step0" / "scripts"))

LEVELS_PER_PARAM = 7
MARKER_SIZE = 5
CONNECT_LINE_WIDTH = 0.7

ORDERING_LABELS = {
    "trace": "Trace",
    "source": "Source",
    "destination": "Destination",
}

# Distinct, colorblind-friendly hues — one color per parameter role.
PARAM_COLORS = {
    "gamma": "#FFB703",
    "agingfactor": "#E63946",
    "spatialReward": "#4361EE",
    "rewardAgingFactor": "#7209B7",
    "dqn_lr": "#06A77D",
    "ppo_lr": "#06A77D",
    "bandit_c": "#00B4D8",
}

PARAM_LABELS = {
    "spatialReward": "SR",
    "gamma": r"$\gamma$",
    "rewardAgingFactor": "RAF",
    "agingfactor": "AF",
    "dqn_lr": "LR",
    "ppo_lr": "LR",
    "bandit_c": "C",
}

OBJECTIVE_CAPTIONS = {
    "speculative_hitrate": "Speculative HR",
    "speculativereactive_hitrate": "Spec.+Reactive HR",
    "speculativereactive_speculation_efficiency": "Speculation Efficiency",
}

BEST_RUNS = [
    {
        "run_id": 3,
        "algorithm": "DQN",
        "objective": "speculative_hitrate",
        "ylabel": "Hit Rate(%)",
        "is_hitrate": True,
    },
    {
        "run_id": 15,
        "algorithm": "PPO",
        "objective": "speculativereactive_hitrate",
        "ylabel": "Hit Rate(%)",
        "is_hitrate": True,
    },
    {
        "run_id": 18,
        "algorithm": "Bandit",
        "objective": "speculativereactive_speculation_efficiency",
        "ylabel": "Speculation Efficiency",
        "is_hitrate": False,
    },
]

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
        "axes.edgecolor": "black",
        "grid.color": "#dddddd",
    }
)


def load_run_command(run_id: int) -> dict:
    with open(RESULTS_DIR / "commands.csv", newline="") as f:
        for row in csv.DictReader(f):
            if int(row["run_id"]) == run_id:
                return row
    raise KeyError(f"run_id {run_id} not found")


def load_grid_start(run_id: int) -> float:
    return float(load_run_command(run_id)["current_value"])


def panel_title(run_info: dict) -> str:
    row = load_run_command(run_info["run_id"])
    ordering = ORDERING_LABELS[row["ordering"].lower()]
    algorithm = run_info["algorithm"]
    return f"{algorithm}({ordering})"


def load_param_order(run_id: int) -> list[str]:
    with open(RESULTS_DIR / str(run_id) / "checkpoint.json") as f:
        return json.load(f)["param_order"]


def extract_segments(run_id: int, grid_start: float) -> list[dict]:
    history = pd.read_csv(RESULTS_DIR / str(run_id) / "best_objective_history.csv")
    param_order = load_param_order(run_id)
    segments: list[dict] = []

    for param in param_order:
        rows = history[history["param_name"] == param].sort_values("param_iter")
        values = [float(v) for v in rows["best_objective"].tolist()]
        if len(values) != LEVELS_PER_PARAM:
            raise ValueError(
                f"run {run_id} param {param}: expected {LEVELS_PER_PARAM} levels, got {len(values)}"
            )
        start_val = grid_start if not segments else segments[-1]["values"][-1]
        delta = values[-1] - start_val
        segments.append({
            "param": param,
            "label": PARAM_LABELS[param],
            "color": PARAM_COLORS[param],
            "values": values,
            "start": start_val,
            "delta": delta,
        })
        grid_start = values[-1]

    if segments and abs(segments[0]["values"][0] - load_grid_start(run_id)) > 1e-6:
        segments[0]["values"][0] = load_grid_start(run_id)

    return segments


def format_delta(delta: float, is_hitrate: bool) -> str:
    if is_hitrate:
        sign = "+" if delta >= 0 else "-"
        return f"{sign}{abs(delta):.2f}%"
    sign = "+" if delta >= 0 else "-"
    return f"{sign}{abs(delta):.2f}"


def style_axes(ax: plt.Axes, *, show_y_tick_labels: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("black")
    ax.spines["left"].set_color("black")
    ax.tick_params(axis="x", colors="black")
    ax.tick_params(axis="y", colors="black", labelleft=show_y_tick_labels)
    ax.xaxis.label.set_color("black")
    ax.yaxis.label.set_color("black")
    ax.grid(True, axis="y", alpha=0.6, linestyle="-", linewidth=0.6)
    ax.grid(False, axis="x")


def add_panel_legend(ax: plt.Axes, handles: list[Line2D]) -> None:
    ax.legend(
        handles=handles,
        loc="upper left",
        frameon=False,
        fontsize=11,
        handlelength=1.2,
        handletextpad=0.3,
        borderaxespad=0.5,
    )


def collect_trajectory_values(run_info: dict) -> list[float]:
    grid_start = load_grid_start(run_info["run_id"])
    segments = extract_segments(run_info["run_id"], grid_start)
    values: list[float] = []
    for seg in segments:
        values.extend(seg["values"])
    return values


def compute_shared_ylim(*, use_grid_delta: bool) -> tuple[float, float]:
    all_values: list[float] = []
    for run_info in BEST_RUNS:
        grid_start = load_grid_start(run_info["run_id"])
        for value in collect_trajectory_values(run_info):
            all_values.append(value - grid_start if use_grid_delta else value)
    y_min, y_max = min(all_values), max(all_values)
    if use_grid_delta:
        tick_step = 0.5
        y_max = max(y_max + tick_step * 0.15, tick_step)
        y_top = tick_step * (int(y_max / tick_step) + (1 if y_max % tick_step > 1e-9 else 0))
        return 0.0, y_top
    pad = max((y_max - y_min) * 0.10, 1.0)
    return y_min - pad, y_max + pad


def apply_shared_y_scale(ax: plt.Axes, ylim: tuple[float, float]) -> None:
    ax.set_ylim(ylim)
    ax.yaxis.set_major_locator(MultipleLocator(0.5))


def plot_run(
    ax: plt.Axes,
    run_info: dict,
    *,
    set_ylim: bool = True,
    use_grid_delta: bool = False,
    show_legend: bool = True,
    show_ylabel: bool = True,
) -> list[Line2D]:
    run_id = run_info["run_id"]
    grid_start = load_grid_start(run_id)
    segments = extract_segments(run_id, grid_start)
    is_hitrate = run_info["is_hitrate"]

    def y_value(raw: float) -> float:
        return raw - grid_start if use_grid_delta else raw

    ys_all: list[float] = []
    legend_handles: list[Line2D] = []

    for seg_idx, seg in enumerate(segments):
        x_offset = seg_idx * LEVELS_PER_PARAM
        xs = [x_offset + level for level in range(LEVELS_PER_PARAM)]
        ys = [y_value(v) for v in seg["values"]]
        ys_all.extend(ys)

        line_x = xs
        line_y = ys
        if seg_idx > 0:
            prev_last_y = y_value(segments[seg_idx - 1]["values"][-1])
            line_x = [xs[0] - 1] + xs
            line_y = [prev_last_y] + ys

        ax.plot(line_x, line_y, color=seg["color"], linestyle="--", linewidth=CONNECT_LINE_WIDTH, zorder=3)
        for x, y in zip(xs, ys):
            ax.plot(
                x,
                y,
                linestyle="none",
                marker="o",
                color=seg["color"],
                markersize=MARKER_SIZE,
                markeredgecolor="white",
                markeredgewidth=0.9,
                zorder=4,
            )

        delta_str = format_delta(seg["delta"], is_hitrate)
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=seg["color"],
                linewidth=CONNECT_LINE_WIDTH,
                linestyle="--",
                marker="o",
                markersize=MARKER_SIZE,
                markeredgecolor="white",
                markeredgewidth=0.9,
                label=f"{seg['label']} ($\\Delta={delta_str}$)",
            )
        )

        if seg_idx > 0:
            ax.axvline(
                x_offset - 0.5,
                color="#cccccc",
                linewidth=0.8,
                linestyle=":",
                zorder=1,
            )

    if set_ylim:
        y_min, y_max = min(ys_all), max(ys_all)
        pad = max((y_max - y_min) * 0.10, 0.05 if use_grid_delta else (0.08 if is_hitrate else 0.8))
        ax.set_ylim(y_min - pad, y_max + pad)

    n_params = len(segments)
    max_iter = n_params * LEVELS_PER_PARAM - 1
    ax.set_xlim(-0.5, max_iter + 0.5)

    tick_positions = list(range(0, max_iter + 2, 5))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([str(t) for t in tick_positions])
    ax.set_xlabel("Tuning Iteration")

    if show_ylabel:
        ax.set_ylabel("Improvement (from Grid Search)" if use_grid_delta else run_info["ylabel"])

    ax.set_title(
        f"{OBJECTIVE_CAPTIONS[run_info['objective']]}\n{panel_title(run_info)}",
        loc="center",
        pad=8,
    )
    if show_legend:
        add_panel_legend(ax, legend_handles)
    style_axes(ax, show_y_tick_labels=show_ylabel)
    ax.tick_params(axis="y", labelsize=11)
    return legend_handles


def save_figure(fig: plt.Figure, base_path: Path) -> None:
    fig.savefig(base_path.with_suffix(".png"), dpi=180, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(base_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.06)


def plot_combined() -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 4.1), sharey=True)
    shared_ylim = compute_shared_ylim(use_grid_delta=True)
    shared_xmax = max(
        len(load_param_order(run_info["run_id"])) * LEVELS_PER_PARAM - 1
        for run_info in BEST_RUNS
    )

    for idx, (ax, run_info) in enumerate(zip(axes, BEST_RUNS)):
        plot_run(
            ax,
            run_info,
            set_ylim=False,
            use_grid_delta=True,
            show_ylabel=idx == 0,
        )
        apply_shared_y_scale(ax, shared_ylim)
        ax.set_xlim(-0.5, shared_xmax + 0.5)
        tick_positions = list(range(0, shared_xmax + 2, 5))
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([str(t) for t in tick_positions])

    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.14, top=0.86, wspace=0.06)

    out_base = PLOTS_DIR / "ternary_search_best_trajectories"
    save_figure(fig, out_base)
    plt.close(fig)
    return out_base


def plot_individual(run_info: dict) -> Path:
    fig, ax = plt.subplots(figsize=(5.5, 3.7))
    plot_run(ax, run_info)
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.14, top=0.84)
    out_base = PLOTS_DIR / f"ternary_search_run{run_info['run_id']}"
    save_figure(fig, out_base)
    plt.close(fig)
    return out_base


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    combined = plot_combined()
    singles = [plot_individual(run_info) for run_info in BEST_RUNS]
    print(f"Wrote {combined}.pdf/.png")
    for path in singles:
        print(f"Wrote {path}.pdf/.png")


if __name__ == "__main__":
    main()
