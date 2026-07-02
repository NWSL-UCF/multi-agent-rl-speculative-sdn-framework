#!/usr/bin/env python3
"""Plot Speculation Efficiency vs Time for all RL methods by flow ordering."""

from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

STEP2_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = STEP2_DIR.parents[2]
RESULTS_DIR = REPO_ROOT / "results" / "step2_second_tuning"
COMMANDS_CSV = STEP2_DIR / "commands.csv"
PLOTS_DIR = STEP2_DIR / "plots"
PLOT_DATA_DIR = PLOTS_DIR / "speculation_efficiency_vs_time_grid_data"
OBJECTIVE = "speculativereactive_speculation_efficiency"
MATCH_TOLERANCE = 0.05

sys.path.insert(0, str(STEP2_DIR / "scripts"))
from generate_step2_tuning_tables import load_results  # noqa: E402

EWMA_ALPHA = 0.05
LINE_WIDTH = 1.2
AVG_LINE_WIDTH = 0.7
AVG_DASH_PATTERN = (0, (8, 2))

# One row height matches hitrate_vs_time_grid (3×3, fig height 5.5, hspace 0.22).
HITRATE_FIG_HEIGHT = 5.5
HITRATE_ROWS = 3
HITRATE_HSPACE = 0.22
PANEL_FIG_HEIGHT = HITRATE_FIG_HEIGHT / (
    HITRATE_ROWS + (HITRATE_ROWS - 1) * HITRATE_HSPACE
)

ALGORITHMS = ["bandit", "ppo", "dqn"]
ALGORITHM_LABELS = {"bandit": "Bandit", "ppo": "PPO", "dqn": "DQN"}
ORDERINGS = ["source", "destination", "trace"]
ORDERING_LABELS = {
    "source": "Source",
    "destination": "Destination",
    "trace": "Trace",
}

RL_COLORS = {
    "bandit": "#0066FF",
    "ppo": "#00C853",
    "dqn": "#FF6600",
}


plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "black",
        "grid.color": "#dddddd",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def load_commands() -> list[dict]:
    with open(COMMANDS_CSV, newline="") as f:
        return list(csv.DictReader(f))


def run_id_for(commands: list[dict], *, algorithm: str, ordering: str) -> int:
    for row in commands:
        if (
            row["algorithm"] == algorithm
            and row["ordering"] == ordering
            and row["objective"] == OBJECTIVE
        ):
            return int(row["run_id"])
    raise KeyError(f"No run for {algorithm}/{ordering}/{OBJECTIVE}")


def mean_aggregated_metric(path: Path, column: str = "speculation_efficiency") -> float:
    return float(pd.read_csv(path)[column].mean())


def iter6_candidates(run_id: int) -> tuple[pd.Series, list[Path]]:
    history_path = RESULTS_DIR / str(run_id) / "best_objective_history.csv"
    history = pd.read_csv(history_path)
    if history.empty:
        raise ValueError(f"Empty history: {history_path}")
    last = history.iloc[-1]
    param_name = str(last["param_name"])
    decision = str(last["decision"])
    agg_dir = RESULTS_DIR / str(run_id) / "aggregated"

    if decision in {"mid_left", "mid_right"}:
        paths = [agg_dir / f"{param_name}_iter6_{decision}.csv"]
    else:
        target = float(last["best_objective"])
        paths: list[Path] = []
        mid_left = agg_dir / f"{param_name}_iter6_mid_left.csv"
        mid_right = agg_dir / f"{param_name}_iter6_mid_right.csv"
        if mid_left.exists() and abs(float(last["obj_mid_left"]) - target) < 1e-6:
            paths.append(mid_left)
        if mid_right.exists() and abs(float(last["obj_mid_right"]) - target) < 1e-6:
            paths.append(mid_right)
        if not paths:
            if mid_left.exists():
                paths.append(mid_left)
            if mid_right.exists():
                paths.append(mid_right)
    existing = [path for path in paths if path.exists()]
    if not existing:
        raise FileNotFoundError(
            f"No iter6 aggregated CSV for run {run_id} "
            f"(param={param_name}, decision={decision})"
        )
    return last, existing


def last_history_row(run_id: int) -> pd.Series:
    history_path = RESULTS_DIR / str(run_id) / "best_objective_history.csv"
    history = pd.read_csv(history_path)
    if history.empty:
        raise ValueError(f"Empty history: {history_path}")
    return history.iloc[-1]


def resolve_aggregated_csv(run_id: int, table_value: float) -> tuple[Path, str, float]:
    agg_dir = RESULTS_DIR / str(run_id) / "aggregated"
    try:
        last, candidates = iter6_candidates(run_id)
        target = float(last["best_objective"])
        best_path = min(
            candidates,
            key=lambda path: abs(mean_aggregated_metric(path) - target),
        )
        best_mean = mean_aggregated_metric(best_path)
        method = "iter6"
    except FileNotFoundError:
        best_path = min(
            agg_dir.glob("*.csv"),
            key=lambda path: abs(mean_aggregated_metric(path) - table_value),
        )
        best_mean = mean_aggregated_metric(best_path)
        method = "fallback"

    if abs(best_mean - table_value) > MATCH_TOLERANCE:
        fallback = min(
            agg_dir.glob("*.csv"),
            key=lambda path: abs(mean_aggregated_metric(path) - table_value),
        )
        fallback_mean = mean_aggregated_metric(fallback)
        if abs(fallback_mean - table_value) < abs(best_mean - table_value):
            best_path = fallback
            best_mean = fallback_mean
            method = "fallback"
    return best_path, method, best_mean


def ewma(series: pd.Series, alpha: float = EWMA_ALPHA) -> pd.Series:
    return series.ewm(alpha=alpha, adjust=False).mean()


def load_efficiency_series(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return pd.DataFrame(
        {
            "time": df["second_bucket"].astype(float),
            "efficiency": ewma(df["speculation_efficiency"].astype(float)),
        }
    )


def plot_avg_line(ax: plt.Axes, avg_value: float, color: str) -> None:
    ax.axhline(
        avg_value,
        color=color,
        linestyle=AVG_DASH_PATTERN,
        linewidth=AVG_LINE_WIDTH,
        zorder=2,
    )


def add_rl_avg_labels(
    ax: plt.Axes,
    averages: list[tuple[str, float]],
    *,
    position: str = "bottom_right",
) -> None:
    line_step = 0.085

    if position == "top_center":
        x_text = 0.5
        y_base = 0.92
        ha = "center"
        entries = averages
        for idx, (algorithm, avg) in enumerate(entries):
            y = y_base - idx * line_step
            color = RL_COLORS[algorithm]
            label = ALGORITHM_LABELS[algorithm]
            ax.text(
                x_text,
                y,
                f"{label}: {avg:.2f}",
                transform=ax.transAxes,
                va="center",
                ha=ha,
                fontsize=8,
                color=color,
                clip_on=False,
                zorder=5,
            )
        return

    x_text = 0.97
    y_base = 0.08

    for idx, (algorithm, avg) in enumerate(reversed(averages)):
        y = y_base + idx * line_step
        color = RL_COLORS[algorithm]
        label = ALGORITHM_LABELS[algorithm]
        ax.text(
            x_text,
            y,
            f"{label}: {avg:.2f}",
            transform=ax.transAxes,
            va="center",
            ha="right",
            fontsize=8,
            color=color,
            clip_on=False,
            zorder=5,
        )


def style_axes(ax: plt.Axes, *, show_ylabel: bool) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("black")
    ax.spines["left"].set_color("black")
    ax.tick_params(axis="x", colors="black")
    ax.tick_params(axis="y", colors="black", labelleft=show_ylabel)
    ax.set_xlabel("Time (s)")
    if show_ylabel:
        ax.set_ylabel("Spec. Eff.")
    ax.xaxis.label.set_color("black")
    ax.yaxis.label.set_color("black")
    ax.grid(True, axis="y", alpha=0.6, linestyle="-", linewidth=0.6)
    ax.grid(False, axis="x")


def prepare_plot_data(commands: list[dict], results: dict) -> list[dict]:
    if PLOT_DATA_DIR.exists():
        for path in PLOT_DATA_DIR.iterdir():
            if path.is_file():
                path.unlink()
    else:
        PLOT_DATA_DIR.mkdir(parents=True)

    manifest: list[dict] = []
    for ordering in ORDERINGS:
        for algorithm in ALGORITHMS:
            run_id = run_id_for(commands, algorithm=algorithm, ordering=ordering)
            table_value = results[(algorithm, ordering, OBJECTIVE)]["final_obj"]
            source, method, csv_mean = resolve_aggregated_csv(run_id, table_value)
            dest = PLOT_DATA_DIR / f"run{run_id:02d}_{algorithm}_{ordering}.csv"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
            last = last_history_row(run_id)
            manifest.append(
                {
                    "run_id": run_id,
                    "algorithm": algorithm,
                    "ordering": ordering,
                    "table_eff": table_value,
                    "csv_mean": csv_mean,
                    "source": str(source),
                    "source_name": source.name,
                    "dest": str(dest),
                    "method": method,
                    "last_param": str(last["param_name"]),
                    "last_decision": str(last["decision"]),
                }
            )

    (PLOT_DATA_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def verify_table_vs_csv(manifest: list[dict]) -> None:
    print("\nVerification: table Spec. Eff. vs mean(speculation_efficiency) from selected CSV")
    print(f"  {'run':>3} {'algo':>6} {'ord':>4} {'table':>7} {'csv':>7} {'diff':>7} method  file")
    ok = 0
    for entry in manifest:
        diff = entry["table_eff"] - entry["csv_mean"]
        match = abs(diff) <= MATCH_TOLERANCE
        ok += int(match)
        print(
            f"  {entry['run_id']:>3} {entry['algorithm']:>6} {entry['ordering']:>4} "
            f"{entry['table_eff']:>7.2f} {entry['csv_mean']:>7.2f} {diff:>+7.2f} "
            f"{entry['method']:>7}  {entry['source_name']}"
        )
    print(f"  Matches within {MATCH_TOLERANCE:.2f}: {ok}/{len(manifest)}")


def plot_grid(commands: list[dict], results: dict) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(11.0, PANEL_FIG_HEIGHT), sharex=True, sharey=True)

    for col_idx, ordering in enumerate(ORDERINGS):
        ax = axes[col_idx]
        averages: list[tuple[str, float]] = []

        for algorithm in ALGORITHMS:
            run_id = run_id_for(commands, algorithm=algorithm, ordering=ordering)
            csv_path = PLOT_DATA_DIR / f"run{run_id:02d}_{algorithm}_{ordering}.csv"
            series = load_efficiency_series(csv_path)
            table_avg = results[(algorithm, ordering, OBJECTIVE)]["final_obj"]
            color = RL_COLORS[algorithm]

            ax.plot(
                series["time"],
                series["efficiency"],
                color=color,
                linewidth=LINE_WIDTH,
                zorder=3,
            )
            plot_avg_line(ax, table_avg, color)
            averages.append((algorithm, table_avg))

        legend_position = "top_center" if col_idx < 2 else "bottom_right"
        add_rl_avg_labels(ax, averages, position=legend_position)
        ax.set_title(ORDERING_LABELS[ordering], pad=2)
        style_axes(ax, show_ylabel=col_idx == 0)

    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.95, wspace=0.12)

    out_base = PLOTS_DIR / "speculation_efficiency_vs_time_grid"
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=180, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    return out_base


def main() -> None:
    commands = load_commands()
    results = load_results()
    manifest = prepare_plot_data(commands, results)
    out_base = plot_grid(commands, results)
    verify_table_vs_csv(manifest)
    print(f"\nWrote {out_base}.pdf/.png")
    print(f"Copied aggregated selections to {PLOT_DATA_DIR}/")
    print(f"EWMA alpha: {EWMA_ALPHA}")


if __name__ == "__main__":
    main()
