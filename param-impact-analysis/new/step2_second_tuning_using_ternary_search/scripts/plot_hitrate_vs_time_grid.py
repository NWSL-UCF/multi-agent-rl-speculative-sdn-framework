#!/usr/bin/env python3
"""Plot 3x3 Hit Rate(%) vs Time grid from step-2 tuned aggregated metrics."""

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
STEP1_GRID_ROOT = REPO_ROOT / "results" / "step1_grid_search"
COMMANDS_CSV = STEP2_DIR / "commands.csv"
REACTIVE_LTI = (
    REPO_ROOT
    / "results"
    / "step0_agingfactor_tablesize_experiment_data"
    / "mode_reactive"
    / "tablesize_50"
    / "lti_metrics.csv"
)
PLOTS_DIR = STEP2_DIR / "plots"
PLOT_DATA_DIR = PLOTS_DIR / "hitrate_vs_time_grid_data"
MATCH_TOLERANCE = 0.011

sys.path.insert(0, str(STEP2_DIR / "scripts"))
from generate_step2_tuning_tables import load_reactive_hit_rate, load_results  # noqa: E402

EWMA_ALPHA = 0.05
LINE_WIDTH = 1.2
AVG_LINE_WIDTH = 0.7
AVG_DASH_PATTERN = (0, (8, 2))

ALGORITHMS = ["bandit", "ppo", "dqn"]
ALGORITHM_LABELS = {"bandit": "Bandit", "ppo": "PPO", "dqn": "DQN"}
ORDERINGS = ["source", "destination", "trace"]
ORDERING_LABELS = {
    "source": "Source",
    "destination": "Destination",
    "trace": "Trace",
}

SERIES = {
    "speculative": {
        "label": "Speculative-Only",
        "color": "#00C853",
        "objective": "speculative_hitrate",
    },
    "speculativereactive": {
        "label": "Speculative+Reactive",
        "color": "#0066FF",
        "objective": "speculativereactive_hitrate",
    },
    "reactive": {
        "label": "Reactive",
        "color": "#FF6600",
    },
}


plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
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


def run_id_for(commands: list[dict], *, algorithm: str, ordering: str, objective: str) -> int:
    for row in commands:
        if (
            row["algorithm"] == algorithm
            and row["ordering"] == ordering
            and row["objective"] == objective
        ):
            return int(row["run_id"])
    raise KeyError(f"No run for {algorithm}/{ordering}/{objective}")


OBJECTIVE_MODE = {
    "speculative_hitrate": "speculative",
    "speculativereactive_hitrate": "speculativereactive",
}

STEP1_PARAM_ORDER: dict[str, list[str]] = {
    "bandit": ["bandit_c", "rewardAgingFactor", "spatialReward"],
    "dqn": [
        "numberofFlowsPerAgent",
        "gamma",
        "dqn_lr",
        "rewardAgingFactor",
        "spatialReward",
        "hidden_layers",
    ],
    "ppo": [
        "gamma",
        "ppo_lr",
        "rewardAgingFactor",
        "spatialReward",
        "hidden_layers",
        "ppo_epochs",
    ],
}


def fmt_step1_num(value: float | int) -> str:
    value = float(value)
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:g}"


def load_history(run_id: int) -> pd.DataFrame:
    history_path = RESULTS_DIR / str(run_id) / "best_objective_history.csv"
    history = pd.read_csv(history_path)
    if history.empty:
        raise ValueError(f"Empty history: {history_path}")
    return history


def last_history_row(run_id: int) -> pd.Series:
    return load_history(run_id).iloc[-1]


def history_all_center(history: pd.DataFrame) -> bool:
    return bool((history["decision"] == "center").all())


def last_committed_row(history: pd.DataFrame) -> tuple[pd.Series, int]:
    for idx in range(len(history) - 1, -1, -1):
        row = history.iloc[idx]
        if str(row["decision"]) in {"mid_left", "mid_right"}:
            return row, idx
    raise ValueError("No committed mid_left/mid_right decision in history")


def committed_aggregated_csv(run_id: int, row: pd.Series) -> Path:
    decision = str(row["decision"])
    return (
        RESULTS_DIR
        / str(run_id)
        / "aggregated"
        / f"{row['param_name']}_iter{int(row['param_iter'])}_{decision}.csv"
    )


def step1_params(cmd: dict, last: pd.Series) -> dict[str, float | int]:
    params = {
        str(col).replace("value_", ""): float(last[col])
        for col in last.index
        if str(col).startswith("value_")
    }
    algorithm = cmd["algorithm"]
    for key in STEP1_PARAM_ORDER[algorithm]:
        raw = cmd.get(key, "")
        if raw == "":
            continue
        params[key] = float(raw) if "." in str(raw) else int(raw)
    return params


def step1_ordering_lti_path(cmd: dict, last: pd.Series) -> Path:
    algorithm = cmd["algorithm"]
    objective = cmd["objective"]
    params = step1_params(cmd, last)
    parts = [
        STEP1_GRID_ROOT / algorithm,
        f"mode_{OBJECTIVE_MODE[objective]}",
        f"objective_{objective}",
    ]
    for key in STEP1_PARAM_ORDER[algorithm]:
        parts.append(f"{key}_{fmt_step1_num(params[key])}")
    parts.append(f"ordering_{cmd['ordering']}")
    return Path(*parts) / "lti_metrics.csv"


def mean_hitrate_file(path: Path) -> float:
    df = pd.read_csv(path)
    column = "hitrate" if "hitrate" in df.columns else "hit_rate"
    return float(df[column].mean())


def resolve_hitrate_csv(
    run_id: int, cmd: dict
) -> tuple[Path, str, float, pd.Series, pd.Series | None]:
    history = load_history(run_id)
    last = history.iloc[-1]
    if history_all_center(history):
        source = step1_ordering_lti_path(cmd, last)
        if not source.exists():
            raise FileNotFoundError(f"Missing step1 aggregated LTI for run {run_id}: {source}")
        return source, "step1_all_center", mean_hitrate_file(source), last, None

    committed, _ = last_committed_row(history)
    source = committed_aggregated_csv(run_id, committed)
    if not source.exists():
        raise FileNotFoundError(f"Missing committed aggregated CSV for run {run_id}: {source}")
    return source, "history_walk", mean_hitrate_file(source), last, committed


def copy_plot_data(source: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)


def ewma(series: pd.Series, alpha: float = EWMA_ALPHA) -> pd.Series:
    return series.ewm(alpha=alpha, adjust=False).mean()


def load_step2_series(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "second_bucket" in df.columns:
        time_col = "second_bucket"
        hr_col = "hitrate"
    else:
        time_col = "lti_start_time"
        hr_col = "hit_rate"
    return pd.DataFrame(
        {
            "time": df[time_col].astype(float),
            "hit_rate": ewma(df[hr_col].astype(float)),
        }
    )


def load_reactive_series() -> pd.DataFrame:
    df = pd.read_csv(REACTIVE_LTI)
    return pd.DataFrame(
        {
            "time": df["lti_start_time"].astype(float),
            "hit_rate": ewma(df["hit_rate"].astype(float)),
        }
    )


def table_averages(
    results: dict,
    *,
    algorithm: str,
    ordering: str,
    reactive_avg: float,
) -> tuple[float, float, float]:
    spec_avg = results[(algorithm, ordering, "speculative_hitrate")]["final_obj"]
    sr_avg = results[(algorithm, ordering, "speculativereactive_hitrate")]["final_obj"]
    return reactive_avg, spec_avg, sr_avg


def plot_avg_line(ax: plt.Axes, avg_hitrate: float, color: str) -> None:
    ax.axhline(
        avg_hitrate,
        color=color,
        linestyle=AVG_DASH_PATTERN,
        linewidth=AVG_LINE_WIDTH,
        zorder=2,
    )


def add_avg_labels(
    ax: plt.Axes,
    *,
    reactive_avg: float,
    spec_avg: float,
    sr_avg: float,
) -> None:
    entries = [
        ("speculativereactive", "Spec.+Reac.", sr_avg),
        ("reactive", "Reac.", reactive_avg),
        ("speculative", "Spec.", spec_avg),
    ]
    x_text = 0.97
    y_base = 0.06
    line_step = 0.078

    for idx, (series_key, label, avg) in enumerate(entries):
        y = y_base + (len(entries) - 1 - idx) * line_step
        color = SERIES[series_key]["color"]
        ax.text(
            x_text,
            y,
            f"{label}: {avg:.2f}%",
            transform=ax.transAxes,
            va="center",
            ha="right",
            fontsize=8,
            color=color,
            clip_on=False,
            zorder=5,
        )


def style_axes(ax: plt.Axes, *, show_ylabel: bool, show_xlabel: bool) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("black")
    ax.spines["left"].set_color("black")
    ax.tick_params(axis="x", colors="black", labelbottom=show_xlabel)
    ax.tick_params(axis="y", colors="black", labelleft=show_ylabel)
    if show_xlabel:
        ax.set_xlabel("Time (s)")
    if show_ylabel:
        ax.set_ylabel("Hit Rate(%)")
    ax.xaxis.label.set_color("black")
    ax.yaxis.label.set_color("black")
    ax.grid(True, axis="y", alpha=0.6, linestyle="-", linewidth=0.6)
    ax.grid(False, axis="x")


def verify_last_best_objective(manifest: list[dict]) -> None:
    print("\nVerification: mean(LTI hit rate) vs last-row best_objective")
    print(
        f"  {'run':>3} {'algo':>6} {'ord':>4} {'obj':>8} {'last_best':>9} {'csv':>7} "
        f"{'diff':>7} method  file"
    )
    ok = 0
    total = 0
    for entry in manifest:
        if entry["objective"] == "reactive":
            continue
        total += 1
        diff = entry["csv_mean"] - entry["last_best_objective"]
        match = abs(diff) <= MATCH_TOLERANCE
        ok += int(match)
        print(
            f"  {entry['run_id']:>3} {entry['algorithm']:>6} {entry['ordering']:>4} "
            f"{entry['objective_short']:>8} {entry['last_best_objective']:>9.2f} "
            f"{entry['csv_mean']:>7.2f} {diff:>+7.2f} {entry['method']:>7}  {entry['source_name']}"
        )
    print(f"  Matches within {MATCH_TOLERANCE:.3f}: {ok}/{total}")


def plot_grid(
    commands: list[dict],
    results: dict,
    reactive: pd.DataFrame,
    reactive_avg: float,
    manifest: list[dict],
) -> Path:
    fig, axes = plt.subplots(3, 3, figsize=(11.0, 5.5), sharex=True, sharey=True)

    for row_idx, algorithm in enumerate(ALGORITHMS):
        for col_idx, ordering in enumerate(ORDERINGS):
            ax = axes[row_idx, col_idx]
            spec_run = run_id_for(
                commands,
                algorithm=algorithm,
                ordering=ordering,
                objective="speculative_hitrate",
            )
            sr_run = run_id_for(
                commands,
                algorithm=algorithm,
                ordering=ordering,
                objective="speculativereactive_hitrate",
            )

            spec_path = PLOT_DATA_DIR / f"run{spec_run:02d}_speculative_hitrate.csv"
            sr_path = PLOT_DATA_DIR / f"run{sr_run:02d}_speculativereactive_hitrate.csv"
            spec = load_step2_series(spec_path)
            sr = load_step2_series(sr_path)
            _, spec_avg, sr_avg = table_averages(
                results,
                algorithm=algorithm,
                ordering=ordering,
                reactive_avg=reactive_avg,
            )

            ax.plot(
                reactive["time"],
                reactive["hit_rate"],
                color=SERIES["reactive"]["color"],
                linewidth=LINE_WIDTH,
                zorder=3,
            )
            ax.plot(
                spec["time"],
                spec["hit_rate"],
                color=SERIES["speculative"]["color"],
                linewidth=LINE_WIDTH,
                zorder=3,
            )
            ax.plot(
                sr["time"],
                sr["hit_rate"],
                color=SERIES["speculativereactive"]["color"],
                linewidth=LINE_WIDTH,
                zorder=3,
            )
            plot_avg_line(ax, reactive_avg, SERIES["reactive"]["color"])
            plot_avg_line(ax, spec_avg, SERIES["speculative"]["color"])
            plot_avg_line(ax, sr_avg, SERIES["speculativereactive"]["color"])
            add_avg_labels(
                ax,
                reactive_avg=reactive_avg,
                spec_avg=spec_avg,
                sr_avg=sr_avg,
            )

            ax.set_title(
                f"{ALGORITHM_LABELS[algorithm]}({ORDERING_LABELS[ordering]})",
                pad=2,
            )
            style_axes(
                ax,
                show_ylabel=col_idx == 0,
                show_xlabel=row_idx == 2,
            )

    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.95, wspace=0.12, hspace=0.22)

    out_base = PLOTS_DIR / "hitrate_vs_time_grid"
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=180, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    return out_base


def prepare_plot_data(commands: list[dict], results: dict, reactive_avg: float) -> list[dict]:
    if PLOT_DATA_DIR.exists():
        for path in PLOT_DATA_DIR.iterdir():
            if path.is_file():
                path.unlink()
    else:
        PLOT_DATA_DIR.mkdir(parents=True)

    reactive_dest = PLOT_DATA_DIR / "reactive_lti_metrics.csv"
    copy_plot_data(REACTIVE_LTI, reactive_dest)

    manifest: list[dict] = [
        {
            "kind": "reactive",
            "objective": "reactive",
            "objective_short": "reactive",
            "table_hr": reactive_avg,
            "csv_mean": float(pd.read_csv(reactive_dest)["hit_rate"].mean()),
            "source": str(REACTIVE_LTI),
            "source_name": REACTIVE_LTI.name,
            "dest": str(reactive_dest),
            "method": "baseline",
        }
    ]

    commands_by_run = {int(row["run_id"]): row for row in commands}

    for algorithm in ALGORITHMS:
        for ordering in ORDERINGS:
            for objective, objective_short in (
                ("speculative_hitrate", "spec"),
                ("speculativereactive_hitrate", "spec+reac"),
            ):
                run_id = run_id_for(
                    commands,
                    algorithm=algorithm,
                    ordering=ordering,
                    objective=objective,
                )
                table_hr = results[(algorithm, ordering, objective)]["final_obj"]
                cmd = commands_by_run[run_id]
                source, method, csv_mean, last, committed = resolve_hitrate_csv(run_id, cmd)
                dest = PLOT_DATA_DIR / f"run{run_id:02d}_{objective}.csv"
                copy_plot_data(source, dest)
                entry = {
                    "run_id": run_id,
                    "algorithm": algorithm,
                    "ordering": ordering,
                    "objective": objective,
                    "objective_short": objective_short,
                    "table_hr": table_hr,
                    "last_best_objective": float(last["best_objective"]),
                    "csv_mean": csv_mean,
                    "source": str(source),
                    "source_name": source.name,
                    "dest": str(dest),
                    "method": method,
                    "last_param": str(last["param_name"]),
                    "last_decision": str(last["decision"]),
                    "last_param_iter": int(last["param_iter"]),
                }
                if committed is not None:
                    entry.update(
                        {
                            "committed_param": str(committed["param_name"]),
                            "committed_decision": str(committed["decision"]),
                            "committed_param_iter": int(committed["param_iter"]),
                        }
                    )
                manifest.append(entry)

    manifest_path = PLOT_DATA_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    commands = load_commands()
    results = load_results()
    reactive_avg = load_reactive_hit_rate()
    manifest = prepare_plot_data(commands, results, reactive_avg)
    reactive = load_reactive_series()
    out_base = plot_grid(commands, results, reactive, reactive_avg, manifest)
    verify_last_best_objective(manifest)
    print(f"\nWrote {out_base}.pdf/.png")
    print(f"Copied aggregated selections to {PLOT_DATA_DIR}/")
    print(f"Run mapping: {COMMANDS_CSV}")
    print(
        "Selection rule: all-center history -> step1 ordering lti_metrics.csv; "
        "else walk history backward to last mid_left/mid_right aggregated CSV"
    )
    print(f"EWMA alpha: {EWMA_ALPHA}")


if __name__ == "__main__":
    main()
