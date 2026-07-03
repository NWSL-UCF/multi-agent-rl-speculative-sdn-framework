#!/usr/bin/env python3
"""Generate LaTeX/PDF tables from step-2 ternary-search checkpoints."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

STEP2_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = STEP2_DIR.parents[2]
RESULTS_DIR = REPO_ROOT / "results" / "step2_second_tuning"
TABLES_DIR = STEP2_DIR / "table"
DEFAULT_GRID_JSON = REPO_ROOT / "code" / "best_grid_tablesize50.json"

sys.path.insert(0, str(REPO_ROOT / "param-impact-analysis" / "new" / "step0" / "scripts"))
from generate_grid_tablesize50_tables import (  # noqa: E402
    _col_widths_from_content,
    _render_wide_table_pdf,
)
from generate_tablesize50_latex_tables import (  # noqa: E402
    IMPROV_COL,
    ORDER_COL,
    format_diff,
    format_efficiency,
    format_hit_rate,
)

TABLESIZE = 50
ALGORITHMS = ["bandit", "dqn", "ppo"]
ORDERINGS = ["source", "destination", "trace"]
OBJECTIVES = [
    "speculative_hitrate",
    "speculativereactive_hitrate",
    "speculativereactive_speculation_efficiency",
]

ALGORITHM_LABELS = {
    "bandit": "Bandit",
    "dqn": "DQN",
    "ppo": "PPO",
}
TABLE_ORDERING_LABELS = {
    "source": "Src",
    "destination": "Dest",
    "trace": "Trace",
}

ALGO_PARAM_KEYS: dict[str, list[str]] = {
    "bandit": ["agingfactor", "rewardAgingFactor", "spatialReward", "bandit_c"],
    "dqn": ["agingfactor", "rewardAgingFactor", "spatialReward", "gamma", "dqn_lr"],
    "ppo": ["agingfactor", "rewardAgingFactor", "spatialReward", "gamma", "ppo_lr"],
}

PARAM_LABELS: dict[str, str] = {
    "agingfactor": "AF",
    "rewardAgingFactor": "RAF",
    "spatialReward": "SR",
    "bandit_c": "c",
    "gamma": r"$\gamma$",
    "dqn_lr": "LR",
    "ppo_lr": "LR",
}

HR_COL = "HR(%)"
EFF_COL = "Eff."
GRID_IMPROV_COL = "Grid Improv."
GROUP_SPEC_HR = "Spec. HR"
GROUP_SR_HR = "Spec.+Reac. HR"
GROUP_SR_EFF = "Spec. Eff."
EXACT_DISPLAY_PARAMS = {"rewardAgingFactor", "spatialReward", "gamma", "bandit_c"}


def load_history_last_row(run_id: int) -> dict[str, str]:
    path = RESULTS_DIR / str(run_id) / "best_objective_history.csv"
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Empty history: {path}")
    return rows[-1]


def history_params(last_row: dict[str, str]) -> dict[str, str]:
    params: dict[str, str] = {}
    for key, value in last_row.items():
        if key.startswith("value_"):
            params[key[len("value_"):]] = value
    return params


def format_param(key: str, value: str | float | int) -> str:
    if key in EXACT_DISPLAY_PARAMS:
        return str(value)
    numeric = float(value)
    if key == "ppo_lr":
        if numeric < 0.01:
            return f"{numeric:.4f}".rstrip("0").rstrip(".")
        return f"{numeric:g}"
    if key == "dqn_lr":
        return f"{numeric:.2f}"
    if isinstance(value, float) or "." in str(value):
        return f"{numeric:g}"
    return str(value)


def load_results() -> dict[tuple[str, str, str], dict]:
    commands_path = RESULTS_DIR / "commands.csv"
    with open(commands_path, newline="") as f:
        commands = {int(row["run_id"]): row for row in csv.DictReader(f)}

    results: dict[tuple[str, str, str], dict] = {}
    for run_id, cmd in commands.items():
        last_row = load_history_last_row(run_id)
        params = history_params(last_row)
        key = (cmd["algorithm"], cmd["ordering"], cmd["objective"])
        grid_obj = float(cmd["current_value"])
        final_obj = float(last_row["best_objective"])
        results[key] = {
            "params": params,
            "grid_obj": grid_obj,
            "final_obj": final_obj,
            "delta": final_obj - grid_obj,
        }
    return results


def param_cells(algorithm: str, params: dict) -> list[str]:
    return [format_param(key, params[key]) for key in ALGO_PARAM_KEYS[algorithm]]


def load_reactive_hit_rate() -> float:
    with open(DEFAULT_GRID_JSON) as f:
        return float(json.load(f)["reactive_hitrate"])


def group_spans(n_params: int) -> list[tuple[int, int, str]]:
    hr_size = n_params + 3
    eff_size = n_params + 2
    start = 1
    groups: list[tuple[int, int, str]] = []
    for label in (GROUP_SPEC_HR, GROUP_SR_HR):
        end = start + hr_size - 1
        groups.append((start, end, label))
        start = end + 1
    groups.append((start, start + eff_size - 1, GROUP_SR_EFF))
    return groups


def build_algo_table_data(
    algorithm: str,
    results: dict,
    reactive_hr: float,
) -> tuple[list[str], list[list[str]], int]:
    param_headers = [PARAM_LABELS[key] for key in ALGO_PARAM_KEYS[algorithm]]
    n_params = len(param_headers)

    headers = [ORDER_COL]
    headers.extend(param_headers)
    headers.extend([HR_COL, IMPROV_COL, GRID_IMPROV_COL])
    headers.extend(param_headers)
    headers.extend([HR_COL, IMPROV_COL, GRID_IMPROV_COL])
    headers.extend(param_headers)
    headers.extend([EFF_COL, GRID_IMPROV_COL])

    data: list[list[str]] = []
    for ordering in ORDERINGS:
        sp = results[(algorithm, ordering, OBJECTIVES[0])]
        sr = results[(algorithm, ordering, OBJECTIVES[1])]
        eff = results[(algorithm, ordering, OBJECTIVES[2])]

        row = [TABLE_ORDERING_LABELS[ordering]]
        row.extend(param_cells(algorithm, sp["params"]))
        row.extend([
            format_hit_rate(sp["final_obj"]),
            format_diff(sp["final_obj"] - reactive_hr),
            format_diff(sp["delta"]),
        ])
        row.extend(param_cells(algorithm, sr["params"]))
        row.extend([
            format_hit_rate(sr["final_obj"]),
            format_diff(sr["final_obj"] - reactive_hr),
            format_diff(sr["delta"]),
        ])
        row.extend(param_cells(algorithm, eff["params"]))
        row.extend([format_efficiency(eff["final_obj"]), format_diff(eff["delta"])])
        data.append(row)

    return headers, data, n_params


def render_algo_latex_table(
    algorithm: str,
    headers: list[str],
    data: list[list[str]],
    n_params: int,
    reactive_hr: float,
) -> str:
    n_cols = len(headers)
    col_spec = "l" + "c" * (n_cols - 1)
    groups = group_spans(n_params)

    group_line_parts = ["&"]
    cmidrules: list[str] = []
    for start, end, label in groups:
        span = end - start + 1
        group_line_parts.append(rf" \multicolumn{{{span}}}{{c}}{{{label}}} &")
        cmidrules.append(rf"\cmidrule(lr){{{start + 1}-{end + 1}}}")
    group_line = "".join(group_line_parts).rstrip(" &")

    lines = [
        r"% Requires: \usepackage{booktabs}",
        r"\begin{table*}[t]",
        r"\centering",
        rf"\caption{{Best ternary-search configuration for {ALGORITHM_LABELS[algorithm]} "
        rf"(SFT size = {TABLESIZE}). Reactive baseline hit rate: {format_hit_rate(reactive_hr)}\%. "
        r"AF=Aging Factor, RAF=Reward Aging Factor, SR=Spatial Reward. "
        r"Only hyperparameters refined by ternary search are shown; all other settings "
        r"are fixed at the best grid-search values from Step~1. "
        r"Improv.\ columns report hit-rate gain over reactive (\%). "
        r"Grid Improv.\ columns report the objective gain over the Step~1 grid-search "
        r"starting point (hit rate in percentage points; speculation efficiency as an "
        r"absolute difference).}",
        rf"\label{{tab:step2-{algorithm}-tablesize50}}",
        r"\small",
        rf"\begin{{tabular}}{{@{{}}{col_spec}@{{}}}}",
        r"\toprule",
        group_line + r" \\",
        " ".join(cmidrules),
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    for row in data:
        lines.append(" & ".join(row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    return "\n".join(lines) + "\n"


def render_algo_pdf(path: Path, algorithm: str, headers: list[str], data: list[list[str]]) -> None:
    n_params = len(ALGO_PARAM_KEYS[algorithm])
    _render_wide_table_pdf(
        path,
        headers,
        data,
        group_spans(n_params),
        _col_widths_from_content(headers, data),
    )


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    results = load_results()
    reactive_hr = load_reactive_hit_rate()

    for algorithm in ALGORITHMS:
        headers, data, n_params = build_algo_table_data(algorithm, results, reactive_hr)
        tex = render_algo_latex_table(algorithm, headers, data, n_params, reactive_hr)
        tex_path = TABLES_DIR / f"step2_{algorithm}_tablesize50.tex"
        pdf_path = TABLES_DIR / f"step2_{algorithm}_tablesize50.pdf"
        tex_path.write_text(tex)
        render_algo_pdf(pdf_path, algorithm, headers, data)
        print(f"Wrote {tex_path} and {pdf_path}")


if __name__ == "__main__":
    main()
