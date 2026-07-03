#!/usr/bin/env python3
"""Build best-parameter JSON files from aggregated grid search results.

Scans ``results/grid/{bandit,dqn,ppo}`` ordering folders and, for each
(objective, algorithm, ordering), picks the configuration with the highest
``average_hitrate_per_lti`` or ``average_speculation_efficiency_per_lti``.
Writes a JSON file in the same shape as ``code/best_agingfactor_tablesize50.json``,
extended with algorithm-specific tuning parameters.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

STEP0_DIR = Path(__file__).resolve().parents[1]
GRID_ROOT = STEP0_DIR.parents[2] / "results" / "grid"
EXPERIMENT_ROOT = STEP0_DIR.parents[2] / "results" / "agingfactor_tablesize_experiment_data"
DEFAULT_OUT = STEP0_DIR.parents[2] / "code" / "best_grid_tablesize50.json"

ALGORITHMS = ("bandit", "dqn", "ppo")
ORDERINGS = ("trace", "source", "destination")
OBJECTIVES = (
    "speculative_hitrate",
    "speculativereactive_hitrate",
    "speculativereactive_speculation_efficiency",
)

PATH_PREFIXES = [
    ("numberofFlowsPerAgent_", "numberofFlowsPerAgent", int),
    ("rewardAgingFactor_", "rewardAgingFactor", float),
    ("spatialReward_", "spatialReward", float),
    ("hidden_layers_", "hidden_layers", int),
    ("ppo_epochs_", "ppo_epochs", int),
    ("bandit_c_", "bandit_c", float),
    ("objective_", "objective", str),
    ("mode_", "mode", str),
    ("gamma_", "gamma", float),
    ("dqn_lr_", "dqn_lr", float),
    ("ppo_lr_", "ppo_lr", float),
    ("ordering_", "ordering", str),
]


def parse_folder_parts(parts: tuple[str, ...]) -> dict:
    info: dict = {}
    for part in parts:
        for prefix, key, cast in PATH_PREFIXES:
            if part.startswith(prefix):
                raw = part[len(prefix) :]
                info[key] = cast(raw)
                break
    return info


def read_reactive_hitrate(experiment_root: Path) -> float:
    summary_path = experiment_root / "mode_reactive" / "tablesize_50" / "summary.json"
    with open(summary_path) as f:
        return float(json.load(f)["average_hitrate_per_lti"])


def leaf_args(ordering_dir: Path) -> dict:
    for summary_path in ordering_dir.glob("trace_*/seed_*/args.json"):
        with open(summary_path) as f:
            return json.load(f)
    return {}


def metric_for_objective(objective: str, summary: dict) -> float:
    if objective.endswith("speculation_efficiency"):
        return float(summary["average_speculation_efficiency_per_lti"])
    return float(summary["average_hitrate_per_lti"])


def metric_key_for_objective(objective: str) -> str:
    if objective.endswith("speculation_efficiency"):
        return "speculation_efficiency"
    return "hitrate"


def build_entry(algo: str, info: dict, summary: dict, args: dict, objective: str) -> dict:
    params = {
        "agingfactor": float(args.get("agingfactor", 0)),
        "rewardAgingFactor": float(info["rewardAgingFactor"]),
        "spatialReward": float(info["spatialReward"]),
    }
    if algo == "bandit":
        params["bandit_c"] = float(info["bandit_c"])
    elif algo == "dqn":
        params["numberofFlowsPerAgent"] = int(info["numberofFlowsPerAgent"])
        params["gamma"] = float(info["gamma"])
        params["dqn_lr"] = float(info["dqn_lr"])
        params["hidden_layers"] = int(info["hidden_layers"])
    elif algo == "ppo":
        params["gamma"] = float(info["gamma"])
        params["ppo_lr"] = float(info["ppo_lr"])
        params["hidden_layers"] = int(info["hidden_layers"])
        params["ppo_epochs"] = int(info["ppo_epochs"])

    metric_key = metric_key_for_objective(objective)
    return {
        "params": params,
        metric_key: round(metric_for_objective(objective, summary), 2),
    }


def find_best_per_group(grid_root: Path) -> dict:
    best: dict[tuple[str, str, str], tuple[float, dict]] = {}

    for algo in ALGORITHMS:
        algo_root = grid_root / algo
        if not algo_root.exists():
            continue
        for summary_path in algo_root.rglob("ordering_*/summary.json"):
            ordering_dir = summary_path.parent
            info = parse_folder_parts(ordering_dir.relative_to(algo_root).parts)
            objective = info.get("objective")
            ordering = info.get("ordering")
            if objective not in OBJECTIVES or ordering not in ORDERINGS:
                continue

            with open(summary_path) as f:
                summary = json.load(f)
            args = leaf_args(ordering_dir)
            value = metric_for_objective(objective, summary)
            key = (objective, algo, ordering)
            if key not in best or value > best[key][0]:
                entry = build_entry(algo, info, summary, args, objective)
                best[key] = (value, entry)

    return best


def to_nested_json(best: dict, reactive_hr: float) -> dict:
    out: dict = {"reactive_hitrate": round(reactive_hr, 2)}
    for objective in OBJECTIVES:
        out[objective] = {algo: {} for algo in ALGORITHMS}
        metric_key = metric_key_for_objective(objective)
        for algo in ALGORITHMS:
            for ordering in ORDERINGS:
                key = (objective, algo, ordering)
                if key not in best:
                    continue
                out[objective][algo][ordering] = best[key][1]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-root", type=Path, default=GRID_ROOT)
    parser.add_argument("--experiment-root", type=Path, default=EXPERIMENT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    reactive_hr = read_reactive_hitrate(args.experiment_root.resolve())
    best = find_best_per_group(args.grid_root.resolve())
    payload = to_nested_json(best, reactive_hr)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=4)
        f.write("\n")

    print(f"Wrote {args.output}")
    for objective in OBJECTIVES:
        for algo in ALGORITHMS:
            for ordering in ORDERINGS:
                key = (objective, algo, ordering)
                if key in best:
                    entry = best[key][1]
                    metric = entry[metric_key_for_objective(objective)]
                    print(f"  {objective} / {algo} / {ordering}: {metric}")


if __name__ == "__main__":
    main()
