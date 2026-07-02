#!/usr/bin/env python3
"""Build best-parameter JSON files after step-2 ternary search.

For each objective, picks the single best configuration across all RL methods
and flow orderings. Tuned hyperparameters come from step-2; fixed settings from
step-1 grid search for the matching (objective, algorithm, ordering).

Outputs are organized under best/:
  all_rl/   - best across Bandit, PPO, and DQN
  ppo_dqn/  - best between PPO and DQN
  dqn/      - best DQN-only configs
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

STEP2_DIR = Path(__file__).resolve().parents[1]
BEST_DIR = STEP2_DIR / "best"
REPO_ROOT = STEP2_DIR.parents[2]
STEP1_GRID_JSON = REPO_ROOT / "code" / "best_grid_tablesize50.json"

sys.path.insert(0, str(STEP2_DIR / "scripts"))
from generate_step2_tuning_tables import (  # noqa: E402
    ALGORITHMS,
    ALGO_PARAM_KEYS,
    ORDERINGS,
    load_reactive_hit_rate,
    load_results,
)

STEP0_TEMPLATE_JSON = (
    STEP2_DIR.parent
    / "step0"
    / "step0_initial_combined_agingfactor_and_tablesize_analysis_on_conference_version_params.json"
)

ALL_RL_DIR = BEST_DIR / "all_rl"
PPO_DQN_DIR = BEST_DIR / "ppo_dqn"
DQN_DIR = BEST_DIR / "dqn"

OUTPUTS: dict[str, str] = {
    "speculative_hitrate": "best_speculative_only_hitrate.json",
    "speculativereactive_hitrate": "best_speculativereactive_hitrate.json",
    "speculativereactive_speculation_efficiency": "best_spec_eff.json",
}

EXPERIMENT_OUTPUTS: dict[str, str] = {
    "speculative_hitrate": "best_speculative_only_hitrate_params.json",
    "speculativereactive_hitrate": "best_speculativereactive_hitrate_params.json",
    "speculativereactive_speculation_efficiency": "best_spec_eff_params.json",
}

PPO_DQN_OUTPUTS: dict[str, str] = {
    "speculative_hitrate": "best_speculative_only_hitrate_ppo_dqn.json",
    "speculativereactive_hitrate": "best_speculativereactive_hitrate_ppo_dqn.json",
    "speculativereactive_speculation_efficiency": "best_spec_eff_ppo_dqn.json",
}

PPO_DQN_EXPERIMENT_OUTPUTS: dict[str, str] = {
    "speculative_hitrate": "best_speculative_only_hitrate_ppo_dqn_params.json",
    "speculativereactive_hitrate": "best_speculativereactive_hitrate_ppo_dqn_params.json",
    "speculativereactive_speculation_efficiency": "best_spec_eff_ppo_dqn_params.json",
}

OBJECTIVE_MODES: dict[str, str] = {
    "speculative_hitrate": "speculative",
    "speculativereactive_hitrate": "speculativereactive",
    "speculativereactive_speculation_efficiency": "speculativereactive",
}

PPO_DQN_ALGORITHMS = ["ppo", "dqn"]


def metric_key(objective: str) -> str:
    if objective.endswith("speculation_efficiency"):
        return "speculation_efficiency"
    return "hitrate"


def merge_params(
    *,
    algorithm: str,
    objective: str,
    ordering: str,
    step2_params: dict[str, float | int],
    step1_grid: dict,
) -> dict[str, float | int]:
    step1_params = step1_grid[objective][algorithm][ordering]["params"]
    merged = dict(step1_params)
    for key in ALGO_PARAM_KEYS[algorithm]:
        merged[key] = step2_params[key]
    return merged


def find_best_overall(
    objective: str,
    results: dict,
    step1_grid: dict,
    *,
    algorithms: list[str] | None = None,
) -> dict:
    mkey = metric_key(objective)
    best_algorithm = ""
    best_ordering = ""
    best_value = float("-inf")
    best_params: dict[str, float | int] = {}
    search_algorithms = algorithms or ALGORITHMS

    for algorithm in search_algorithms:
        for ordering in ORDERINGS:
            result = results[(algorithm, ordering, objective)]
            value = float(result["final_obj"])
            if value > best_value:
                best_value = value
                best_algorithm = algorithm
                best_ordering = ordering
                best_params = merge_params(
                    algorithm=algorithm,
                    objective=objective,
                    ordering=ordering,
                    step2_params=result["params"],
                    step1_grid=step1_grid,
                )

    return {
        "reactive_hitrate": load_reactive_hit_rate(),
        "algorithm": best_algorithm,
        "ordering": best_ordering,
        "params": best_params,
        mkey: round(best_value, 2),
    }


def build_experiment_params(best: dict, objective: str) -> dict:
    with open(STEP0_TEMPLATE_JSON) as f:
        payload = json.load(f)

    payload["tablesize"] = [50]
    payload["algorithm"] = [best["algorithm"]]
    payload["ordering"] = [best["ordering"]]
    payload["mode"] = [OBJECTIVE_MODES[objective]]
    for key, value in best["params"].items():
        payload[key] = [value]
    return payload


def write_step0_style_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["{"]
    items = list(payload.items())
    for idx, (key, value) in enumerate(items):
        rendered = json.dumps(value, separators=(", ", ": "))
        comma = "," if idx < len(items) - 1 else ""
        lines.append(f'    "{key}": {rendered}{comma}')
    lines.append("}")
    lines.append("")
    path.write_text("\n".join(lines))


def write_best_outputs(
    *,
    out_dir: Path,
    objective: str,
    summary_filename: str,
    params_filename: str,
    payload: dict,
    label: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / summary_filename
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=4)
        f.write("\n")

    mkey = metric_key(objective)
    print(
        f"Wrote {out_path} ({label}): {payload['algorithm']}/{payload['ordering']} "
        f"{mkey}={payload[mkey]}"
    )

    experiment = build_experiment_params(payload, objective)
    exp_path = out_dir / params_filename
    write_step0_style_json(exp_path, experiment)
    print(f"Wrote {exp_path} ({label})")


def write_params_only(
    *,
    out_dir: Path,
    objective: str,
    params_filename: str,
    payload: dict,
    label: str,
) -> None:
    experiment = build_experiment_params(payload, objective)
    exp_path = out_dir / params_filename
    write_step0_style_json(exp_path, experiment)
    mkey = metric_key(objective)
    print(
        f"Wrote {exp_path} ({label}): {payload['algorithm']}/{payload['ordering']} "
        f"{mkey}={payload[mkey]}"
    )


def main() -> None:
    with open(STEP1_GRID_JSON) as f:
        step1_grid = json.load(f)

    results = load_results()

    for objective, filename in OUTPUTS.items():
        payload = find_best_overall(objective, results, step1_grid)
        write_best_outputs(
            out_dir=ALL_RL_DIR,
            objective=objective,
            summary_filename=filename,
            params_filename=EXPERIMENT_OUTPUTS[objective],
            payload=payload,
            label="all RL",
        )

    for objective, filename in PPO_DQN_OUTPUTS.items():
        payload = find_best_overall(
            objective,
            results,
            step1_grid,
            algorithms=PPO_DQN_ALGORITHMS,
        )
        write_best_outputs(
            out_dir=PPO_DQN_DIR,
            objective=objective,
            summary_filename=filename,
            params_filename=PPO_DQN_EXPERIMENT_OUTPUTS[objective],
            payload=payload,
            label="PPO vs DQN",
        )

    dqn_sr = find_best_overall(
        "speculativereactive_hitrate",
        results,
        step1_grid,
        algorithms=["dqn"],
    )
    write_params_only(
        out_dir=DQN_DIR,
        objective="speculativereactive_hitrate",
        params_filename="best_dqn_speculativereactive_hitrate_params.json",
        payload=dqn_sr,
        label="DQN",
    )


if __name__ == "__main__":
    main()
