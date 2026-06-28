"""Command-line parsing and objective interpretation.

Orchestrator-only flags are declared explicitly. Every other ``--key value`` on
the command line is captured as a pass-through job parameter and forwarded to
``main.py`` unchanged, so the script automatically supports any simulation
argument without code edits.
"""

import argparse

from .config import (
    DEFAULT_ENV_FILE,
    DEFAULT_SEEDS,
    DEFAULT_TRACES,
    MAX_ITR,
    POLL_INTERVAL,
)


def str2bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


def parse_passthrough(unknown):
    """Turn leftover ``--key value`` argv tokens into a dict of strings."""
    params = {}
    i = 0
    while i < len(unknown):
        token = unknown[i]
        if token.startswith("--"):
            key = token[2:]
            if "=" in key:
                k, v = key.split("=", 1)
                params[k] = v
                i += 1
            elif i + 1 < len(unknown) and not unknown[i + 1].startswith("--"):
                params[key] = unknown[i + 1]
                i += 2
            else:
                params[key] = "True"  # bare flag with no value
                i += 1
        else:
            i += 1
    return params


def parse_args(argv=None):
    """Return ``(control_namespace, passthrough_job_params_dict)``."""
    parser = argparse.ArgumentParser(
        description="Ternary-search hyper-parameter tuning over the jd job server.",
        allow_abbrev=False,
    )
    parser.add_argument("--fresh_start", type=str2bool, default=False,
                        help="True: start from scratch. False (default): resume from checkpoint.")
    parser.add_argument("--tuning_seed", type=int, default=0,
                        help="Seed used only to randomise the order tunable params are picked.")
    parser.add_argument("--tunable_params", type=str, required=True,
                        help="Comma-separated list of parameters to tune, e.g. 'agingfactor,bandit_c'.")
    parser.add_argument("--starting_datapoints", type=str, required=True,
                        help="Comma-separated starting values aligned with --tunable_params.")
    parser.add_argument("--lower_bound", type=str, required=True,
                        help="Comma-separated lower bounds aligned with --tunable_params.")
    parser.add_argument("--upper_bound", type=str, required=True,
                        help="Comma-separated upper bounds aligned with --tunable_params.")
    parser.add_argument("--objective", type=str, required=True,
                        help="<mode>_<metric>, e.g. speculative_hitrate or "
                             "speculativereactive_speculation_efficiency. Not forwarded to jobs.")
    parser.add_argument("--current_value", type=float, required=True,
                        help="Objective value of the starting configuration (same scale as the "
                             "aggregated metric: hit-rate is a percentage).")
    parser.add_argument("--base_path", type=str, required=True,
                        help="Local directory for results/checkpoints. Not forwarded to jobs.")
    parser.add_argument("--env_file", type=str, default=DEFAULT_ENV_FILE,
                        help="Path to the jd credentials .env file.")
    parser.add_argument("--max_iter", type=int, default=MAX_ITR,
                        help="Max ternary-search iterations per parameter.")
    parser.add_argument("--poll_interval", type=int, default=POLL_INTERVAL,
                        help="Seconds between job-status polls.")
    parser.add_argument("--traces", type=str, default=",".join(str(t) for t in DEFAULT_TRACES),
                        help="Comma-separated traces to run per midpoint.")
    parser.add_argument("--seeds", type=str, default=",".join(str(s) for s in DEFAULT_SEEDS),
                        help="Comma-separated simulation seeds to run per midpoint.")

    control, unknown = parser.parse_known_args(argv)
    job_params = parse_passthrough(unknown)
    return control, job_params


def parse_objective(objective):
    """Split ``<mode>_<metric>`` into ``(mode, decision_metric)``.

    ``hitrate`` and ``speculation_efficiency`` are the only supported metrics.
    """
    obj = objective.strip()
    if obj.endswith("speculation_efficiency"):
        metric = "speculation_efficiency"
        mode = obj[: -len("_speculation_efficiency")]
    elif obj.endswith("hitrate"):
        metric = "hitrate"
        mode = obj[: -len("_hitrate")]
    else:
        raise ValueError(
            f"Cannot parse objective '{objective}'. Expected it to end with "
            f"'hitrate' or 'speculation_efficiency'."
        )
    if not mode:
        raise ValueError(f"Objective '{objective}' has an empty mode prefix.")
    return mode, metric
