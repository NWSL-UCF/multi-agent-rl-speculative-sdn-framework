"""Build, submit and score the jobs for a single midpoint value.

A "point" is one candidate value of the parameter being tuned. Evaluating it
means running the simulation across every (trace, seed) combination, downloading
the results and aggregating them into a single objective number.
"""

from . import jd_client
from .aggregation import aggregate_point
from .logging_setup import get_logger
from .persistence import record_job_mapping

logger = get_logger()


def _stringify(value):
    """jd job parameters are strings; preserve float precision with repr."""
    return repr(value) if isinstance(value, float) else str(value)


def build_point_jobs(ctx, tunable_state, param_name, point_value):
    """Build the per-(trace, seed) job parameter dicts for one midpoint value."""
    shared = dict(ctx.job_params)
    shared["mode"] = ctx.mode
    # All tunable params sit at their current best value...
    for name, info in tunable_state.items():
        shared[name] = info["value"]
    # ...except the one being probed, which uses the midpoint value.
    shared[param_name] = point_value

    jobs = []
    for trace in ctx.traces:
        for seed in ctx.seeds:
            job = dict(shared)
            job["trace"] = trace
            job["seed"] = seed
            jobs.append({k: _stringify(v) for k, v in job.items()})
    return jobs


def submit_point(ctx, tunable_state, param_name, point_value, side, iter_tag):
    """Create the jobs for one midpoint and return their ids (does NOT wait).

    Both midpoints are submitted before waiting so the workers run them in parallel.
    """
    logger.info("-" * 70)
    logger.info(f"Submitting {param_name}={point_value} [{side}] ({iter_tag})")

    job_dicts = build_point_jobs(ctx, tunable_state, param_name, point_value)
    job_ids = jd_client.create_jobs(job_dicts)
    record_job_mapping(ctx.run_dir, iter_tag, param_name, side, point_value, job_dicts, job_ids)
    return job_ids


def score_point(ctx, job_ids, point_value, side, iter_tag):
    """Aggregate already-downloaded results for one midpoint and return its objective.

    Returns ``(objective_value, avg_hitrate, avg_speculation_efficiency)`` where
    ``objective_value`` is the metric selected by the objective.
    """
    agg_csv = ctx.agg_dir / f"{iter_tag}_{side}.csv"
    avg_hitrate, avg_spec_eff = aggregate_point(job_ids, ctx.jobs_dir, agg_csv)

    metric_value = avg_hitrate if ctx.metric == "hitrate" else avg_spec_eff
    logger.info(f"{side} {point_value} objective ({ctx.metric}) = {metric_value:.6f}")
    return metric_value, avg_hitrate, avg_spec_eff
