"""Wiring that turns parsed CLI arguments into a complete tuning run."""

from pathlib import Path

from . import jd_client
from .cli import parse_args, parse_objective
from .config import JOB_PARAM_BLOCKLIST
from .context import RunContext
from .logging_setup import get_logger, setup_logging
from .persistence import load_checkpoint, save_checkpoint
from .search import build_initial_state, run_tuning

logger = get_logger()


def _parse_aligned_lists(control):
    """Parse and validate the comma-separated tunable/bounds inputs."""
    tunable = [p.strip() for p in control.tunable_params.split(",") if p.strip()]
    sdp = [float(x) for x in control.starting_datapoints.split(",")]
    lb = [float(x) for x in control.lower_bound.split(",")]
    ub = [float(x) for x in control.upper_bound.split(",")]

    if not (len(tunable) == len(sdp) == len(lb) == len(ub)):
        raise ValueError(
            "tunable_params, starting_datapoints, lower_bound and upper_bound must all "
            f"have the same length (got {len(tunable)}, {len(sdp)}, {len(lb)}, {len(ub)})."
        )
    return tunable, sdp, lb, ub


def _build_run_dir(base_path, algorithm, objective, tuning_seed):
    return (
        Path(base_path)
        / f"algorithm_{algorithm}"
        / f"objective_{objective}"
        / f"tunable_seed_{tuning_seed}"
    )


def _static_job_params(raw_job_params, tunable):
    """Pass-through params minus everything we control and minus the tunable params."""
    blocked = JOB_PARAM_BLOCKLIST | set(tunable)
    return {k: v for k, v in raw_job_params.items() if k not in blocked}


def _log_configuration(control, ctx, mode, metric, tunable, sdp, lb, ub):
    logger.info("Ternary-search tuning starting")
    logger.info(f"  objective         : {control.objective} -> mode={mode}, metric={metric}")
    logger.info(f"  tunable_params    : {tunable}")
    logger.info(f"  starting_datapts  : {sdp}")
    logger.info(f"  lower_bound       : {lb}")
    logger.info(f"  upper_bound       : {ub}")
    logger.info(f"  current_value     : {control.current_value}")
    logger.info(f"  tuning_seed       : {control.tuning_seed}")
    logger.info(f"  traces            : {ctx.traces}")
    logger.info(f"  seeds             : {ctx.seeds}")
    logger.info(f"  jobs per midpoint : {ctx.jobs_per_midpoint}")
    logger.info(f"  max_iter          : {ctx.max_iter}")
    logger.info(f"  poll_interval     : {ctx.poll_interval}s")
    logger.info(f"  fresh_start       : {control.fresh_start}")
    logger.info(f"  run_dir           : {ctx.run_dir}")
    logger.info(f"  static job params : {ctx.job_params}")


def _resolve_state(control, ctx, mode, metric, tunable, sdp, lb, ub):
    """Load the checkpoint (unless fresh_start) or build a new state."""
    if not control.fresh_start:
        state = load_checkpoint(ctx.run_dir)
        if state is not None:
            logger.info(
                f"Resuming from checkpoint: order_index={state['order_index']}, "
                f"param_iter={state['param_iter']}, current_objective={state['current_objective']}"
            )
            return state
        logger.info("No checkpoint found; starting fresh.")

    state = build_initial_state(control, mode, metric, tunable, sdp, lb, ub)
    save_checkpoint(ctx.run_dir, state)
    logger.info("Initialised fresh tuning state.")
    return state


def main(argv=None):
    control, raw_job_params = parse_args(argv)
    mode, metric = parse_objective(control.objective)
    tunable, sdp, lb, ub = _parse_aligned_lists(control)

    traces = [int(t) for t in control.traces.split(",") if t.strip()]
    seeds = [int(s) for s in control.seeds.split(",") if s.strip()]
    job_params = _static_job_params(raw_job_params, tunable)
    algorithm = job_params.get("algorithm", "unknown")

    ctx = RunContext(
        run_dir=_build_run_dir(control.base_path, algorithm, control.objective, control.tuning_seed),
        job_params=job_params,
        mode=mode,
        metric=metric,
        max_iter=control.max_iter,
        poll_interval=control.poll_interval,
        tunable_order=tunable,
        traces=traces,
        seeds=seeds,
    )

    setup_logging(ctx.run_dir)
    _log_configuration(control, ctx, mode, metric, tunable, sdp, lb, ub)

    jd_client.init(control.env_file)
    state = _resolve_state(control, ctx, mode, metric, tunable, sdp, lb, ub)

    try:
        run_tuning(ctx, state)
        logger.info("Tuning completed successfully.")
    except KeyboardInterrupt:
        logger.warning("Interrupted by user; checkpoint already saved per iteration.")
        save_checkpoint(ctx.run_dir, state)
        raise
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Tuning failed: {exc}", exc_info=True)
        save_checkpoint(ctx.run_dir, state)
        raise
