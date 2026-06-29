"""Wiring that turns parsed CLI arguments into a complete tuning run."""

from pathlib import Path

from . import jd_client
from .cli import parse_args, parse_objective, resolve_tunable_spec
from .config import (
    DEFAULT_BASE_PATH,
    ENABLE_JD,
    JOB_PARAM_BLOCKLIST,
    ORCHESTRATOR_PARAM_BLOCKLIST,
    tunable_bound_keys,
)
from .context import RunContext
from .logging_setup import get_logger, setup_logging
from .persistence import load_checkpoint, save_checkpoint
from .search import build_initial_state, run_tuning

logger = get_logger()


def _resolve_base_path(raw_job_params):
    """Resolve and create the output root, matching ``code/main.py`` behaviour."""
    if ENABLE_JD:
        output_dir = jd_client.job_dir()
    else:
        base = raw_job_params.get("base_path") or DEFAULT_BASE_PATH
        output_dir = Path(base)
    output_dir.mkdir(parents=True, exist_ok=True)
    return str(output_dir)


def _build_run_dir(base_path, algorithm, objective, tuning_seed):
    root = Path(base_path)
    if ENABLE_JD:
        return root
    return (
        root
        / f"algorithm_{algorithm}"
        / f"objective_{objective}"
        / f"tunable_seed_{tuning_seed}"
    )


def _static_job_params(raw_job_params, tunable):
    """Pass-through params minus orchestrator/tuning keys and tunable param columns."""
    blocked = (
        JOB_PARAM_BLOCKLIST
        | ORCHESTRATOR_PARAM_BLOCKLIST
        | tunable_bound_keys()
        | set(tunable)
    )
    return {k: v for k, v in raw_job_params.items() if k not in blocked}


def _log_configuration(control, ctx, mode, metric, tunable, sdp, lb, ub, base_path):
    logger.info("Ternary-search tuning starting")
    logger.info(f"  objective         : {control.objective} -> mode={mode}, metric={metric}")
    logger.info(f"  tunable_params    : {tunable}")
    logger.info(f"  starting_datapts  : {sdp}")
    logger.info(f"  lower_bound       : {lb}")
    logger.info(f"  upper_bound       : {ub}")
    logger.info(f"  current_value     : {control.current_value}")
    logger.info(f"  tuning_seed       : {control.tuning_seed}")
    logger.info(f"  base_path         : {base_path}")
    logger.info(f"  traces            : {ctx.traces}")
    logger.info(f"  seeds             : {ctx.seeds}")
    logger.info(f"  jobs per midpoint : {ctx.jobs_per_midpoint}")
    logger.info(f"  max_iter          : {ctx.max_iter}")
    logger.info(f"  poll_interval     : {ctx.poll_interval}s")
    logger.info(f"  fresh_start       : {control.fresh_start}")
    logger.info(f"  run_dir           : {ctx.run_dir}")
    logger.info(f"  env_file          : {ctx.env_file}")
    logger.info(f"  simulation exp    : ternary-search")
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
    tunable, sdp, lb, ub = resolve_tunable_spec(control, raw_job_params)
    base_path = _resolve_base_path(raw_job_params)

    traces = [int(t) for t in control.traces.split(",") if t.strip()]
    seeds = [int(s) for s in control.seeds.split(",") if s.strip()]
    job_params = _static_job_params(raw_job_params, tunable)
    job_params["base_path"] = base_path
    algorithm = job_params.get("algorithm", "unknown")

    run_dir = _build_run_dir(base_path, algorithm, control.objective, control.tuning_seed)
    run_dir.mkdir(parents=True, exist_ok=True)

    ctx = RunContext(
        run_dir=run_dir,
        job_params=job_params,
        mode=mode,
        metric=metric,
        max_iter=control.max_iter,
        poll_interval=control.poll_interval,
        tunable_order=tunable,
        traces=traces,
        seeds=seeds,
        env_file=control.env_file,
    )

    setup_logging(ctx.run_dir)
    _log_configuration(control, ctx, mode, metric, tunable, sdp, lb, ub, base_path)

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
