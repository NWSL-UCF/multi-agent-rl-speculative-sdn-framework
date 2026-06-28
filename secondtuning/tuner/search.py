"""The ternary-search algorithm and its persistent state.

State shape (a plain dict so it serialises straight to checkpoint.json)::

    {
      "objective": str, "mode": str, "metric": str, "tuning_seed": int,
      "current_objective": float,        # best objective of the current config
      "param_order": [name, ...],        # randomised tuning order
      "order_index": int,                # which param we are on
      "param_iter": int,                 # iteration within the current param
      "global_iteration": int,
      "params": {
          name: {"value", "lower", "upper", "orig_lower", "orig_upper", "tuned"}
      },
    }

``current_objective`` is always the objective at the current centre, because every
other parameter is fixed at its best value -- so the centre never needs re-running.
"""

import random

from . import jd_client
from .config import CONVERGENCE_EPS, MIDPOINT_PRECISION
from .jobs import score_point, submit_point
from .logging_setup import get_logger
from .persistence import record_history, save_checkpoint

logger = get_logger()


def build_initial_state(control, mode, metric, tunable, sdp, lb, ub):
    """Create a fresh state dict from the parsed CLI inputs."""
    rng = random.Random(control.tuning_seed)
    param_order = tunable[:]
    rng.shuffle(param_order)

    params = {}
    for name, start, low, high in zip(tunable, sdp, lb, ub):
        center = min(max(start, low), high)  # clamp the starting datapoint into bounds
        params[name] = {
            "value": center,
            "lower": low,
            "upper": high,
            "orig_lower": low,
            "orig_upper": high,
            "tuned": False,
        }

    return {
        "objective": control.objective,
        "mode": mode,
        "metric": metric,
        "tuning_seed": control.tuning_seed,
        "current_objective": control.current_value,
        "param_order": param_order,
        "order_index": 0,
        "param_iter": 0,
        "global_iteration": 0,
        "params": params,
    }


def _apply_decision(info, center, center_obj, mid_left, mid_right, ml_obj, mr_obj):
    """Shrink the window based on which point scored best (higher is better).

    Returns ``(decision_label, new_center_objective)`` and mutates ``info`` in place.
    """
    if center_obj >= ml_obj and center_obj >= mr_obj:
        # Centre wins: narrow the window around it, keep centre/objective.
        info["lower"] = mid_left
        info["upper"] = mid_right
        return "center", center_obj
    if ml_obj >= mr_obj:
        # Left midpoint wins: optimum is below the centre.
        info["upper"] = center
        info["value"] = mid_left
        return "mid_left", ml_obj
    # Right midpoint wins: optimum is above the centre.
    info["lower"] = center
    info["value"] = mid_right
    return "mid_right", mr_obj


def _evaluate_midpoints(ctx, tunable_state, param_name, mid_left, mid_right, iter_tag):
    """Submit both midpoints in parallel, wait once, then score each.

    Returns ``(mid_left_objective, mid_right_objective)``.
    """
    ml_jobs = submit_point(ctx, tunable_state, param_name, mid_left, "mid_left", iter_tag)
    mr_jobs = submit_point(ctx, tunable_state, param_name, mid_right, "mid_right", iter_tag)

    jd_client.wait_and_download(ml_jobs + mr_jobs, ctx.jobs_dir, ctx.poll_interval)

    ml_obj, _, _ = score_point(ctx, ml_jobs, mid_left, "mid_left", iter_tag)
    mr_obj, _, _ = score_point(ctx, mr_jobs, mid_right, "mid_right", iter_tag)
    return ml_obj, mr_obj


def _run_iteration(ctx, state, param_name):
    """Run one ternary-search iteration for ``param_name``.

    Returns False if the window has converged (search should stop), else True.
    """
    info = state["params"][param_name]
    lower, upper, center = info["lower"], info["upper"], info["value"]
    center_obj = state["current_objective"]
    it = state["param_iter"]

    if (upper - lower) < CONVERGENCE_EPS:
        logger.info(
            f"Window ({lower}, {upper}) below convergence eps {CONVERGENCE_EPS}; "
            f"stopping search for '{param_name}'."
        )
        return False

    mid_left = round((lower + center) / 2.0, MIDPOINT_PRECISION)
    mid_right = round((center + upper) / 2.0, MIDPOINT_PRECISION)
    iter_tag = f"{param_name}_iter{it}"

    logger.info("#" * 70)
    logger.info(
        f"[{param_name}] iter {it}: lower={lower:.6f} center={center:.6f} upper={upper:.6f} | "
        f"mid_left={mid_left:.6f} mid_right={mid_right:.6f} | obj(center)={center_obj:.6f}"
    )

    ml_obj, mr_obj = _evaluate_midpoints(ctx, state["params"], param_name, mid_left, mid_right, iter_tag)
    decision, state["current_objective"] = _apply_decision(
        info, center, center_obj, mid_left, mid_right, ml_obj, mr_obj
    )

    logger.info(
        f"[{param_name}] iter {it} decision={decision} -> "
        f"new window=({info['lower']:.6f}, {info['upper']:.6f}) center={info['value']:.6f} "
        f"best_objective={state['current_objective']:.6f}"
    )

    state["param_iter"] = it + 1
    record_history(ctx.run_dir, ctx.tunable_order, state["params"], {
        "global_iteration": state["global_iteration"],
        "param_name": param_name,
        "param_iter": it,
        "lower": lower, "center": center, "upper": upper,
        "mid_left": mid_left, "mid_right": mid_right,
        "obj_center": center_obj, "obj_mid_left": ml_obj, "obj_mid_right": mr_obj,
        "decision": decision,
        "best_objective": state["current_objective"],
    })
    state["global_iteration"] += 1
    save_checkpoint(ctx.run_dir, state)
    return True


def _tune_parameter(ctx, state, param_name):
    """Run the ternary search for a single parameter up to ``max_iter`` iterations."""
    info = state["params"][param_name]
    logger.info("=" * 80)
    logger.info(
        f"Tuning '{param_name}' | window=({info['lower']}, {info['upper']}) | "
        f"center={info['value']} | current_objective={state['current_objective']:.6f} | "
        f"resuming at param_iter={state['param_iter']}"
    )

    while state["param_iter"] < ctx.max_iter:
        if not _run_iteration(ctx, state, param_name):
            break

    info["tuned"] = True
    logger.info(
        f"Finished '{param_name}': best value={info['value']:.6f}, "
        f"best_objective={state['current_objective']:.6f}"
    )


def run_tuning(ctx, state):
    """Tune every parameter in order, resuming wherever the checkpoint left off."""
    param_order = state["param_order"]

    while state["order_index"] < len(param_order):
        param_name = param_order[state["order_index"]]

        if not state["params"][param_name]["tuned"]:
            _tune_parameter(ctx, state, param_name)

        state["order_index"] += 1
        state["param_iter"] = 0
        save_checkpoint(ctx.run_dir, state)

    logger.info("=" * 80)
    logger.info("All parameters tuned. Final values:")
    for name in param_order:
        logger.info(f"  {name} = {state['params'][name]['value']:.6f}")
    logger.info(f"Final best objective ({ctx.metric}): {state['current_objective']:.6f}")
