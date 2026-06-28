"""On-disk artefacts: checkpoint, best-objective history and job mapping."""

import csv
import json
import os
import time

from .config import CHECKPOINT_FILE, HISTORY_FILE, JOB_MAPPING_FILE
from .logging_setup import get_logger

logger = get_logger()


def save_checkpoint(run_dir, state):
    """Atomically write the tuning ``state`` dict so a crash can resume cleanly."""
    path = run_dir / CHECKPOINT_FILE
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, path)
    logger.info(f"Checkpoint saved: {path}")


def load_checkpoint(run_dir):
    path = run_dir / CHECKPOINT_FILE
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Failed to read checkpoint {path}: {exc}")
        return None


def _append_csv(path, header, row):
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def record_job_mapping(run_dir, iter_tag, param_name, side, point_value, job_dicts, job_ids):
    """Persist which job ids back a given (parameter, side, midpoint value)."""
    path = run_dir / JOB_MAPPING_FILE
    header = ["timestamp", "iter_tag", "param_name", "side", "point_value",
              "trace", "seed", "job_id"]
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    for idx, job_id in enumerate(job_ids):
        job = job_dicts[idx] if idx < len(job_dicts) else {}
        _append_csv(path, header, {
            "timestamp": ts,
            "iter_tag": iter_tag,
            "param_name": param_name,
            "side": side,
            "point_value": point_value,
            "trace": job.get("trace", ""),
            "seed": job.get("seed", ""),
            "job_id": job_id,
        })
    logger.info(f"Recorded {len(job_ids)} job ids in {path}")


def record_history(run_dir, tunable_order, tunable_state, record):
    """Append one row per iteration: all tunable values + best objective + decision."""
    path = run_dir / HISTORY_FILE
    tunable_cols = [f"value_{name}" for name in tunable_order]
    header = (
        ["timestamp", "global_iteration", "param_name", "param_iter",
         "lower", "center", "upper", "mid_left", "mid_right",
         "obj_center", "obj_mid_left", "obj_mid_right", "decision", "best_objective"]
        + tunable_cols
    )
    row = dict(record)
    row["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    for name in tunable_order:
        row[f"value_{name}"] = tunable_state[name]["value"]
    _append_csv(path, header, row)
    logger.info(f"Appended iteration result to {path}")
