"""Submit ternary-search tuning jobs to JobDistributor from a CSV file."""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

from . import jd_client
from .config import DEFAULT_ENV_FILE, LOGGER_NAME, ORCHESTRATOR_EXP_ID
from .logging_setup import get_logger

logger = get_logger()


def _setup_console_logging():
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s")
    log = logging.getLogger(LOGGER_NAME)
    log.setLevel(logging.INFO)
    log.handlers.clear()
    log.propagate = False
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(fmt)
    log.addHandler(handler)

SKIP_COLUMNS = {"run_id"}
REQUIRED_COLUMNS = {"objective", "algorithm", "ordering", "tuning_seed", "current_value"}


def _is_empty(value) -> bool:
    return value is None or str(value).strip() == ""


def row_to_job(row: dict) -> dict:
    """Convert one ``commands.csv`` row to a jd job parameter dict."""
    job = {"fresh_start": "False"}
    for key, value in row.items():
        if key in SKIP_COLUMNS or _is_empty(value):
            continue
        job[key] = str(value).strip()
    return job


def load_jobs(csv_path: Path, run_ids: set[int] | None = None) -> list[dict]:
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"No rows found in {csv_path}")

    missing = REQUIRED_COLUMNS - set(rows[0].keys())
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {sorted(missing)}")

    jobs = []
    for row in rows:
        if run_ids is not None:
            rid = int(row["run_id"])
            if rid not in run_ids:
                continue
        jobs.append(row_to_job(row))
    if not jobs:
        raise ValueError("No jobs selected (check --run_id filters)")
    return jobs


def submit_jobs(csv_path: Path, env_file: str, run_ids: set[int] | None = None, dry_run: bool = False):
    jobs = load_jobs(csv_path, run_ids)
    logger.info(f"Loaded {len(jobs)} tuning job(s) from {csv_path}")

    if dry_run:
        for i, job in enumerate(jobs, 1):
            logger.info(f"  [{i}] {job}")
        return []

    job_ids = jd_client.create_orchestrator_jobs(jobs, env_file)
    logger.info(f"Submitted {len(job_ids)} job(s) to experiment '{ORCHESTRATOR_EXP_ID}'")
    for job_id, job in zip(job_ids, jobs):
        logger.info(
            f"  id={job_id} objective={job.get('objective')} "
            f"algorithm={job.get('algorithm')} ordering={job.get('ordering')} "
            f"tuning_seed={job.get('tuning_seed')}"
        )
    return job_ids


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Submit ternary-search tuning jobs from commands.csv to JobDistributor.",
    )
    default_csv = Path(__file__).resolve().parents[1] / "commands.csv"
    parser.add_argument(
        "--csv",
        type=Path,
        default=default_csv,
        help=f"CSV file with one tuning job per row (default: {default_csv.name}).",
    )
    parser.add_argument("--env_file", type=str, default=DEFAULT_ENV_FILE)
    parser.add_argument(
        "--run_id",
        type=int,
        nargs="+",
        help="Submit only these run_id values from the CSV.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print job dicts without calling jd.create_jobs.",
    )
    args = parser.parse_args(argv)

    _setup_console_logging()
    run_ids = set(args.run_id) if args.run_id else None
    submit_jobs(args.csv.resolve(), args.env_file, run_ids=run_ids, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
