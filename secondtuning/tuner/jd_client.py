"""Every interaction with the JobDistributor (jd) job server.

Two experiments are used:

- **second-tuning** — one orchestrator job per ``commands.csv`` row (``start_tuning.py``).
- **ternary-search** — simulation jobs (9 per midpoint × 2 midpoints = 18 per iteration).
"""

from __future__ import annotations

import random
import time
from pathlib import Path

from .config import (
    DOWNLOAD_MAX_RETRIES,
    LIST_MAX_RETRIES,
    NEW_ID_POLL_ATTEMPTS,
    ORCHESTRATOR_EXP_ID,
    RESULT_FILES,
    RETRY_SLEEP_MIN,
    RETRY_SLEEP_MAX,
    SIMULATION_EXP_ID,
    TERMINAL_BAD_STATUSES,
)
from .logging_setup import get_logger

try:
    import jd
except ImportError:  # pragma: no cover - jd is only available where workers run
    jd = None

logger = get_logger()

_active_exp_id: str | None = None


def _retry_sleep(reason, attempt):
    delay = random.uniform(RETRY_SLEEP_MIN, RETRY_SLEEP_MAX)
    logger.warning(f"{reason} (attempt {attempt}); retrying in {delay:.1f}s")
    time.sleep(delay)


def _require_jd():
    if jd is None:
        raise RuntimeError(
            "The 'jd' package is not installed in this environment. Install jd-worker "
            "(see requirements.txt) before using JobDistributor."
        )


def init_experiment(env_file: str, exp_id: str):
    """Connect the job-management client to a specific experiment."""
    global _active_exp_id
    _require_jd()
    env_path = Path(env_file).expanduser()
    if not env_path.exists():
        raise FileNotFoundError(f"jd env file not found: {env_path}")
    logger.info(f"Initialising jd for experiment '{exp_id}' from {env_path}")
    jd.init(str(env_path), exp_id=exp_id)
    _active_exp_id = exp_id
    logger.info(f"jd ready. exp_path={jd.exp_path()}")


def init_orchestrator(env_file: str):
    """Connect to the second-tuning queue (orchestrator jobs)."""
    init_experiment(env_file, ORCHESTRATOR_EXP_ID)


def init_simulation(env_file: str):
    """Connect to the ternary-search queue (simulation midpoint jobs)."""
    init_experiment(env_file, SIMULATION_EXP_ID)


def ensure_simulation(env_file: str):
    if _active_exp_id != SIMULATION_EXP_ID:
        init_simulation(env_file)


def job_dir():
    """Return the current worker job directory (second-tuning job on workers)."""
    _require_jd()
    return Path(jd.jd_job_dir())


def list_all_jobs():
    """Return every job dict for the active experiment, with retries."""
    _require_jd()
    for attempt in range(1, LIST_MAX_RETRIES + 1):
        try:
            page = jd.list_jobs(fetch_all=True)
            return page.get("jobs") or []
        except Exception as exc:  # noqa: BLE001 - network errors are expected
            _retry_sleep(f"list_jobs failed: {exc}", attempt)
    logger.error("list_jobs failed after maximum retries; returning empty list.")
    return []


def max_job_id():
    return max((int(j["id"]) for j in list_all_jobs()), default=0)


def new_ids_since(since_id):
    jobs = list_all_jobs()
    return sorted(int(j["id"]) for j in jobs if int(j["id"]) > since_id)


def _create_jobs_impl(param_dicts):
    expected = len(param_dicts)
    before = max_job_id()
    logger.info(
        f"Creating {expected} job(s) on experiment '{_active_exp_id}' "
        f"(max existing id={before})"
    )

    attempt = 0
    while True:
        attempt += 1
        try:
            result = jd.create_jobs(param_dicts)
            logger.info(f"create_jobs response: {result}")
            break
        except Exception as exc:  # noqa: BLE001
            _retry_sleep(f"create_jobs failed: {exc}", attempt)

    new_ids = []
    for poll in range(NEW_ID_POLL_ATTEMPTS):
        new_ids = new_ids_since(before)
        if len(new_ids) >= expected:
            chosen = new_ids[-expected:]
            logger.info(f"Created job ids: {chosen}")
            return chosen
        logger.info(f"Waiting for created ids to appear ({len(new_ids)}/{expected}); poll {poll + 1}")
        time.sleep(RETRY_SLEEP_MIN)

    logger.warning(
        f"Only {len(new_ids)} of {expected} new job ids appeared: {new_ids}. "
        f"Proceeding with what is available."
    )
    return new_ids


def create_orchestrator_jobs(param_dicts, env_file: str):
    """Upload orchestrator jobs to the second-tuning experiment."""
    init_orchestrator(env_file)
    return _create_jobs_impl(param_dicts)


def create_simulation_jobs(param_dicts, env_file: str):
    """Upload simulation jobs to the ternary-search experiment."""
    ensure_simulation(env_file)
    return _create_jobs_impl(param_dicts)


def download(job_id, filename, dest_path, env_file: str):
    """Download one result file for one simulation job."""
    ensure_simulation(env_file)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, DOWNLOAD_MAX_RETRIES + 1):
        try:
            jd.download_result(job_id, filename, dest=str(dest_path))
            if dest_path.exists():
                logger.info(f"Downloaded {filename} for job {job_id} -> {dest_path}")
                return True
            logger.warning(f"download_result returned but {dest_path} missing")
        except Exception as exc:  # noqa: BLE001
            _retry_sleep(f"download {filename} for job {job_id} failed: {exc}", attempt)
    logger.error(f"Giving up downloading {filename} for job {job_id} after retries.")
    return False


def wait_and_download(job_ids, jobs_dir, poll_interval, env_file: str):
    """Poll ternary-search until every simulation job is DONE and download results."""
    ensure_simulation(env_file)
    pending = set(job_ids)
    done = set()
    failed = set()

    logger.info(f"Waiting for {len(pending)} simulation job(s): {sorted(pending)}")
    while pending:
        status = {int(j["id"]): str(j.get("status", "")).upper() for j in list_all_jobs()}

        aborted = {}
        for job_id in sorted(pending):
            st = status.get(job_id, "UNKNOWN")
            if st == "DONE":
                dest_dir = jobs_dir / str(job_id)
                ok = all(
                    download(job_id, fname, dest_dir / fname, env_file)
                    for fname in RESULT_FILES
                )
                pending.discard(job_id)
                (done if ok else failed).add(job_id)
                if not ok:
                    logger.error(f"Job {job_id} DONE but result download incomplete.")
            elif st in TERMINAL_BAD_STATUSES:
                aborted[job_id] = st

        if aborted:
            logger.warning(
                f"{len(aborted)} jobs are in a bad state but await server reassignment: {aborted}"
            )

        logger.info(
            f"Poll: done={len(done)} failed={len(failed)} pending={len(pending)} ({sorted(pending)})"
        )
        if pending:
            time.sleep(poll_interval)

    if failed:
        logger.warning(f"{len(failed)} jobs finished DONE but could not be downloaded: {sorted(failed)}")
    return done
