"""Every interaction with the JobDistributor (jd) job server.

All calls are wrapped with retries so transient network/server errors do not
abort a long tuning run. This is the only module that imports ``jd``.
"""

import random
import time
from pathlib import Path

from .config import (
    DOWNLOAD_MAX_RETRIES,
    LIST_MAX_RETRIES,
    NEW_ID_POLL_ATTEMPTS,
    RESULT_FILES,
    RETRY_SLEEP_MIN,
    RETRY_SLEEP_MAX,
    TERMINAL_BAD_STATUSES,
)
from .logging_setup import get_logger

try:
    import jd
except ImportError:  # pragma: no cover - jd is only available where workers run
    jd = None

logger = get_logger()


def _retry_sleep(reason, attempt):
    delay = random.uniform(RETRY_SLEEP_MIN, RETRY_SLEEP_MAX)
    logger.warning(f"{reason} (attempt {attempt}); retrying in {delay:.1f}s")
    time.sleep(delay)


def init(env_file):
    """Initialise jd from the credentials .env file."""
    if jd is None:
        raise RuntimeError(
            "The 'jd' package is not installed in this environment. Install jd-worker "
            "(see requirements.txt) before running the orchestrator."
        )
    env_path = Path(env_file).expanduser()
    if not env_path.exists():
        raise FileNotFoundError(f"jd env file not found: {env_path}")
    logger.info(f"Initialising jd from {env_path}")
    jd.init(str(env_path))
    logger.info(f"jd initialised. exp_path={jd.exp_path()}")


def list_all_jobs():
    """Return every job dict for the experiment, with retries. May return []."""
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


def create_jobs(param_dicts):
    """Create a batch of jobs and return their (ascending) job ids.

    Retries the create call until it returns without raising, then polls
    ``list_jobs`` for the new ids (the API does not return them directly).
    """
    expected = len(param_dicts)
    before = max_job_id()
    logger.info(f"Creating {expected} jobs (max existing id={before})")

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


def download(job_id, filename, dest_path):
    """Download one result file for one job, with retries. Returns True on success."""
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


def wait_and_download(job_ids, jobs_dir, poll_interval):
    """Poll until every job is DONE, downloading result files as each one finishes.

    Only ``DONE`` is terminal. Statuses such as ``ABORTED`` are NOT given up on:
    the jd server reassigns those jobs, so we keep polling until they eventually
    reach ``DONE``. Download failures (DONE but files missing) are the only way a
    job lands in ``failed``.

    Returns the set of job ids whose results were fully downloaded.
    """
    pending = set(job_ids)
    done = set()
    failed = set()

    logger.info(f"Waiting for {len(pending)} jobs: {sorted(pending)}")
    while pending:
        status = {int(j["id"]): str(j.get("status", "")).upper() for j in list_all_jobs()}

        aborted = {}
        for job_id in sorted(pending):
            st = status.get(job_id, "UNKNOWN")
            if st == "DONE":
                dest_dir = jobs_dir / str(job_id)
                ok = all(download(job_id, fname, dest_dir / fname) for fname in RESULT_FILES)
                pending.discard(job_id)
                (done if ok else failed).add(job_id)
                if not ok:
                    logger.error(f"Job {job_id} DONE but result download incomplete.")
            elif st in TERMINAL_BAD_STATUSES:
                # The server will reassign it; keep waiting for it to become DONE.
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
