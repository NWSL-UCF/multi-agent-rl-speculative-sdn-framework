"""Constants and tunable knobs for the ternary-search orchestrator."""

# Name of the shared logger used across every module.
LOGGER_NAME = "ternary_search"

# Traces and seeds that make up the jobs evaluated per midpoint (3 x 3 = 9).
DEFAULT_TRACES = [1, 2, 3]
DEFAULT_SEEDS = [101, 297, 413]

# Maximum ternary-search iterations per parameter.
MAX_ITR = 7

# Stop searching a parameter once its window is smaller than this.
CONVERGENCE_EPS = 1e-4

# Midpoints are rounded to this many decimal places before they become jobs.
MIDPOINT_PRECISION = 4

# How often (seconds) to poll job status while waiting for results.
POLL_INTERVAL = 120

# Retry knobs (seconds / attempt counts).
RETRY_SLEEP_MIN = 5
RETRY_SLEEP_MAX = 30
LIST_MAX_RETRIES = 10
DOWNLOAD_MAX_RETRIES = 8
NEW_ID_POLL_ATTEMPTS = 6

# Default location of the jd credentials .env file (kept OUTSIDE the repo).
DEFAULT_ENV_FILE = "/home/ab823254/data/ternary-search.env"

# Files downloaded for every finished job.
RESULT_FILES = ("lti_metrics.csv", "summary.json")

# jd job statuses we treat as "finished but produced no usable result".
TERMINAL_BAD_STATUSES = {"ABORTED", "FAILED", "ERROR", "CANCELLED", "CANCELED"}

# Raw lti_metrics columns summed per second-bucket during aggregation.
# ``total_flows`` is intentionally excluded: it is derived as
# ``reactive_flows + speculative_flows`` when computing per-bucket metrics.
SUM_COLS = [
    "total_packets",
    "total_hits",
    "reactive_hits",
    "speculative_hits",
    "reactive_flows",
    "speculative_flows",
]

# Pass-through keys that must never reach the job (we control these ourselves).
JOB_PARAM_BLOCKLIST = {"seed", "trace", "mode", "objective", "base_path", "fresh_start"}

# On-disk artefact filenames (relative to the run directory).
CHECKPOINT_FILE = "checkpoint.json"
HISTORY_FILE = "best_objective_history.csv"
JOB_MAPPING_FILE = "job_mapping.csv"
LOG_FILE = "tuning_log.txt"
JOBS_DIRNAME = "jobs"
AGGREGATED_DIRNAME = "aggregated"
