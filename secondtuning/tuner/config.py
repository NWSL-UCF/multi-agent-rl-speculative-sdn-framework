"""Constants and tunable knobs for the ternary-search orchestrator."""

# Name of the shared logger used across every module.
LOGGER_NAME = "ternary_search"

# Traces and seeds that make up the jobs evaluated per midpoint (3 x 3 = 9).
DEFAULT_TRACES = [1, 2, 3]
DEFAULT_SEEDS = [101, 297, 413]

# Maximum ternary-search iterations per parameter.
MAX_ITR = 7

# When True, resolve output paths via ``jd.jd_job_dir()`` (same as code/main.py).
ENABLE_JD = True

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
DEFAULT_ENV_FILE = "/jet/home/arouf/data/ternary-search.env"

# JobDistributor experiment ids.
ORCHESTRATOR_EXP_ID = "second-tuning"
SIMULATION_EXP_ID = "ternary-search"

# Same default as code/main.py --base_path.
DEFAULT_BASE_PATH = (
    "/home/ab823254/data/multi-agent-rl-speculative-sdn-framework/results/debug"
)

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
JOB_PARAM_BLOCKLIST = {"seed", "trace", "mode", "objective", "fresh_start"}

# Params that may be tuned (one column each in commands.csv / JobDistributor).
TUNABLE_PARAM_NAMES = (
    "bandit_c",
    "gamma",
    "rewardAgingFactor",
    "spatialReward",
    "agingfactor",
    "dqn_lr",
    "ppo_lr",
)

# Per-algorithm tunable params in CSV column order (unused params are omitted per row).
ALGO_TUNABLE_PARAMS = {
    "bandit": ("bandit_c", "rewardAgingFactor", "spatialReward", "agingfactor"),
    "dqn": ("gamma", "rewardAgingFactor", "spatialReward", "agingfactor", "dqn_lr"),
    "ppo": ("gamma", "rewardAgingFactor", "spatialReward", "agingfactor", "ppo_lr"),
}

# CSV / orchestrator keys that must not be forwarded to simulation jobs.
ORCHESTRATOR_PARAM_BLOCKLIST = {"run_id"}


def tunable_bound_keys():
    """Return ``{param}_lower`` / ``{param}_upper`` keys for every tunable param."""
    keys = set()
    for name in TUNABLE_PARAM_NAMES:
        keys.add(f"{name}_lower")
        keys.add(f"{name}_upper")
    return keys

# On-disk artefact filenames (relative to the run directory).
CHECKPOINT_FILE = "checkpoint.json"
HISTORY_FILE = "best_objective_history.csv"
JOB_MAPPING_FILE = "job_mapping.csv"
LOG_FILE = "tuning_log.txt"
JOBS_DIRNAME = "jobs"
AGGREGATED_DIRNAME = "aggregated"
