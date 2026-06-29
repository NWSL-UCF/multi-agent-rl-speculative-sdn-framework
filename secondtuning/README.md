# Ternary-search tuning (jd job server)

Tunes simulation hyper-parameters one at a time with a ternary search. For each
parameter it evaluates a left/right midpoint, runs every `(trace, seed)` combo on
the [JobDistributor](https://hub.jobdistributor.net) workers, aggregates the
downloaded `lti_metrics.csv` into per-second buckets, and shrinks the window
toward the best objective. The orchestrator only talks to the jd API — it never
runs the simulation locally.

## Setup

1. Install deps (repo root): `pip install -r requirements.txt` (needs `jd-worker`).
2. Credentials live in an `.env` **outside the repo**, default
   `/home/ab823254/data/ternary-search.env`:

   ```
   JD_API_KEY=jd_...
   JD_EXP_ID=ternary-search
   ```

   Override the path with `--env_file`.
3. The jd experiment must exist and have workers running.

## Run

### JobDistributor / CSV columns (recommended)

Map each column from `commands.csv` to a CLI flag (`run_id` is ignored). Tunable
params use three columns each: `param`, `param_lower`, `param_upper`.

```bash
cd secondtuning
python start_tuning.py \
    --fresh_start False \
    --tuning_seed 101 \
    --objective speculative_hitrate \
    --current_value 50.28 \
    --ordering trace \
    --algorithm bandit \
    --tablesize 50 \
    --bandit_c 5.0 --bandit_c_lower 0.001 --bandit_c_upper 100.0 \
    --rewardAgingFactor 0.9 --rewardAgingFactor_lower 0.7 --rewardAgingFactor_upper 0.999 \
    --spatialReward 0.75 --spatialReward_lower 0.7 --spatialReward_upper 0.999 \
    --agingfactor 0.997 --agingfactor_lower 0.7 --agingfactor_upper 0.999 \
    --tablesize 50
```

When launched via JobDistributor, output is written to ``jd.jd_job_dir()`` automatically
(same as ``code/main.py``). Pass ``--base_path`` only for local runs with
``ENABLE_JD=False`` in ``tuner/config.py``.

### Legacy comma-separated bounds

```bash
cd secondtuning
python start_tuning.py \
    --fresh_start False \
    --tuning_seed 345 \
    --tunable_params "bandit_c,rewardAgingFactor,spatialReward,agingfactor" \
    --starting_datapoints "5.0,0.9,0.75,0.997" \
    --lower_bound "0.001,0.7,0.7,0.7" \
    --upper_bound "100.0,0.999,0.999,0.999" \
    --objective "speculative_hitrate" \
    --current_value "50.28" \
    --ordering "trace" \
    --algorithm "bandit"
```

## Key arguments

| Arg | Meaning |
| --- | --- |
| Per-param columns | e.g. `--bandit_c`, `--bandit_c_lower`, `--bandit_c_upper`. One flag per CSV column; unused algo params can be omitted. Requires `--algorithm`. |
| `--tunable_params` | Legacy comma list of params to tune. |
| `--starting_datapoints` / `--lower_bound` / `--upper_bound` | Legacy bounds aligned with `--tunable_params`. Tunable params **always** start from the datapoint, even if also passed directly. |
| `--objective` | `<mode>_<metric>`, e.g. `speculative_hitrate`. Sets the job `--mode` and the decision metric (`hitrate` or `speculation_efficiency`). Not forwarded to jobs. |
| `--current_value` | Objective of the starting config (hit-rate is a percentage). |
| `--tuning_seed` | Seed for the order params are picked. Also part of the output path. |
| `--fresh_start` | `False` (default) resumes from checkpoint; `True` starts over. |
| `--base_path` | Used only when ``ENABLE_JD=False`` (local runs). Default: `results/debug`. When running as a jd job, output goes to ``jd.jd_job_dir()`` like ``code/main.py``. |
| `--max_iter`, `--poll_interval`, `--traces`, `--seeds`, `--env_file` | Optional overrides. |

Any other `--key value` (e.g. `--tablesize`, `--base_path`, `--LTI`, `--device`, `--ordering`,
`--algorithm`) is forwarded straight to `code/main.py` as a job parameter.

## Output

When ``ENABLE_JD=True`` (default), each orchestrator run writes to its jd job
directory (``jd.jd_job_dir()``), matching ``code/main.py``.

When ``ENABLE_JD=False``, output lands under
`<base_path>/algorithm_<algo>/objective_<obj>/tunable_seed_<seed>/`:

- `tuning_log.txt` — verbose log
- `checkpoint.json` — resume state (used when `--fresh_start False`)
- `best_objective_history.csv` — per-iteration values + best objective
- `job_mapping.csv` — job id for each `(param, side, value, trace, seed)`
- `jobs/<id>/` — downloaded `lti_metrics.csv` + `summary.json`
- `aggregated/<tag>.csv` — per-second aggregated metrics

Interrupt with `Ctrl-C` anytime; rerun the same command to resume.
