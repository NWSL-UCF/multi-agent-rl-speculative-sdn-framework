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

```bash
cd secondtuning
python start_tuning.py \
    --fresh_start False \
    --tuning_seed 345 \
    --tunable_params "bandit_c,rewardAgingFactor,spatialReward,agingfactor" \
    --starting_datapoints "1.0,0.95,0.75,0.997" \
    --lower_bound "0.01,0.5,0.5,0.85" \
    --upper_bound "5.0,0.99,0.99,0.999" \
    --objective "speculative_hitrate" \
    --current_value "50.28" \
    --ordering "trace" \
    --algorithm "bandit" \
    --base_path /home/ab823254/data/multi-agent-rl-speculative-sdn-framework/results/ternary_search
```

## Key arguments

| Arg | Meaning |
| --- | --- |
| `--tunable_params` | Comma list of params to tune. |
| `--starting_datapoints` / `--lower_bound` / `--upper_bound` | Aligned with `--tunable_params`. Tunable params **always** start from the datapoint, even if also passed directly. |
| `--objective` | `<mode>_<metric>`, e.g. `speculative_hitrate`. Sets the job `--mode` and the decision metric (`hitrate` or `speculation_efficiency`). Not forwarded to jobs. |
| `--current_value` | Objective of the starting config (hit-rate is a percentage). |
| `--tuning_seed` | Seed for the order params are picked. Also part of the output path. |
| `--fresh_start` | `False` (default) resumes from checkpoint; `True` starts over. |
| `--base_path` | Local results dir. **Not** sent to jobs. |
| `--max_iter`, `--poll_interval`, `--traces`, `--seeds`, `--env_file` | Optional overrides. |

Any other `--key value` (e.g. `--tablesize`, `--LTI`, `--device`, `--ordering`,
`--algorithm`) is forwarded straight to `code/main.py` as a job parameter.

## Output

Everything lands under
`<base_path>/algorithm_<algo>/objective_<obj>/tunable_seed_<seed>/`:

- `tuning_log.txt` — verbose log
- `checkpoint.json` — resume state (used when `--fresh_start False`)
- `best_objective_history.csv` — per-iteration values + best objective
- `job_mapping.csv` — job id for each `(param, side, value, trace, seed)`
- `jobs/<id>/` — downloaded `lti_metrics.csv` + `summary.json`
- `aggregated/<tag>.csv` — per-second aggregated metrics

Interrupt with `Ctrl-C` anytime; rerun the same command to resume.
