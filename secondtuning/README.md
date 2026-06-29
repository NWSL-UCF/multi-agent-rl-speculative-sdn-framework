# Second-tuning (JobDistributor)

Orchestrator jobs are on the **second-tuning** queue (upload `commands.csv` via the
jd dashboard). Workers run one tuning job each; each iteration submits **18**
simulation jobs to **ternary-search**.

| Experiment | Role | Entry script |
| --- | --- | --- |
| **second-tuning** | Ternary-search orchestrator (one row per job) | `start_tuning.py` |
| **ternary-search** | Simulation midpoints (3 traces × 3 seeds × 2 sides) | `code/main.py` |

## Setup

1. `pip install -r requirements.txt` (needs `jd-worker`).
2. Set `JD_API_KEY` (and optional `JD_HUB_URL`) in the environment or `.env`.

## Run workers

### Orchestrator (second-tuning)

```bash
cd secondtuning
jd_worker_cli expId=second-tuning entry_script=start_tuning.py
```

Or on PSC:

```bash
sbatch run.slurm
```

jd passes each CSV column as a CLI flag (`--objective`, `--algorithm`, tunable
params, …). Checkpoints go to `jd.jd_job_dir()`.

### Simulation (ternary-search)

```bash
jd_worker_cli expId=ternary-search entry_script=code/main.py
```

## Output

Under each **second-tuning** job directory:

- `tuning_log.txt`, `checkpoint.json`, `best_objective_history.csv`
- `jobs/<sim-job-id>/` — downloaded results from **ternary-search**
