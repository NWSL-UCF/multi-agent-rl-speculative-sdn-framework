# Second-tuning (JobDistributor)

Two separate jd experiments:

| Experiment | Role | Entry script |
| --- | --- | --- |
| **second-tuning** | One orchestrator job per `commands.csv` row | `run_tuning.py` |
| **ternary-search** | Simulation jobs (18 per tuning iteration) | `code/main.py` |

Each tuning iteration evaluates left/right midpoints → **2 × 3 traces × 3 seeds = 18** jobs on **ternary-search**.

## Setup

1. `pip install -r requirements.txt` (needs `jd-worker`).
2. Credentials in `.env` (default `/jet/home/arouf/data/ternary-search.env`):

   ```
   JD_API_KEY=jd_...
   ```

   Experiment ids are set in code (`second-tuning`, `ternary-search`).

## Workflow

### 1. Upload orchestrator jobs (login node)

```bash
cd secondtuning
python start_tuning.py --csv commands.csv
```

### 2. Run orchestrator workers

```bash
cd secondtuning
jd_worker_cli expId=second-tuning entry_script=run_tuning.py
```

Each job receives CSV columns as CLI flags (`--objective`, `--algorithm`, tunable params, …).

### 3. Run simulation workers

```bash
jd_worker_cli expId=ternary-search entry_script=code/main.py
```

These execute the 18 midpoint evaluations per tuning iteration.

## Scripts

| Script | Purpose |
| --- | --- |
| `start_tuning.py` | Submit rows from `commands.csv` → **second-tuning** |
| `run_tuning.py` | Worker entry for **second-tuning** (runs ternary search loop) |
| `code/main.py` | Worker entry for **ternary-search** (single simulation) |

## Output

Orchestrator checkpoints/logs: `jd.jd_job_dir()` on the **second-tuning** worker.

Simulation results: uploaded by `main.py` workers on **ternary-search**, downloaded into
`<second-tuning-job-dir>/jobs/<sim-job-id>/`.
