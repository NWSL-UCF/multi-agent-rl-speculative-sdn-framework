# Multi-Agent RL Speculative SDN Framework

**Pre-install SDN flow rules before packets arrive — fewer misses, lower control-plane latency.**

This repo implements RL-driven speculative placement with **Bandit, DQN, and PPO** learners, five strategies (Reactive → Optimal), and **124,659 runs** on real IoT traces. **Speculative+Reactive** beats Reactive by **7.92%** hit rate and a frequency heuristic by **6.84%**.

---

## Paper Experimental Results

The raw results from all experiments for our paper **"Reinforcement Learning Framework for Speculative Flow Placement in SDN"** (revised version submitted to the [IEEE Transactions on Machine Learning in Communications and Networking](https://www.comsoc.org/publications/journals/ieee-tmlcn)) are **not included in this repository** due to their size.

To browse the raw outputs:

1. Download [`SPECULATIVE_SDN_JOURNAL_PAPER_RESULTS.zip`](https://drive.google.com/file/d/1KaNWbGiRsA7DedimXqTAxgukEDUIIgUF/view?usp=sharing) from Google Drive
2. Unzip the archive (in the repository root or any location you prefer)
3. Open the `results/` folder inside the extracted contents to inspect JSON/CSV outputs from all experiments

---

## Quick Start

```bash
# 1. Clone and enter the repo
git clone https://github.com/NWSL-UCF/multi-agent-rl-speculative-sdn-framework.git
cd multi-agent-rl-speculative-sdn-framework

# 2. Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place trace CSVs in Pcap/ (see "Trace Data" below)

# 5. Run a simulation
cd code
python main.py \
  --mode speculativereactive \
  --algorithm ppo \
  --trace 1 \
  --seed 101 \
  --simulation_time 200 \
  --base_path ../results/my_run
```

> **Local vs cluster runs:** `main.py` sets `ENABLE_JD = True` by default, which routes output through [JobDistributor](https://jobdistributor.net/) via the [`jd-worker`](https://pypi.org/project/jd-worker/) library when running large-scale experiments on a cluster. For local development, set `ENABLE_JD = False` at the top of `code/main.py` so results are written to `--base_path`.

---

## Installation

### Requirements

- Python 3.8+
- pip

Optional: CUDA-capable GPU and PyTorch with CUDA support for faster DQN/PPO training.

### Install Python packages

From the repository root:

```bash
pip install -r requirements.txt
```

| Package | Purpose |
|---|---|
| `torch` | DQN and PPO neural networks |
| `numpy`, `pandas` | Data processing and metrics export |
| `psutil` | CPU/memory logging (`--enable_resource_logging`) |
| `matplotlib`, `wandb` | Plotting and experiment tracking (parameter tuning scripts) |
| `bloom-filter`, `mmh3`, `bitarray` | Flow-table data structures |
| `jd-worker` | [JobDistributor](https://jobdistributor.net/) worker client for cluster runs (optional for local runs) |

---

## Large-Scale Cluster Runs (JobDistributor)

The paper's 124,659-run parameter study was orchestrated with [JobDistributor](https://jobdistributor.net/) — a framework for running parameterized experiments in parallel across heterogeneous machines (laptops, HPC clusters, cloud VMs).

1. Sign up and create an experiment at [jobdistributor.net](https://jobdistributor.net/)
2. Install the worker on each compute node: `pip install jd-worker`
3. Run workers pointing at `code/main.py` as the entry script (see `parameter_tuning/` for batch scripts)
4. Keep `ENABLE_JD = True` in `code/main.py` so each job writes results to its JobDistributor job directory and uploads `summary.json` / `lti_metrics.csv` automatically

For single-machine local runs, set `ENABLE_JD = False` and use `--base_path` instead.

---

## Trace Data

The simulator replays real IoT packet captures converted to CSV. Place three trace files in a directory:

```
Pcap/
├── 1.csv
├── 2.csv
└── 3.csv
```

Each CSV must contain these columns: `No.`, `Time`, `Source`, `Destination`, `Protocol`, `Length`, `Info`.

Point the simulator at your data directory with `--pcap_base_path` (default: `<repo>/Pcap`).

---

## Running Simulations

All commands are run from the `code/` directory:

```bash
cd code
python main.py [OPTIONS]
```

### Example commands

**Speculative + reactive with PPO (recommended starting point):**
```bash
python main.py \
  --mode speculativereactive \
  --algorithm ppo \
  --trace 1 \
  --seed 101 \
  --simulation_time 200 \
  --base_path ../results/speculativereactive_ppo
```

**Reactive baseline (no learning):**
```bash
python main.py \
  --mode reactive \
  --trace 1 \
  --seed 101 \
  --simulation_time 200 \
  --base_path ../results/reactive
```

**Speculative + reactive with DQN:**
```bash
python main.py \
  --mode speculativereactive \
  --algorithm dqn \
  --trace 2 \
  --seed 101 \
  --device auto \
  --base_path ../results/speculativereactive_dqn
```

**Combinatorial bandit learner:**
```bash
python main.py \
  --mode speculativereactive \
  --algorithm bandit \
  --bandit_c 0.5 \
  --trace 1 \
  --base_path ../results/speculativereactive_bandit
```

**Heuristic baseline:**
```bash
python main.py \
  --mode heuristicspeculativereactive \
  --heuristic hitcount \
  --trace 1 \
  --base_path ../results/heuristic
```

**Use tuned parameters from the paper study:**
```bash
python main.py \
  --objective speculativereactive_hitrate \
  --algorithm bandit \
  --ordering trace \
  --trace 1 \
  --seed 101 \
  --base_path ../results/tuned_run
```

When `--objective` is set, the best `agingfactor` for the given `(objective, algorithm, ordering)` combination is loaded from `code/best_agingfactor_tablesize50.json`, and the simulation mode is set automatically.

**Enable optional detailed logging:**
```bash
python main.py \
  --mode speculativereactive \
  --algorithm ppo \
  --trace 1 \
  --enable_per_packet_logging \
  --enable_resource_logging \
  --base_path ../results/with_overhead
```

---

## Simulation Modes

| Mode | Description |
|---|---|
| `reactive` | Baseline — installs rules only after a flow miss |
| `speculative` | Learner pre-installs rules for predicted flows |
| `speculativereactive` | Combines reactive and speculative (recommended) |
| `reactiveoptimal` | Reactive with oracle eviction using future knowledge |
| `speculativereactiveoptimal` | Reactive optimal plus oracle speculative pre-install at each LTI |
| `heuristicspeculativereactive` | Hit-count heuristic instead of RL for speculative placement |

## Learning Algorithms

Used by `speculative` and `speculativereactive` modes (selected with `--algorithm`):

| Algorithm | Flag | Description |
|---|---|---|
| PPO | `--algorithm ppo` | Actor-critic policy gradient (default) |
| DQN | `--algorithm dqn` | Multi-agent Deep Q-Network |
| Bandit | `--algorithm bandit` | Combinatorial UCB bandit |

---

## Command-Line Options

Run `python main.py --help` for the full list. Options are grouped below by purpose.

### Core simulation

| Argument | Default | Description |
|---|---|---|
| `--mode` | `speculativereactive` | Simulation mode (see table above) |
| `--trace` | `1` | Trace file: `1`, `2`, or `3` |
| `--seed` | `101` | Random seed |
| `--simulation_time` | `200.0` | Duration in seconds |
| `--trace_start_time` | `0.0` | Trace timestamp (seconds) where replay begins |
| `--tablesize` | `50` | Switch flow-table capacity |
| `--ordering` | `source` | Flow sort order: `trace`, `source`, `destination` |
| `--LTI` | `0.1` | Learning Time Interval (seconds) |
| `--RTI` | `0.01` | Reactive Time Interval — control-plane delay scale (seconds) |
| `--reset_age` | `1.0` | Reset age for reactive flows |
| `--speculative_reset_age` | `0.5` | Reset age for speculative flows |
| `--agingfactor` | `0.995` | Flow aging factor |
| `--rewardAgingFactor` | `0.95` | Reward aging factor |
| `--spatialReward` | `0.75` | Spatial reward weight |
| `--LFUTimeInterval` | `10` | LFU time interval |

### Paths and device

| Argument | Default | Description |
|---|---|---|
| `--pcap_base_path` | `<repo>/Pcap` | Directory containing `1.csv`, `2.csv`, `3.csv` |
| `--base_path` | `<repo>/results/debug` | Output directory (used when `ENABLE_JD = False`) |
| `--device` | `cpu` | `auto`, `cpu`, `cuda`, `cuda:0`, `cuda:1` |

### Shared learner parameters

| Argument | Default | Description |
|---|---|---|
| `--algorithm` | `ppo` | `dqn`, `ppo`, or `bandit` |
| `--numberofFlowsPerAgent` | `10` | Flows managed per agent |
| `--gamma` | `0.9` | Discount factor (DQN and PPO) |
| `--hidden_layers` | `2` | Hidden layers (DQN and PPO) |
| `--hidden_layer_size` | `None` | Uniform hidden-layer size (`None` = auto) |
| `--batch_size` | `32` | DQN replay batch size / PPO rollout length |

### DQN (`--algorithm dqn`)

| Argument | Default | Description |
|---|---|---|
| `--dqn_lr` | `0.5` | Learning rate |
| `--dqn_epsilon_start` | `1.0` | Initial exploration rate |
| `--dqn_epsilon_end` | `0.01` | Final exploration rate |
| `--dqn_epsilon_decay` | `0.995` | Epsilon decay per LTI |
| `--dqn_target_replace_iter` | `100` | Target network update interval |
| `--dqn_memory_capacity` | `500` | Replay buffer size |
| `--dqn_learning_start_size` | `100` | Min transitions before training starts |

### PPO (`--algorithm ppo`)

| Argument | Default | Description |
|---|---|---|
| `--ppo_lr` | `3e-4` | Learning rate |
| `--ppo_clip` | `0.2` | Clipped surrogate ratio |
| `--ppo_epochs` | `4` | Update epochs per rollout |
| `--ppo_entropy_coef` | `0.01` | Entropy bonus coefficient |
| `--ppo_value_coef` | `0.5` | Value loss coefficient |
| `--ppo_gae_lambda` | `0.95` | GAE lambda |

### Bandit (`--algorithm bandit`)

| Argument | Default | Description |
|---|---|---|
| `--bandit_c` | `1.0` | UCB exploration constant |

### Heuristic (`--mode heuristicspeculativereactive`)

| Argument | Default | Description |
|---|---|---|
| `--heuristic` | `hitcount` | Heuristic for ranking flows |
| `--speculative_window_size` | `100` | Sliding window size (recent LTIs) |

### Objective-driven tuning

| Argument | Default | Description |
|---|---|---|
| `--objective` | `None` | `speculative_hitrate`, `speculativereactive_hitrate`, or `speculativereactive_speculation_efficiency` — overrides `--agingfactor` and `--mode` from `best_agingfactor_tablesize50.json` |

### Optional logging

| Argument | Default | Description |
|---|---|---|
| `--enable_per_packet_logging` | off | Write `per_packet_metrics.csv` |
| `--switch_processing_rate` | `200000000` | Switch processing rate (packets/s) for delay model |
| `--enable_resource_logging` | off | Write `lti_resource_metrics.csv` |
| `--num_cpus` | `1.0` | CPU cores for utilisation normalisation |
| `--total_ram_gb` | `8.0` | Total RAM (GB) for utilisation normalisation |

---

## Output Files

After a run completes, results are saved under `--base_path` (local runs) or the [JobDistributor](https://jobdistributor.net/) job directory (when `ENABLE_JD = True`).

### Default output (every run)

```
results/my_run/
├── args.json              # Full run configuration
├── summary.json           # Aggregate metrics
├── lti_metrics.csv        # Per-LTI performance
├── environments.json      # Python/package versions
└── info.log               # Console log
```

### Optional output (when flags are enabled)

```
results/my_run/
├── per_packet_metrics.csv      # --enable_per_packet_logging
└── lti_resource_metrics.csv    # --enable_resource_logging
```

### File descriptions and examples

#### `summary.json` — run-level aggregates

```json
{
  "total_packets": 1194,
  "total_hits": 947,
  "total_misses": 247,
  "total_speculative_flows": 25205,
  "total_reactive_flows": 247,
  "overall_hit_rate": 79.31,
  "average_hitrate_per_lti": 81.49,
  "overall_miss_rate": 20.69,
  "overall_speculation_efficiency": 6.55,
  "average_speculation_efficiency_per_lti": 0.055,
  "simulation_duration_seconds": 404.09,
  "total_run_time": "00:06:44",
  "wall_clock_time_seconds": 382.44,
  "wall_clock_time": "00:06:22",
  "total_lti_intervals": 587,
  "timestamp": "2026-07-07T20:09:08.481983"
}
```

#### `args.json` — configuration snapshot

Stores every CLI argument, the selected device, learner architecture metadata, and a timestamp. Useful for reproducing a run.

```json
{
  "mode": "speculativereactive",
  "algorithm": "bandit",
  "trace": 1,
  "seed": 101,
  "tablesize": 50,
  "simulation_time": 200.0,
  "agingfactor": 0.9247,
  "selected_device": "cpu",
  "network_architecture": {
    "algorithm": "bandit",
    "bandit_type": "comb_ucb1",
    "num_arms": 651,
    "budget_per_round": 50
  },
  "timestamp": "2026-07-07T20:09:08.480038"
}
```

#### `lti_metrics.csv` — per learning-time-interval metrics

One row per LTI with hit rates, flow counts, rewards, and speculation efficiency:

| lti_number | lti_start_time | lti_end_time | lti_duration | total_packets | total_hits | total_misses | reactive_hits | speculative_hits | total_flows | reactive_flows | speculative_flows | hit_rate | speculation_efficiency | total_evicted_flows | reward | delta_reward |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 200.140 | 200.726 | 0.586 | 1 | 0 | 1 | 0 | 0 | 50 | 0 | 50 | 0.0 | 0.0 | 0 | 0.0 | 0.0 |
| 4 | 202.103 | 202.935 | 0.832 | 1 | 1 | 0 | 0 | 1 | 50 | 2 | 48 | 100.0 | 0.0 | 46 | 12.03 | 12.03 |

#### `per_packet_metrics.csv` — per-packet delay breakdown (optional)

Written when `--enable_per_packet_logging` is set:

| id | arrival_time | switch_processing_time | control_plane_delay | total_delay | is_hit | is_speculative |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.0 | 3.63e-09 | 0.0085 | 0.0085 | 0 | 1 |
| 1 | 0.000123 | 1.44e-10 | 0.0019 | 0.0019 | 0 | 1 |

- `is_hit`: `1` if the packet matched an installed flow, `0` on miss
- `is_speculative`: `0` if hit was from a speculative rule, `1` otherwise

#### `lti_resource_metrics.csv` — CPU and memory per LTI (optional)

Written when `--enable_resource_logging` is set:

| lti_number | lti_start_time | lti_end_time | avg_rss_mb | ram_utilization_percent | cpu_utilization_percent | rss_mb | vms_mb | peak_rss_mb | cpu_time_s_delta | wall_time_s_delta |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.0 | 0.126 | 542.5 | 6.62 | 99.95 | 544.0 | 3326.5 | 583.7 | 0.93 | 0.93 |
| 1 | 0.126 | 0.298 | 544.2 | 6.64 | 99.43 | 544.4 | 3326.5 | 583.7 | 0.68 | 0.68 |

#### `environments.json` — reproducibility metadata

Records Python version, installed package versions from `requirements.txt`, and GPU info:

```json
{
  "python_version": "3.8.6",
  "compute": {
    "cuda_available": false,
    "device": "cpu"
  },
  "packages": {
    "numpy": "1.24.4",
    "pandas": "2.0.3",
    "torch": "2.4.1"
  }
}
```

#### `info.log` — runtime log

Human-readable log with device info, data-loading progress, per-LTI debug messages, and final metrics printed at the end of the run.

---

## Project Structure

```
code/
├── main.py                         # Entry point
├── best_agingfactor_tablesize50.json  # Tuned params for --objective
├── core/
│   ├── multiagent_dqn.py           # DQN learner
│   ├── ppo_agent.py                # PPO learner
│   ├── combinatorial_bandit.py     # Bandit learner
│   ├── heuristic_learner.py        # Heuristic baseline
│   ├── priority_policy.py          # Flow eviction policy
│   └── reward_function.py          # Reward shaping
├── simulation/
│   ├── reactive.py
│   ├── speculative.py
│   ├── speculativereactive.py
│   ├── reactive_optimal.py
│   ├── speculative_reactive_optimal.py
│   └── heuristic_speculativereactive.py
└── util/
    ├── data_loader.py
    ├── data_collector.py           # Metrics collection and file export
    ├── environment.py              # environments.json writer
    └── logger.py

parameter_tuning/                   # Scripts for large-scale studies
Pcap/                               # Trace CSV files (not in repo)
results/                            # Local run outputs (not in repo)
```

For the full paper experiment archive, see [Paper Experimental Results](#paper-experimental-results) above.

---

*Last updated: July 9, 2026*
