# Multi-Agent RL Speculative SDN Framework

Multi-agent Deep Q-Network framework for speculative flow placement in SDN. RL agents proactively install rules for unseen flows to reduce control-plane latency. Includes large-scale parameter study (130K+ runs), interpolated DQN architecture, Ternary Search tuning, and optimal benchmark. Evaluated on real IoT traces.

---

## Setup

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Prepare data**

Place pcap CSV trace files (`1.csv`, `2.csv`, `3.csv`) in a directory (default: `/home/rouf/data/raw/Pcap`). Each CSV must have columns: `No.`, `Time`, `Source`, `Destination`, `Protocol`, `Length`, `Info`.

---

## Running

All commands are run from the `code/` directory:

```bash
cd code
python main.py [OPTIONS]
```

**Minimal example (recommended defaults):**
```bash
python main.py --mode speculativereactive --trace 1 --seed 101 \
  --reset_age 1.0 --speculative_reset_age 0.3 --simulation_time 600 --device auto
```

---

## Key Options

| Argument | Default | Description |
|---|---|---|
| `--mode` | `speculativereactive` | `reactive`, `speculative`, `speculativereactive`, `reactiveoptimal`, `speculativereactiveoptimal` |
| `--trace` | `1` | Trace file to use: `1`, `2`, or `3` |
| `--simulation_time` | `20` | Simulation duration in seconds |
| `--tablesize` | `70` | Switch flow table size |
| `--seed` | `101` | Random seed for reproducibility |
| `--device` | `auto` | `auto`, `cpu`, `cuda`, `cuda:0`, `cuda:1` |
| `--pcap_base_path` | `/home/rouf/data/raw/Pcap` | Path to pcap CSV files |
| `--base_path` | `./results_speculativereactive` | Output directory for results |
| `--LR` | `0.75` | DQN learning rate |
| `--gamma` | `0.9` | Discount factor |
| `--batch_size` | `64` | Experience replay batch size |

---

## Modes

- **`reactive`** — baseline; installs rules only after a flow miss
- **`speculative`** — DQN agents pre-install rules for predicted flows
- **`speculativereactive`** — combines both approaches (recommended)
- **`reactiveoptimal`** — reactive install-on-miss with oracle eviction using future knowledge
- **`speculativereactiveoptimal`** — reactive optimal plus oracle speculative pre-install at each LTI

---

## Project Structure

```
code/
├── main.py                  # Entry point
├── core/
│   ├── multiagent_dqn.py    # Multi-agent DQN
│   ├── priority_policy.py   # Flow eviction policy
│   └── reward_function.py   # Reward shaping
├── simulation/
│   ├── reactive.py
│   ├── speculative.py
│   ├── speculativereactive.py
│   ├── reactive_optimal.py
│   └── speculative_reactive_optimal.py
└── util/
    ├── data_loader.py
    ├── data_collector.py
    └── logger.py
```

Results are saved to `--base_path` as JSON/CSV files after each run.
