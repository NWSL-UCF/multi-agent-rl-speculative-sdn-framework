#!/usr/bin/env bash
set -u

CODE=/home/ab823254/data/multi-agent-rl-speculative-sdn-framework/code
VENV=/home/ab823254/data/multi-agent-rl-speculative-sdn-framework/venv/bin/python
BASE=/home/ab823254/data/multi-agent-rl-speculative-sdn-framework/results/heuristic_sweep_tablesize50

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CODE VENV BASE

WINDOWS=(10 20 30 40 50 60 70 80 90 100 150 200)
AGING=(0.75 0.8 0.85 0.9 0.95 0.99 0.991 0.993 0.995 0.997 0.999)
TRACES=(1 2 3)

run_one() {
  local trace=$1 win=$2 af=$3
  local out=$BASE/trace_${trace}/win${win}_af${af}
  cd "$CODE" || exit 1
  "$VENV" main.py --mode heuristicspeculativereactive --heuristic hitcount \
    --speculative_window_size "$win" --agingfactor "$af" \
    --tablesize 50 --trace "$trace" --LTI 0.1 --RTI 0.01 \
    --LFUTimeInterval 10 --simulation_time 200 --device cpu \
    --base_path "$out" > "$out/run.log" 2>&1
  echo "done trace=$trace win=$win af=$af rc=$?"
}
export -f run_one

# Pre-create output dirs, then emit combos for parallel execution.
for t in "${TRACES[@]}"; do
  for w in "${WINDOWS[@]}"; do
    for a in "${AGING[@]}"; do
      mkdir -p "$BASE/trace_${t}/win${w}_af${a}"
      echo "$t $w $a"
    done
  done
done | xargs -P 12 -n 3 bash -c 'run_one "$@"' _

echo "ALL_SWEEP_RUNS_COMPLETE"
