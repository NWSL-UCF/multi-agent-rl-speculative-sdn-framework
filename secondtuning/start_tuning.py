"""Entry point for ternary-search hyper-parameter tuning over the jd job server.

For each tunable parameter this performs a ternary-style search: it evaluates a
left and a right midpoint around the current best value, runs every midpoint
across all traces and seeds on the jd workers, downloads the per-job
``lti_metrics.csv`` / ``summary.json``, aggregates them into per-second buckets
and compares the aggregated objective to decide how to shrink the search window.

It never runs the simulation locally -- it only talks to the jd job server API.
The implementation lives in the ``tuner`` package; this file is just the CLI shim.

Example
-------
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
"""

from tuner.app import main

if __name__ == "__main__":
    main()
