"""Entry point for ternary-search hyper-parameter tuning over the jd job server.

For each tunable parameter this performs a ternary-style search: it evaluates a
left and a right midpoint around the current best value, runs every midpoint
across all traces and seeds on the jd workers, downloads the per-job
``lti_metrics.csv`` / ``summary.json``, aggregates them into per-second buckets
and compares the aggregated objective to decide how to shrink the search window.

It never runs the simulation locally -- it only talks to the jd job server API.
The implementation lives in the ``tuner`` package; this file is just the CLI shim.

When ``ENABLE_JD=True`` (default in ``tuner/config.py``), orchestrator output is
written to ``jd.jd_job_dir()`` -- the same pattern as ``code/main.py``.

Example (JobDistributor / commands.csv columns as individual params)
--------------------------------------------------------------------
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

Example (legacy comma-separated bounds)
---------------------------------------
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
"""

from tuner.app import main

if __name__ == "__main__":
    main()
