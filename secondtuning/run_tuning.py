"""JD worker entry point for one second-tuning orchestrator job.

Run with::

    jd_worker_cli expId=second-tuning entry_script=run_tuning.py

Each iteration submits 18 simulation jobs (2 midpoints × 3 traces × 3 seeds)
to the **ternary-search** experiment. Checkpoints and logs stay in this job's
``jd.jd_job_dir()``.
"""

from tuner.app import main

if __name__ == "__main__":
    main()
