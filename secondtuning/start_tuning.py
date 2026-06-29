"""Upload second-tuning orchestrator jobs from commands.csv to JobDistributor.

This script does **not** run tuning. It reads ``commands.csv`` and creates one
job per row on the **second-tuning** experiment. Workers pick those up via::

    jd_worker_cli expId=second-tuning entry_script=run_tuning.py

Each worker run then submits simulation midpoint jobs to **ternary-search**.

Examples
--------
cd secondtuning
python start_tuning.py --csv commands.csv

Preview without uploading::

    python start_tuning.py --csv commands.csv --dry_run
"""

from tuner.submit import main

if __name__ == "__main__":
    main()
