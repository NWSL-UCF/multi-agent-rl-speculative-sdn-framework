"""JD worker entry for second-tuning orchestrator jobs.

Jobs are already on the **second-tuning** queue (uploaded via the jd dashboard
from ``commands.csv``). Start workers with::

    jd_worker_cli expId=second-tuning entry_script=start_tuning.py

Each run performs ternary search and submits 18 simulation jobs per iteration
to **ternary-search**.
"""

from tuner.app import main

if __name__ == "__main__":
    main()
