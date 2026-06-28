"""Ternary-search hyper-parameter tuning over the JobDistributor (jd) server.

The package is split into small, single-responsibility modules:

- ``config``        : constants and tunable knobs
- ``cli``           : command-line parsing and objective parsing
- ``logging_setup`` : the shared logger and its handlers
- ``jd_client``     : every interaction with the jd job server (with retries)
- ``aggregation``   : per-second bucket aggregation of lti_metrics
- ``persistence``   : checkpoint / history / job-mapping files on disk
- ``context``       : the ``RunContext`` value object shared across the run
- ``jobs``          : build, submit and score a single midpoint (9 jobs)
- ``search``        : the ternary-search algorithm and state management
- ``app``           : wiring that turns CLI arguments into a full run
"""
