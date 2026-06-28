"""The immutable run configuration shared across the tuning run."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from .config import AGGREGATED_DIRNAME, JOBS_DIRNAME


@dataclass
class RunContext:
    """Everything a tuning run needs that does not change between iterations."""

    run_dir: Path
    job_params: Dict[str, str]   # static params forwarded to every job
    mode: str                    # simulation mode derived from the objective
    metric: str                  # "hitrate" or "speculation_efficiency"
    max_iter: int
    poll_interval: int
    tunable_order: List[str]     # original CLI order of tunable params (for CSV columns)
    traces: List[int]
    seeds: List[int]
    jobs_dir: Path = field(init=False)
    agg_dir: Path = field(init=False)

    def __post_init__(self):
        self.jobs_dir = self.run_dir / JOBS_DIRNAME
        self.agg_dir = self.run_dir / AGGREGATED_DIRNAME
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.jobs_dir.mkdir(exist_ok=True)
        self.agg_dir.mkdir(exist_ok=True)

    @property
    def jobs_per_midpoint(self):
        return len(self.traces) * len(self.seeds)
