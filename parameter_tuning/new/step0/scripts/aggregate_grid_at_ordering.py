#!/usr/bin/env python3
"""Aggregate leaf runs under each grid ``ordering_*`` folder into 1-second buckets.

For every ``ordering_{trace|source|destination}`` directory under
``results/grid/{bandit,dqn,ppo}``, collects all ``trace_*/seed_*`` leaf runs
(9 runs: 3 traces x 3 seeds), writes ``lti_metrics.csv`` and ``summary.json``
into the ordering folder using the same logic as ``aggregate_lti_at_agingfactor.py``.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path

STEP0_DIR = Path(__file__).resolve().parents[1]
GRID_ROOT = STEP0_DIR.parents[2] / "results" / "grid"
ALGORITHMS = ("bandit", "dqn", "ppo")
EXPECTED_LEAVES = 9


def _load_aggregate_module():
    path = STEP0_DIR / "scripts" / "aggregate_lti_at_agingfactor.py"
    spec = importlib.util.spec_from_file_location("aggregate_lti_at_agingfactor", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def find_ordering_dirs(algo_root: Path) -> list[Path]:
    """Find ordering folders without walking every leaf ``seed_*`` directory."""
    ordering_dirs: list[Path] = []
    for root, dirs, _files in os.walk(algo_root):
        root_path = Path(root)
        if root_path.name.startswith("ordering_"):
            ordering_dirs.append(root_path)
            dirs.clear()
            continue
        dirs[:] = [d for d in dirs if not d.startswith("seed_")]
    return sorted(ordering_dirs)


def aggregate_algorithm(algo_root: Path, agg, skip_existing: bool) -> None:
    ordering_dirs = find_ordering_dirs(algo_root)
    print(f"Found {len(ordering_dirs)} ordering folders under {algo_root}", flush=True)

    total_leaves = 0
    written = 0
    skipped = 0
    short_runs = []
    t0 = time.time()

    for i, ordering_dir in enumerate(ordering_dirs, 1):
        if skip_existing and (ordering_dir / "lti_metrics.csv").exists() and (ordering_dir / "summary.json").exists():
            skipped += 1
            continue

        leaf_dirs = agg.find_agingfactor_leaf_dirs(ordering_dir)
        leaf_count = agg.aggregate_one(ordering_dir, leaf_dirs, skip_existing=skip_existing)
        if leaf_count == 0:
            continue
        total_leaves += leaf_count
        written += 1
        if leaf_count != EXPECTED_LEAVES:
            short_runs.append((str(ordering_dir.relative_to(algo_root)), leaf_count))
        if written and written % 500 == 0:
            elapsed = time.time() - t0
            print(f"  wrote {written}, skipped {skipped}, elapsed {elapsed:.0f}s", flush=True)

    elapsed = time.time() - t0
    print(
        f"Wrote lti_metrics.csv + summary.json to {written} ordering folders "
        f"(skipped {skipped} existing) in {elapsed:.0f}s",
        flush=True,
    )
    print(f"Total leaf runs aggregated: {total_leaves}", flush=True)
    if short_runs:
        print(f"Folders with != {EXPECTED_LEAVES} leaf runs: {len(short_runs)}", flush=True)
        for path, count in short_runs[:5]:
            print(f"  {path}: {count} runs", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--algorithm",
        choices=[*ALGORITHMS, "all"],
        default="all",
        help="Which grid algorithm tree to aggregate (default: all).",
    )
    parser.add_argument(
        "--grid-root",
        type=Path,
        default=GRID_ROOT,
        help="Root directory containing bandit/, dqn/, ppo/ trees.",
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip ordering folders that already have aggregated outputs (default: true).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-aggregate even when outputs already exist.",
    )
    args = parser.parse_args()
    agg = _load_aggregate_module()

    skip_existing = args.skip_existing and not args.force
    algos = ALGORITHMS if args.algorithm == "all" else (args.algorithm,)
    grid_root = args.grid_root.resolve()

    for algo in algos:
        algo_root = grid_root / algo
        if not algo_root.exists():
            print(f"Skipping missing algorithm root: {algo_root}", flush=True)
            continue
        print("=" * 70, flush=True)
        print(f"Aggregating {algo}", flush=True)
        aggregate_algorithm(algo_root, agg, skip_existing=skip_existing)
        if algo != algos[-1]:
            print(flush=True)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    main()
