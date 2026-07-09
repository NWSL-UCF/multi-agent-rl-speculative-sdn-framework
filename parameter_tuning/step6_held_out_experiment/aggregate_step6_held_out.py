#!/usr/bin/env python3
"""Aggregate step6 held-out experiment leaf runs into per-mode summaries.

Uses the same 1-second bucket logic as ``aggregate_lti_at_agingfactor.py``:

  - ``mode_speculative``, ``mode_speculativereactive``: 9 runs (3 traces x 3 seeds)
  - all other modes: 3 runs (3 traces)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

STEP6_DIR = Path(__file__).resolve().parent
STEP0_SCRIPT = STEP6_DIR.parent / "step0_tablesize_agingfactor" / "scripts" / "aggregate_lti_at_agingfactor.py"
RESULTS_ROOT = STEP6_DIR.parents[1] / "results" / "step6_held_out_experiment"

TRACE_X_SEED_MODES = {"mode_speculative", "mode_speculativereactive"}
EXPECTED_LEAVES = {
    "trace_x_seed": 9,  # 3 traces x 3 seeds
    "trace_only": 3,    # 3 traces
}


def _load_aggregate_module():
    spec = importlib.util.spec_from_file_location("aggregate_lti_at_agingfactor", STEP0_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def find_leaf_dirs(mode_dir: Path) -> list[Path]:
    if mode_dir.name in TRACE_X_SEED_MODES:
        return sorted(
            p for p in mode_dir.glob("trace_*/seed_*")
            if (p / "lti_metrics.csv").exists()
        )
    return sorted(
        p for p in mode_dir.glob("trace_*")
        if p.is_dir() and (p / "lti_metrics.csv").exists()
    )


def expected_leaf_count(mode_dir: Path) -> int:
    return EXPECTED_LEAVES["trace_x_seed"] if mode_dir.name in TRACE_X_SEED_MODES else EXPECTED_LEAVES["trace_only"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=RESULTS_ROOT,
        help="step6_held_out_experiment results root",
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip mode folders that already have aggregated outputs (default: true).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-aggregate even when outputs already exist.",
    )
    args = parser.parse_args()

    agg = _load_aggregate_module()
    root = args.root.resolve()
    skip_existing = args.skip_existing and not args.force

    mode_dirs = sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("mode_"))
    print(f"Found {len(mode_dirs)} mode folders under {root}", flush=True)

    written = 0
    skipped = 0
    short_runs: list[tuple[str, int, int]] = []
    total_leaves = 0
    t0 = time.time()

    for mode_dir in mode_dirs:
        expected = expected_leaf_count(mode_dir)
        if skip_existing and (mode_dir / "lti_metrics.csv").exists() and (mode_dir / "summary.json").exists():
            skipped += 1
            continue

        leaf_dirs = find_leaf_dirs(mode_dir)
        leaf_count = agg.aggregate_one(mode_dir, leaf_dirs, skip_existing=False)
        if leaf_count == 0:
            print(f"  SKIP {mode_dir.name}: no leaf runs found", flush=True)
            continue

        total_leaves += leaf_count
        written += 1
        if leaf_count != expected:
            short_runs.append((mode_dir.name, leaf_count, expected))
        print(f"  {mode_dir.name}: aggregated {leaf_count}/{expected} runs", flush=True)

    elapsed = time.time() - t0
    print(
        f"Wrote lti_metrics.csv + summary.json to {written} mode folders "
        f"(skipped {skipped} existing) in {elapsed:.0f}s",
        flush=True,
    )
    print(f"Total leaf runs aggregated: {total_leaves}", flush=True)
    if short_runs:
        print("Mode folders with unexpected leaf counts:", flush=True)
        for name, count, expected in short_runs:
            print(f"  {name}: {count} runs (expected {expected})", flush=True)

    manifest = [
        {
            "mode": mode_dir.name,
            "aggregation_dir": str(mode_dir.relative_to(root)),
            "leaf_count": len(find_leaf_dirs(mode_dir)),
            "expected_leaf_count": expected_leaf_count(mode_dir),
            "has_aggregated_lti": (mode_dir / "lti_metrics.csv").exists(),
            "has_aggregated_summary": (mode_dir / "summary.json").exists(),
        }
        for mode_dir in mode_dirs
    ]
    manifest_path = root / "aggregate_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {manifest_path}", flush=True)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    main()
