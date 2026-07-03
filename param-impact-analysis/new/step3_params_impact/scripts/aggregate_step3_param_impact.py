#!/usr/bin/env python3
"""Aggregate 9 leaf runs (3 traces x 3 seeds) per step3 param-impact config folder.

Reuses bucket-sum logic from ``aggregate_lti_at_agingfactor.py`` (same as step0 grid
``aggregate_grid_at_ordering.py``). Writes ``lti_metrics.csv`` and ``summary.json`` at
the param-value folder that contains ``traces_*`` or ``trace_*`` leaf trees.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

STEP3_DIR = Path(__file__).resolve().parents[1]
STEP0_SCRIPT = STEP3_DIR.parent / "step0" / "scripts" / "aggregate_lti_at_agingfactor.py"
RESULTS_ROOT = STEP3_DIR.parents[2] / "results" / "step3_param_impact"
EXPECTED_LEAVES = 9


def _load_aggregate_module():
    spec = importlib.util.spec_from_file_location("aggregate_lti_at_agingfactor", STEP0_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def is_leaf_dir(path: Path) -> bool:
    return path.is_dir() and path.name.startswith("seed_") and (path / "lti_metrics.csv").exists()


def is_trace_dir(name: str) -> bool:
    return name.startswith("traces_") or name.startswith("trace_")


def find_leaf_dirs(agg_dir: Path) -> list[Path]:
    leaves: list[Path] = []
    for trace_dir in sorted(agg_dir.iterdir()):
        if not trace_dir.is_dir() or not is_trace_dir(trace_dir.name):
            continue
        for seed_dir in sorted(trace_dir.iterdir()):
            if is_leaf_dir(seed_dir):
                leaves.append(seed_dir)
    return leaves


def discover_aggregation_dirs(root: Path) -> dict[Path, list[Path]]:
    grouped: dict[Path, list[Path]] = defaultdict(list)
    for lti_path in root.rglob("lti_metrics.csv"):
        leaf = lti_path.parent
        if not is_leaf_dir(leaf):
            continue
        trace_dir = leaf.parent
        if not is_trace_dir(trace_dir.name):
            continue
        agg_dir = trace_dir.parent
        grouped[agg_dir].append(leaf)

    return {agg_dir: sorted(set(leaves)) for agg_dir, leaves in grouped.items()}


def discover_fixed_hidden_layers_rollups(root: Path) -> dict[Path, list[Path]]:
    """Collect all leaf runs under hidden_layers_N across every hidden_layer_size."""
    groups: dict[Path, list[Path]] = {}
    for hl_dir in sorted(root.glob("hidden_layers/*/numberofFlowsPerAgent_*/fixed/hidden_layers_*")):
        if not hl_dir.is_dir() or not any(hl_dir.glob("hidden_layer_size_*")):
            continue

        leaves: list[Path] = []
        for size_dir in sorted(hl_dir.glob("hidden_layer_size_*")):
            for trace_dir in sorted(size_dir.iterdir()):
                if not trace_dir.is_dir() or not is_trace_dir(trace_dir.name):
                    continue
                for seed_dir in sorted(trace_dir.iterdir()):
                    if is_leaf_dir(seed_dir):
                        leaves.append(seed_dir)

        if leaves:
            groups[hl_dir] = sorted(set(leaves))
    return groups


def aggregate_fixed_hidden_layers_rollups(
    agg,
    root: Path,
    *,
    skip_existing: bool,
    force: bool,
) -> None:
    groups = discover_fixed_hidden_layers_rollups(root)
    expected_leaves = 45  # 5 hidden_layer_size x 3 traces x 3 seeds
    print(f"Found {len(groups)} fixed hidden_layers rollup folders", flush=True)

    written = 0
    skipped = 0
    short_runs: list[tuple[str, int]] = []
    total_leaves = 0
    t0 = time.time()

    for agg_dir, leaf_dirs in sorted(groups.items(), key=lambda item: str(item[0])):
        if skip_existing and not force and (agg_dir / "lti_metrics.csv").exists() and (agg_dir / "summary.json").exists():
            skipped += 1
            continue

        leaf_count = agg.aggregate_one(agg_dir, leaf_dirs, skip_existing=False)
        if leaf_count == 0:
            continue

        total_leaves += leaf_count
        written += 1
        if leaf_count != expected_leaves:
            short_runs.append((str(agg_dir.relative_to(root)), leaf_count))

    elapsed = time.time() - t0
    print(
        f"Fixed hidden_layers rollups: wrote {written} folders "
        f"(skipped {skipped} existing) in {elapsed:.0f}s",
        flush=True,
    )
    print(f"Total leaf runs rolled up: {total_leaves}", flush=True)
    if short_runs:
        print(f"Rollup folders with != {expected_leaves} leaf runs: {len(short_runs)}", flush=True)
        for path, count in short_runs:
            print(f"  {path}: {count} runs", flush=True)

    manifest = [
        {
            "aggregation_dir": str(agg_dir.relative_to(root)),
            "leaf_count": len(leaf_dirs),
            "rollup": "fixed_hidden_layers_all_sizes",
            "has_aggregated_lti": (agg_dir / "lti_metrics.csv").exists(),
            "has_aggregated_summary": (agg_dir / "summary.json").exists(),
        }
        for agg_dir, leaf_dirs in sorted(groups.items(), key=lambda item: str(item[0]))
    ]
    manifest_path = root / "aggregate_fixed_hidden_layers_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {manifest_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=RESULTS_ROOT,
        help="step3_param_impact results root",
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip folders that already have aggregated outputs (default: true).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-aggregate even when outputs already exist.",
    )
    parser.add_argument(
        "--rollup-fixed-hidden-layers",
        action="store_true",
        help="Only aggregate fixed hidden_layers_N across all hidden_layer_size x trace x seed.",
    )
    parser.add_argument(
        "--include-fixed-hidden-layers-rollup",
        action="store_true",
        help="After normal aggregation, also roll up fixed hidden_layers_N folders.",
    )
    args = parser.parse_args()

    agg = _load_aggregate_module()
    root = args.root.resolve()
    skip_existing = args.skip_existing and not args.force

    if args.rollup_fixed_hidden_layers:
        aggregate_fixed_hidden_layers_rollups(
            agg, root, skip_existing=skip_existing, force=args.force
        )
        return

    groups = discover_aggregation_dirs(root)
    print(f"Found {len(groups)} aggregation folders under {root}", flush=True)

    written = 0
    skipped = 0
    short_runs: list[tuple[str, int]] = []
    total_leaves = 0
    t0 = time.time()

    for i, (agg_dir, leaf_dirs) in enumerate(sorted(groups.items(), key=lambda item: str(item[0])), 1):
        if skip_existing and (agg_dir / "lti_metrics.csv").exists() and (agg_dir / "summary.json").exists():
            skipped += 1
            continue

        leaf_count = agg.aggregate_one(agg_dir, leaf_dirs, skip_existing=False)
        if leaf_count == 0:
            continue

        total_leaves += leaf_count
        written += 1
        if leaf_count != EXPECTED_LEAVES:
            short_runs.append((str(agg_dir.relative_to(root)), leaf_count))

        if written and written % 100 == 0:
            elapsed = time.time() - t0
            print(f"  wrote {written}, skipped {skipped}, elapsed {elapsed:.0f}s", flush=True)

    elapsed = time.time() - t0
    print(
        f"Wrote lti_metrics.csv + summary.json to {written} folders "
        f"(skipped {skipped} existing) in {elapsed:.0f}s",
        flush=True,
    )
    print(f"Total leaf runs aggregated: {total_leaves}", flush=True)
    if short_runs:
        print(f"Folders with != {EXPECTED_LEAVES} leaf runs: {len(short_runs)}", flush=True)
        for path, count in short_runs[:10]:
            print(f"  {path}: {count} runs", flush=True)

    manifest = [
        {
            "aggregation_dir": str(agg_dir.relative_to(root)),
            "leaf_count": len(leaf_dirs),
            "has_aggregated_lti": (agg_dir / "lti_metrics.csv").exists(),
            "has_aggregated_summary": (agg_dir / "summary.json").exists(),
        }
        for agg_dir, leaf_dirs in sorted(groups.items(), key=lambda item: str(item[0]))
    ]
    manifest_path = root / "aggregate_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {manifest_path}", flush=True)

    if args.include_fixed_hidden_layers_rollup:
        print(flush=True)
        aggregate_fixed_hidden_layers_rollups(
            agg, root, skip_existing=skip_existing, force=args.force
        )


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    main()
