#!/usr/bin/env python3
"""Aggregate leaf lti_metrics.csv files into 1-second buckets.

Targets:
  agingfactor — speculative/speculativereactive agingfactor_* folders (9 runs: 3 traces x 3 seeds)
  reactive    — mode_reactive tablesize_* folders (3 runs: 3 traces)
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd

STEP0_DIR = Path(__file__).resolve().parents[1]
ROOT = STEP0_DIR.parents[2] / "results" / "agingfactor_tablesize_experiment_data"
REACTIVE_ROOT = ROOT / "mode_reactive"

SUM_COLUMNS = [
    "total_packets",
    "total_hits",
    "total_misses",
    "reactive_hits",
    "speculative_hits",
    "reactive_flows",
    "speculative_flows",
    "total_evicted_flows",
    "reward",
    "delta_reward",
]


def seconds_to_hms(seconds: float) -> str:
    if seconds <= 0:
        return "00:00:00"
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def bucket_leaf_lti(df: pd.DataFrame) -> dict[int, dict[str, float]]:
    """Assign each LTI row to floor(lti_start_time) bucket and sum counters."""
    if df.empty or "lti_start_time" not in df.columns:
        return {}

    cols = [col for col in SUM_COLUMNS if col in df.columns]
    grouped = df.assign(_bucket=df["lti_start_time"].astype(float).astype(int)).groupby("_bucket")[cols].sum()
    return {
        int(bucket): {col: float(row[col]) for col in cols}
        for bucket, row in grouped.iterrows()
    }


def merge_bucket_dicts(all_runs: list[dict[int, dict[str, float]]]) -> dict[int, dict[str, float]]:
    merged: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for run_buckets in all_runs:
        for bucket, counts in run_buckets.items():
            for col, value in counts.items():
                merged[bucket][col] += value
    return merged


def compute_bucket_metrics(counts: dict[str, float]) -> dict[str, float]:
    total_packets = counts["total_packets"]
    total_hits = counts["total_hits"]
    reactive_hits = counts["reactive_hits"]
    speculative_hits = counts["speculative_hits"]
    reactive_flows = counts["reactive_flows"]
    speculative_flows = counts["speculative_flows"]
    total_flows = reactive_flows + speculative_flows

    hit_rate = (total_hits / total_packets * 100) if total_packets > 0 else 0.0
    speculation_efficiency = 0.0
    if (
        speculative_hits > 0
        and reactive_hits > 0
        and speculative_flows > 0
        and total_flows > 0
    ):
        speculation_efficiency = (speculative_hits / reactive_hits) / (speculative_flows / total_flows)

    return {
        **counts,
        "total_flows": total_flows,
        "hit_rate": hit_rate,
        "speculation_efficiency": speculation_efficiency,
    }


def buckets_to_dataframe(buckets: dict[int, dict[str, float]]) -> pd.DataFrame:
    rows = []
    for bucket in sorted(buckets):
        metrics = compute_bucket_metrics(buckets[bucket])
        rows.append(
            {
                "lti_number": bucket,
                "lti_start_time": float(bucket),
                "lti_end_time": float(bucket + 1),
                "lti_duration": 1.0,
                "total_packets": int(metrics["total_packets"]),
                "total_hits": int(metrics["total_hits"]),
                "total_misses": int(metrics["total_misses"]),
                "reactive_hits": int(metrics["reactive_hits"]),
                "speculative_hits": int(metrics["speculative_hits"]),
                "total_flows": int(metrics["total_flows"]),
                "reactive_flows": int(metrics["reactive_flows"]),
                "speculative_flows": int(metrics["speculative_flows"]),
                "hit_rate": metrics["hit_rate"],
                "speculation_efficiency": metrics["speculation_efficiency"],
                "total_evicted_flows": int(metrics["total_evicted_flows"]),
                "reward": metrics["reward"],
                "delta_reward": metrics["delta_reward"],
            }
        )
    return pd.DataFrame(rows)


def compute_summary_metrics(lti_df: pd.DataFrame, leaf_summaries: list[dict]) -> dict:
    """Mirror data_collector summary fields from aggregated bucket rows."""
    total_packets = int(lti_df["total_packets"].sum())
    total_hits = int(lti_df["total_hits"].sum())
    total_misses = int(lti_df["total_misses"].sum())

    num_buckets = len(lti_df)
    average_hitrate_per_lti = float(lti_df["hit_rate"].mean()) if num_buckets > 0 else 0.0
    average_speculation_efficiency_per_lti = (
        float(lti_df["speculation_efficiency"].mean()) if num_buckets > 0 else 0.0
    )

    speculative_hits = int(lti_df["speculative_hits"].sum())
    reactive_hits = int(lti_df["reactive_hits"].sum())

    speculation_rate_sum = 0.0
    if num_buckets > 0:
        flows = lti_df["total_flows"]
        mask = flows > 0
        speculation_rate_sum = float((lti_df.loc[mask, "speculative_flows"] / flows[mask]).sum())
    avg_speculation_rate = speculation_rate_sum / num_buckets if num_buckets > 0 else 0.0

    overall_speculation_efficiency = 0.0
    if speculative_hits > 0 and reactive_hits > 0 and avg_speculation_rate > 0:
        overall_speculation_efficiency = (speculative_hits / reactive_hits) / avg_speculation_rate

    total_speculative_flows = sum(s.get("total_speculative_flows", 0) for s in leaf_summaries)
    total_reactive_flows = sum(s.get("total_reactive_flows", 0) for s in leaf_summaries)

    wall_times = [s["wall_clock_time_seconds"] for s in leaf_summaries if "wall_clock_time_seconds" in s]
    wall_clock_time_seconds = statistics.mean(wall_times) if wall_times else 0.0

    simulation_duration_seconds = float(lti_df["lti_end_time"].max()) if num_buckets > 0 else 0.0

    return {
        "total_packets": total_packets,
        "total_hits": total_hits,
        "total_misses": total_misses,
        "total_speculative_flows": int(total_speculative_flows),
        "total_reactive_flows": int(total_reactive_flows),
        "overall_hit_rate": (total_hits / max(1, total_packets)) * 100,
        "average_hitrate_per_lti": average_hitrate_per_lti,
        "overall_miss_rate": (total_misses / max(1, total_packets)) * 100,
        "overall_speculation_efficiency": overall_speculation_efficiency,
        "average_speculation_efficiency_per_lti": average_speculation_efficiency_per_lti,
        "simulation_duration_seconds": simulation_duration_seconds,
        "total_run_time": seconds_to_hms(simulation_duration_seconds),
        "wall_clock_time_seconds": wall_clock_time_seconds,
        "wall_clock_time": seconds_to_hms(wall_clock_time_seconds),
        "total_lti_intervals": num_buckets,
        "aggregated_from_runs": len(leaf_summaries),
        "timestamp": datetime.now().isoformat(),
    }


def find_agingfactor_leaf_dirs(agingfactor_dir: Path) -> list[Path]:
    return sorted(
        p for p in agingfactor_dir.glob("trace_*/seed_*")
        if (p / "lti_metrics.csv").exists()
    )


def find_reactive_leaf_dirs(tablesize_dir: Path) -> list[Path]:
    return sorted(
        p for p in tablesize_dir.glob("trace_*")
        if p.is_dir() and (p / "lti_metrics.csv").exists()
    )


def is_agingfactor_dir(path: Path) -> bool:
    return path.name.startswith("agingfactor_") and any(path.glob("trace_*/seed_*"))


def is_reactive_tablesize_dir(path: Path) -> bool:
    return path.name.startswith("tablesize_") and any(path.glob("trace_*/lti_metrics.csv"))


def aggregate_one(output_dir: Path, leaf_dirs: list[Path], skip_existing: bool = False) -> int:
    if not leaf_dirs:
        return 0

    if skip_existing and (output_dir / "lti_metrics.csv").exists() and (output_dir / "summary.json").exists():
        return len(leaf_dirs)

    leaf_summaries = []
    frames = []

    for leaf in leaf_dirs:
        lti_path = leaf / "lti_metrics.csv"
        if not lti_path.exists():
            continue
        df = pd.read_csv(lti_path)
        if df.empty or "lti_start_time" not in df.columns:
            continue
        cols = [col for col in SUM_COLUMNS if col in df.columns]
        part = df[cols].copy()
        part["_bucket"] = df["lti_start_time"].astype(float).astype(int)
        frames.append(part)

        summary_path = leaf / "summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                leaf_summaries.append(json.load(f))

    if not frames:
        return 0

    all_df = pd.concat(frames, ignore_index=True)
    cols = [col for col in SUM_COLUMNS if col in all_df.columns]
    merged_df = all_df.groupby("_bucket")[cols].sum()
    buckets = {
        int(bucket): {col: float(row[col]) for col in cols}
        for bucket, row in merged_df.iterrows()
    }
    lti_df = buckets_to_dataframe(buckets)
    summary = compute_summary_metrics(lti_df, leaf_summaries)

    lti_df.to_csv(output_dir / "lti_metrics.csv", index=False)
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return len(leaf_dirs)


def aggregate_agingfactor_all(root: Path) -> None:
    agingfactor_dirs = sorted(p for p in root.rglob("agingfactor_*") if is_agingfactor_dir(p))
    print(f"Found {len(agingfactor_dirs)} agingfactor folders under {root}")

    total_leaves = 0
    written = 0
    short_runs = []

    for i, aging_dir in enumerate(agingfactor_dirs, 1):
        leaf_dirs = find_agingfactor_leaf_dirs(aging_dir)
        leaf_count = aggregate_one(aging_dir, leaf_dirs)
        if leaf_count == 0:
            continue
        total_leaves += leaf_count
        written += 1
        if leaf_count != 9:
            short_runs.append((str(aging_dir.relative_to(root)), leaf_count))
        if i % 100 == 0 or i == len(agingfactor_dirs):
            print(f"  processed {i}/{len(agingfactor_dirs)}")

    print(f"\nWrote lti_metrics.csv + summary.json to {written} agingfactor folders")
    print(f"Total leaf runs aggregated: {total_leaves}")
    if short_runs:
        print(f"Folders with != 9 leaf runs: {len(short_runs)}")
        for path, count in short_runs[:5]:
            print(f"  {path}: {count} runs")


def aggregate_reactive_all(reactive_root: Path) -> None:
    tablesize_dirs = sorted(p for p in reactive_root.glob("tablesize_*") if is_reactive_tablesize_dir(p))
    print(f"Found {len(tablesize_dirs)} tablesize folders under {reactive_root}")

    total_leaves = 0
    written = 0
    short_runs = []

    for tablesize_dir in tablesize_dirs:
        leaf_dirs = find_reactive_leaf_dirs(tablesize_dir)
        leaf_count = aggregate_one(tablesize_dir, leaf_dirs)
        if leaf_count == 0:
            continue
        total_leaves += leaf_count
        written += 1
        if leaf_count != 3:
            short_runs.append((str(tablesize_dir.relative_to(reactive_root)), leaf_count))
        print(f"  {tablesize_dir.name}: aggregated {leaf_count} traces")

    print(f"\nWrote lti_metrics.csv + summary.json to {written} tablesize folders")
    print(f"Total leaf runs aggregated: {total_leaves}")
    if short_runs:
        print(f"Folders with != 3 trace runs: {len(short_runs)}")
        for path, count in short_runs:
            print(f"  {path}: {count} runs")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        choices=["agingfactor", "reactive", "all"],
        default="agingfactor",
        help="Aggregation target: agingfactor (RL modes), reactive (mode_reactive), or all",
    )
    parser.add_argument("--root", type=Path, default=ROOT, help="Root for agingfactor aggregation")
    parser.add_argument(
        "--reactive-root",
        type=Path,
        default=REACTIVE_ROOT,
        help="Root for reactive tablesize aggregation",
    )
    args = parser.parse_args()

    if args.target in ("agingfactor", "all"):
        aggregate_agingfactor_all(args.root.resolve())
    if args.target in ("reactive", "all"):
        if args.target == "all":
            print()
        aggregate_reactive_all(args.reactive_root.resolve())


if __name__ == "__main__":
    main()
