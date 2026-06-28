"""Per-second bucket aggregation of the per-job ``lti_metrics.csv`` files.

Mirrors the bucketing/metric logic from
``param-impact-analysis/new/step0/scripts/aggregate_lti_at_agingfactor.py``:
each LTI row is assigned to its ``floor(lti_start_time)`` one-second bucket, the
raw counters are summed across every job, then per-bucket hit-rate and
speculation-efficiency are derived. ``total_flows`` is recomputed as
``reactive_flows + speculative_flows`` rather than read from the CSV.

The same module can also emit a ``summary.json`` that mirrors the
data-collector summary fields produced by the reference script.
"""

import json
import statistics
from datetime import datetime

import pandas as pd

from .config import SUM_COLS
from .logging_setup import get_logger

logger = get_logger()


def _seconds_to_hms(seconds):
    """Format a duration in seconds as ``HH:MM:SS``."""
    if seconds <= 0:
        return "00:00:00"
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _sum_counters_per_bucket(job_ids, jobs_dir):
    """Sum the raw counters across all jobs, grouped by ``floor(lti_start_time)``."""
    buckets = {}
    files_used = 0

    for job_id in job_ids:
        path = jobs_dir / str(job_id) / "lti_metrics.csv"
        if not path.exists():
            logger.warning(f"Missing lti_metrics.csv for job {job_id}; excluded from aggregation.")
            continue
        try:
            df = pd.read_csv(path)
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Failed to read {path}: {exc}")
            continue
        if df.empty or "lti_start_time" not in df.columns:
            logger.warning(f"{path} is empty or missing lti_start_time; skipping.")
            continue

        files_used += 1
        df["_bucket"] = df["lti_start_time"].astype(float).astype(int)
        grouped = df.groupby("_bucket")[SUM_COLS].sum()
        for bucket, row in grouped.iterrows():
            acc = buckets.setdefault(int(bucket), {col: 0.0 for col in SUM_COLS})
            for col in SUM_COLS:
                acc[col] += float(row[col])

    logger.info(f"Aggregation used {files_used}/{len(job_ids)} job files.")
    return buckets


def _bucket_metrics(counters):
    """Derive ``total_flows``, hit-rate (percentage) and speculation-efficiency."""
    total_packets = counters["total_packets"]
    total_hits = counters["total_hits"]
    reactive_hits = counters["reactive_hits"]
    speculative_hits = counters["speculative_hits"]
    reactive_flows = counters["reactive_flows"]
    speculative_flows = counters["speculative_flows"]
    total_flows = reactive_flows + speculative_flows

    hitrate = (total_hits / total_packets * 100.0) if total_packets > 0 else 0.0

    speculation_efficiency = 0.0
    if (
        speculative_hits > 0
        and reactive_hits > 0
        and speculative_flows > 0
        and total_flows > 0
    ):
        speculation_efficiency = (
            (speculative_hits / reactive_hits) / (speculative_flows / total_flows)
        )
    return total_flows, hitrate, speculation_efficiency


def _build_agg_df(buckets):
    """Turn the summed-per-bucket counters into the aggregated metrics table."""
    rows = []
    for bucket in sorted(buckets):
        counters = buckets[bucket]
        total_flows, hitrate, spec_eff = _bucket_metrics(counters)
        row = {"second_bucket": bucket}
        row.update({col: counters[col] for col in SUM_COLS})
        row["total_flows"] = total_flows
        row["hitrate"] = hitrate
        row["speculation_efficiency"] = spec_eff
        rows.append(row)
    return pd.DataFrame(rows)


def _load_leaf_summaries(job_ids, jobs_dir):
    """Read each job's ``summary.json`` (skipping any that are missing/unreadable)."""
    summaries = []
    for job_id in job_ids:
        path = jobs_dir / str(job_id) / "summary.json"
        if not path.exists():
            logger.warning(f"Missing summary.json for job {job_id}; excluded from summary.")
            continue
        try:
            with open(path) as f:
                summaries.append(json.load(f))
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Failed to read {path}: {exc}")
    return summaries


def compute_summary(agg_df, leaf_summaries):
    """Mirror the data-collector summary fields from the aggregated bucket rows."""
    num_buckets = len(agg_df)

    total_packets = int(agg_df["total_packets"].sum()) if num_buckets else 0
    total_hits = int(agg_df["total_hits"].sum()) if num_buckets else 0
    total_misses = total_packets - total_hits

    average_hitrate_per_lti = float(agg_df["hitrate"].mean()) if num_buckets else 0.0
    average_speculation_efficiency_per_lti = (
        float(agg_df["speculation_efficiency"].mean()) if num_buckets else 0.0
    )

    speculative_hits = int(agg_df["speculative_hits"].sum()) if num_buckets else 0
    reactive_hits = int(agg_df["reactive_hits"].sum()) if num_buckets else 0

    speculation_rate_sum = 0.0
    for _, row in agg_df.iterrows():
        if row["total_flows"] > 0:
            speculation_rate_sum += row["speculative_flows"] / row["total_flows"]
    avg_speculation_rate = speculation_rate_sum / num_buckets if num_buckets else 0.0

    overall_speculation_efficiency = 0.0
    if speculative_hits > 0 and reactive_hits > 0 and avg_speculation_rate > 0:
        overall_speculation_efficiency = (speculative_hits / reactive_hits) / avg_speculation_rate

    total_speculative_flows = sum(s.get("total_speculative_flows", 0) for s in leaf_summaries)
    total_reactive_flows = sum(s.get("total_reactive_flows", 0) for s in leaf_summaries)

    wall_times = [s["wall_clock_time_seconds"] for s in leaf_summaries if "wall_clock_time_seconds" in s]
    wall_clock_time_seconds = statistics.mean(wall_times) if wall_times else 0.0

    simulation_duration_seconds = float(int(agg_df["second_bucket"].max()) + 1) if num_buckets else 0.0

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
        "total_run_time": _seconds_to_hms(simulation_duration_seconds),
        "wall_clock_time_seconds": wall_clock_time_seconds,
        "wall_clock_time": _seconds_to_hms(wall_clock_time_seconds),
        "total_lti_intervals": num_buckets,
        "aggregated_from_runs": len(leaf_summaries),
        "timestamp": datetime.now().isoformat(),
    }


def aggregate_point(job_ids, jobs_dir, agg_csv_path, summary_path=None):
    """Aggregate per-job metrics into per-second buckets and average across buckets.

    Saves the aggregated table to ``agg_csv_path`` and, when ``summary_path`` is
    given, also writes a ``summary.json`` mirroring the reference data-collector
    fields. Returns
    ``(average_per_lti_hitrate, average_per_lti_speculation_efficiency)``.
    """
    buckets = _sum_counters_per_bucket(job_ids, jobs_dir)
    agg_df = _build_agg_df(buckets)

    agg_csv_path.parent.mkdir(parents=True, exist_ok=True)
    agg_df.to_csv(agg_csv_path, index=False)

    if summary_path is not None:
        leaf_summaries = _load_leaf_summaries(job_ids, jobs_dir)
        summary = compute_summary(agg_df, leaf_summaries)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Wrote summary.json to {summary_path}")

    avg_hitrate = float(agg_df["hitrate"].mean()) if not agg_df.empty else 0.0
    avg_spec_eff = float(agg_df["speculation_efficiency"].mean()) if not agg_df.empty else 0.0
    logger.info(
        f"Aggregated -> average_per_lti_hitrate={avg_hitrate:.6f}, "
        f"average_per_lti_speculation_efficiency={avg_spec_eff:.6f} (saved {agg_csv_path})"
    )
    return avg_hitrate, avg_spec_eff
