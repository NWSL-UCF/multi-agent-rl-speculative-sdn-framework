"""Per-second bucket aggregation of the per-job ``lti_metrics.csv`` files."""

import math

import pandas as pd

from .config import SUM_COLS
from .logging_setup import get_logger

logger = get_logger()


def _second_bucket(end_time):
    """Map an LTI end time to its 1-second bucket (rows <= 1.0 -> bucket 1)."""
    return max(1, int(math.ceil(round(float(end_time), 6))))


def _sum_counters_per_bucket(job_ids, jobs_dir):
    """Sum the raw counters across all jobs, grouped by second bucket."""
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
        if df.empty or "lti_end_time" not in df.columns:
            logger.warning(f"{path} is empty or missing lti_end_time; skipping.")
            continue

        files_used += 1
        df["_bucket"] = df["lti_end_time"].apply(_second_bucket)
        grouped = df.groupby("_bucket")[SUM_COLS].sum()
        for bucket, row in grouped.iterrows():
            acc = buckets.setdefault(bucket, {col: 0.0 for col in SUM_COLS})
            for col in SUM_COLS:
                acc[col] += float(row[col])

    logger.info(f"Aggregation used {files_used}/{len(job_ids)} job files.")
    return buckets


def _bucket_metrics(counters):
    """Derive hit-rate (percentage) and speculation-efficiency for one bucket."""
    total_packets = counters["total_packets"]
    total_hits = counters["total_hits"]
    reactive_hits = counters["reactive_hits"]
    speculative_hits = counters["speculative_hits"]
    speculative_flows = counters["speculative_flows"]
    total_flows = counters["total_flows"]

    hitrate = (total_hits / total_packets * 100.0) if total_packets > 0 else 0.0

    speculation_efficiency = 0.0
    if reactive_hits > 0 and speculative_flows > 0 and total_flows > 0:
        speculation_efficiency = (
            (speculative_hits / reactive_hits) / (speculative_flows / total_flows)
        )
    return hitrate, speculation_efficiency


def aggregate_point(job_ids, jobs_dir, agg_csv_path):
    """Aggregate per-job metrics into per-second buckets and average across buckets.

    Saves the aggregated table to ``agg_csv_path`` and returns
    ``(average_per_lti_hitrate, average_per_lti_speculation_efficiency)``.
    """
    buckets = _sum_counters_per_bucket(job_ids, jobs_dir)

    rows = []
    for bucket in sorted(buckets):
        counters = buckets[bucket]
        hitrate, spec_eff = _bucket_metrics(counters)
        row = {"second_bucket": bucket}
        row.update({col: counters[col] for col in SUM_COLS})
        row["hitrate"] = hitrate
        row["speculation_efficiency"] = spec_eff
        rows.append(row)

    agg_df = pd.DataFrame(rows)
    agg_csv_path.parent.mkdir(parents=True, exist_ok=True)
    agg_df.to_csv(agg_csv_path, index=False)

    avg_hitrate = float(agg_df["hitrate"].mean()) if not agg_df.empty else 0.0
    avg_spec_eff = float(agg_df["speculation_efficiency"].mean()) if not agg_df.empty else 0.0
    logger.info(
        f"Aggregated -> average_per_lti_hitrate={avg_hitrate:.6f}, "
        f"average_per_lti_speculation_efficiency={avg_spec_eff:.6f} (saved {agg_csv_path})"
    )
    return avg_hitrate, avg_spec_eff
