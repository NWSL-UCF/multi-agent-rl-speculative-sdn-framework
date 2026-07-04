import json
import os
import csv

BASE = "/home/ab823254/data/multi-agent-rl-speculative-sdn-framework/results/heuristic_sweep_tablesize50"
WINDOWS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200]
AGING = [0.75, 0.8, 0.85, 0.9, 0.95, 0.99, 0.991, 0.993, 0.995, 0.997, 0.999]
TRACES = [1, 2, 3]

rows = []
missing = []
for t in TRACES:
    for w in WINDOWS:
        for a in AGING:
            path = os.path.join(BASE, f"win{w}_af{a}", f"trace_{t}", "summary.json")
            if not os.path.exists(path):
                missing.append(path)
                continue
            with open(path) as f:
                s = json.load(f)
            rows.append({
                "trace": t,
                "window": w,
                "agingfactor": a,
                "hit_rate": s.get("overall_hit_rate"),
                "avg_hit_rate_lti": s.get("average_hitrate_per_lti"),
                "spec_eff": s.get("overall_speculation_efficiency"),
                "speculative_flows": s.get("total_speculative_flows"),
                "reactive_flows": s.get("total_reactive_flows"),
                "total_packets": s.get("total_packets"),
            })

out_csv = os.path.join(BASE, "sweep_results.csv")
with open(out_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print(f"Collected {len(rows)} runs, {len(missing)} missing.")
if missing:
    for m in missing[:20]:
        print("MISSING:", m)

# Best config per trace by hit rate
for t in TRACES:
    tr = [r for r in rows if r["trace"] == t]
    best = max(tr, key=lambda r: r["hit_rate"])
    print(f"\nTrace {t}: best hit_rate={best['hit_rate']:.2f}% "
          f"(window={best['window']}, af={best['agingfactor']}, "
          f"spec_eff={best['spec_eff']:.2f}, spec_flows={best['speculative_flows']})")

# Best config averaged across the 3 traces (same window+af)
from collections import defaultdict
agg = defaultdict(list)
for r in rows:
    agg[(r["window"], r["agingfactor"])].append(r["hit_rate"])
avg_rows = [(w, a, sum(v) / len(v), len(v)) for (w, a), v in agg.items()]
avg_rows.sort(key=lambda x: x[2], reverse=True)
print("\nTop 10 (window, af) by mean hit rate across traces:")
for w, a, mean_hr, n in avg_rows[:10]:
    print(f"  window={w:>3} af={a:<6} mean_hit_rate={mean_hr:.2f}% (n={n})")

print(f"\nCSV written to {out_csv}")
