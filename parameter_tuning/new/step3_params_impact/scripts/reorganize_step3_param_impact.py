#!/usr/bin/env python3
"""Reorganize flat step3_param_impact run folders into a hierarchical layout."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

STEP3_ANALYSIS = Path(__file__).resolve().parents[1]
REPO_ROOT = STEP3_ANALYSIS.parents[2]
RESULTS_ROOT = REPO_ROOT / "results" / "step3_param_impact"
DIRECTORY_JSON = STEP3_ANALYSIS / "directory.json"

PARAM_VALUE_KEYS = {
    "LFUTimeInterval": "LFUTimeInterval",
    "LTI": "LTI",
    "rewardAgingFactor": "rewardAgingFactor",
    "spatialReward": "spatialReward",
    "gamma": "gamma",
    "LR": "dqn_lr",
    "numberofFlowsPerAgent": "numberofFlowsPerAgent",
}


def fmt_value(value: float | int | None) -> str:
    if value is None:
        raise ValueError("Missing parameter value")
    value = float(value)
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    text = f"{value:g}"
    if "." not in text and "e" not in text.lower():
        return text
    return text.rstrip("0").rstrip(".") if "." in text else text


def load_directory() -> dict:
    return json.loads(DIRECTORY_JSON.read_text())


def build_run_lookup(directory: dict) -> dict[int, list[dict]]:
    lookup: dict[int, list[dict]] = {}

    for param_name, spec in directory.items():
        if param_name == "hidden_layers":
            for variant in ("interpolated", "fixed"):
                variant_spec = spec[variant]
                objectives = list(variant_spec["objective"].keys())
                bounds = {obj: variant_spec["objective"][obj] for obj in objectives}
                if len({tuple(v) for v in bounds.values()}) != 1:
                    raise ValueError(f"hidden_layers/{variant} objectives use different run ranges")
                start, end = next(iter(bounds.values()))
                for run_id in range(start, end + 1):
                    lookup[run_id] = [
                        {
                            "param": "hidden_layers",
                            "variant": variant,
                            "objective": objective,
                        }
                        for objective in objectives
                    ]
            continue

        for objective, bounds in spec["objective"].items():
            start, end = bounds
            for run_id in range(start, end + 1):
                lookup[run_id] = [
                    {
                        "param": param_name,
                        "objective": objective,
                    }
                ]

    return lookup


def destination_paths(args: dict, entry: dict) -> list[str]:
    trace = int(args["trace"])
    seed = int(args["seed"])
    objective = entry["objective"]

    if entry["param"] == "hidden_layers":
        variant = entry["variant"]
        hidden_layers = int(args["hidden_layers"])
        nfa = fmt_value(args["numberofFlowsPerAgent"])
        parts = [
            "hidden_layers",
            objective,
            f"numberofFlowsPerAgent_{nfa}",
            variant,
            f"hidden_layers_{hidden_layers}",
        ]
        if variant == "fixed":
            hidden_layer_size = int(args["hidden_layer_size"])
            parts.append(f"hidden_layer_size_{hidden_layer_size}")
        parts.extend([f"trace_{trace}", f"seed_{seed}"])
        return ["/".join(parts)]

    param_name = entry["param"]
    value_key = PARAM_VALUE_KEYS[param_name]
    value = fmt_value(args[value_key])
    return [
        "/".join(
            [
                param_name,
                objective,
                value,
                f"traces_{trace}",
                f"seed_{seed}",
            ]
        )
    ]


def collect_plan(lookup: dict[int, list[dict]]) -> list[dict]:
    plan: list[dict] = []
    missing_args: list[int] = []
    unmapped: list[int] = []

    def sort_key(path: Path) -> tuple[int, str]:
        return (0, int(path.name)) if path.name.isdigit() else (1, path.name)

    for child in sorted(RESULTS_ROOT.iterdir(), key=sort_key):
        if not child.is_dir() or not child.name.isdigit():
            continue
        run_id = int(child.name)
        entries = lookup.get(run_id)
        if not entries:
            unmapped.append(run_id)
            continue

        args_path = child / "args.json"
        if not args_path.exists():
            missing_args.append(run_id)
            continue

        args = json.loads(args_path.read_text())
        rel_paths = []
        for entry in entries:
            rel_paths.extend(destination_paths(args, entry))

        # Deduplicate while preserving order.
        seen: set[str] = set()
        unique_paths: list[str] = []
        for rel_path in rel_paths:
            if rel_path not in seen:
                seen.add(rel_path)
                unique_paths.append(rel_path)

        if entry["param"] != "hidden_layers" and len(unique_paths) != 1:
            raise ValueError(f"run {run_id} resolved to multiple destinations: {unique_paths}")

        plan.append(
            {
                "run_id": run_id,
                "entries": entries,
                "dest_paths": unique_paths,
                "primary_dest": unique_paths[0],
                "source": str(child),
            }
        )

    return plan, missing_args, unmapped


def move_run(plan_entry: dict, *, dry_run: bool) -> str:
    source = Path(plan_entry["source"])
    dest_paths = [RESULTS_ROOT / rel for rel in plan_entry["dest_paths"]]
    primary_dest = dest_paths[0]

    if not source.exists():
        if primary_dest.exists():
            return "skipped_existing"
        raise FileNotFoundError(f"Missing source run folder: {source}")

    if primary_dest.exists():
        raise FileExistsError(f"Destination already exists: {primary_dest}")

    if dry_run:
        return "planned"

    primary_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(primary_dest))

    for extra_dest in dest_paths[1:]:
        extra_dest.parent.mkdir(parents=True, exist_ok=True)
        if extra_dest.exists():
            raise FileExistsError(f"Destination already exists: {extra_dest}")
        os.symlink(os.path.relpath(primary_dest, extra_dest.parent), extra_dest)

    return "moved"


def main() -> None:
    dry_run = "--dry-run" in sys.argv
    directory = load_directory()
    lookup = build_run_lookup(directory)
    plan, missing_args, unmapped = collect_plan(lookup)

    print(f"Mapped runs: {len(plan)}")
    if missing_args:
        print(f"Missing args.json: {len(missing_args)} ({missing_args[:10]}...)")
    if unmapped:
        print(f"Unmapped numeric folders: {len(unmapped)} ({unmapped[:10]}...)")

    if dry_run:
        for item in plan[:5]:
            print(
                f"run {item['run_id']:>4} -> {item['primary_dest']}"
                + (
                    f" (+{len(item['dest_paths']) - 1} symlink)"
                    if len(item["dest_paths"]) > 1
                    else ""
                )
            )
        print("...")
        print(f"Dry run only. Total planned moves: {len(plan)}")
        return

    manifest: list[dict] = []
    moved = skipped = 0
    for item in plan:
        status = move_run(item, dry_run=False)
        item["status"] = status
        manifest.append(item)
        if status == "moved":
            moved += 1
        elif status == "skipped_existing":
            skipped += 1

    (RESULTS_ROOT / "reorganize_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Reorganized {moved} runs into {RESULTS_ROOT}")
    if skipped:
        print(f"Skipped {skipped} runs already present at destination")
    print(f"Wrote {RESULTS_ROOT / 'reorganize_manifest.json'}")


if __name__ == "__main__":
    main()
