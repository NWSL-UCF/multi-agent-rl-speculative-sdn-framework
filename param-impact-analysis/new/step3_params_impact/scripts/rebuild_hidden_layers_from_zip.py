#!/usr/bin/env python3
"""Rebuild hidden_layers tree from param-impact zip using directory.json mapping."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

STEP3_DIR = Path(__file__).resolve().parents[1]
REORG_SCRIPT = STEP3_DIR / "scripts" / "reorganize_step3_param_impact.py"
ZIP_ROOT = (
    STEP3_DIR.parents[2]
    / "results"
    / "compressed_results"
    / "param-impact_extracted"
    / "param-impact"
)
RESULTS_ROOT = STEP3_DIR.parents[2] / "results" / "step3_param_impact"
HIDDEN_ROOT = RESULTS_ROOT / "hidden_layers"


def _load_reorg_module():
    spec = importlib.util.spec_from_file_location("reorganize_step3_param_impact", REORG_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def hidden_layers_plan(reorg) -> list[dict]:
    directory = reorg.load_directory()
    lookup = reorg.build_run_lookup(directory)
    plan: list[dict] = []

    for run_id, entries in sorted(lookup.items()):
        if entries[0]["param"] != "hidden_layers":
            continue
        source = ZIP_ROOT / str(run_id)
        if not (source / "args.json").exists():
            raise FileNotFoundError(f"Missing zip run {run_id}: {source}")
        args = json.loads((source / "args.json").read_text())
        rel_paths: list[str] = []
        for entry in entries:
            rel_paths.extend(reorg.destination_paths(args, entry))
        seen: set[str] = set()
        unique_paths = []
        for rel in rel_paths:
            if rel not in seen:
                seen.add(rel)
                unique_paths.append(rel)
        plan.append(
            {
                "run_id": run_id,
                "source": str(source),
                "dest_paths": unique_paths,
                "primary_dest": unique_paths[0],
            }
        )
    return plan


def deploy_run(item: dict, *, dry_run: bool) -> None:
    source = Path(item["source"])
    dest_paths = [RESULTS_ROOT / rel for rel in item["dest_paths"]]
    primary_dest = dest_paths[0]

    if dry_run:
        return

    if primary_dest.exists():
        if primary_dest.is_symlink():
            primary_dest.unlink()
        else:
            shutil.rmtree(primary_dest)

    primary_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, primary_dest)

    for extra_dest in dest_paths[1:]:
        if extra_dest.exists():
            if extra_dest.is_symlink():
                extra_dest.unlink()
            else:
                shutil.rmtree(extra_dest)
        extra_dest.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(os.path.relpath(primary_dest, extra_dest.parent), extra_dest)


def main() -> None:
    dry_run = "--dry-run" in sys.argv
    reorg = _load_reorg_module()
    plan = hidden_layers_plan(reorg)

    print(f"hidden_layers runs from zip: {len(plan)}")
    print(f"example run {plan[0]['run_id']}: {plan[0]['dest_paths']}")

    if dry_run:
        for item in plan[:4]:
            print(f"  {item['run_id']} -> {item['dest_paths']}")
        print("...")
        return

    if HIDDEN_ROOT.exists():
        shutil.rmtree(HIDDEN_ROOT)

    for item in plan:
        deploy_run(item, dry_run=False)

    manifest = plan
    out = RESULTS_ROOT / "rebuild_hidden_layers_manifest.json"
    out.write_text(json.dumps(manifest, indent=2))
    print(f"Rebuilt {HIDDEN_ROOT} from {ZIP_ROOT}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
