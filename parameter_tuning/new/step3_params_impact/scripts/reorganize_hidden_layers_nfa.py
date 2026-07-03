#!/usr/bin/env python3
"""Insert numberofFlowsPerAgent level under hidden_layers/{objective}/."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

STEP3_DIR = Path(__file__).resolve().parents[1]
RESULTS_ROOT = STEP3_DIR.parents[2] / "results" / "step3_param_impact"
HIDDEN_ROOT = RESULTS_ROOT / "hidden_layers"
VARIANTS = ("fixed", "interpolated")


def is_leaf_dir(path: Path) -> bool:
    return path.is_dir() and path.name.startswith("seed_") and (path / "args.json").exists()


def nfa_for_objective_branch(objective_dir: Path, variant: str) -> int:
    for args_path in (objective_dir / variant).rglob("args.json"):
        if is_leaf_dir(args_path.parent):
            return int(json.loads(args_path.read_text())["numberofFlowsPerAgent"])
    raise FileNotFoundError(f"No leaf args.json under {objective_dir / variant}")


def main() -> None:
    dry_run = "--dry-run" in sys.argv
    moves: list[tuple[Path, Path]] = []

    for objective_dir in sorted(p for p in HIDDEN_ROOT.iterdir() if p.is_dir()):
        objective = objective_dir.name
        for variant in VARIANTS:
            src = objective_dir / variant
            if not src.is_dir():
                continue
            if src.parent.name.startswith("numberofFlowsPerAgent_"):
                continue

            nfa = nfa_for_objective_branch(objective_dir, variant)
            dest = objective_dir / f"numberofFlowsPerAgent_{nfa}" / variant
            if dest.exists():
                print(f"Skip existing destination: {dest}")
                continue
            moves.append((src, dest))

    print(f"Planned moves: {len(moves)}")
    for src, dest in moves:
        print(f"  {src.relative_to(RESULTS_ROOT)} -> {dest.relative_to(RESULTS_ROOT)}")

    if dry_run:
        return

    manifest = []
    for src, dest in moves:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dest))
        manifest.append(
            {
                "source": str(src.relative_to(RESULTS_ROOT)),
                "dest": str(dest.relative_to(RESULTS_ROOT)),
            }
        )

    out = RESULTS_ROOT / "reorganize_hidden_layers_nfa_manifest.json"
    out.write_text(json.dumps(manifest, indent=2))
    print(f"Moved {len(manifest)} branches. Wrote {out}")


if __name__ == "__main__":
    main()
