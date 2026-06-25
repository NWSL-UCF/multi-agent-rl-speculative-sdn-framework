#!/usr/bin/env python3
"""Move ordering_* folders to sit directly after algorithm_* in result paths.

Old: mode_*/algorithm_*/tablesize_*/agingfactor_*/trace_*/seed_*/ordering_*/
New: mode_*/algorithm_*/ordering_*/tablesize_*/agingfactor_*/trace_*/seed_*/
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

STEP0_DIR = Path(__file__).resolve().parents[1]
ROOT = STEP0_DIR.parents[2] / "results" / "agingfactor_tablesize_experiment_data"

OLD_PATTERN = re.compile(
    r"(?P<prefix>.+/algorithm_(?P<algorithm>dqn|ppo|bandit))"
    r"/tablesize_(?P<tablesize>\d+)"
    r"/agingfactor_(?P<agingfactor>[\d.]+)"
    r"/trace_(?P<trace>[123])"
    r"/seed_(?P<seed>\d+)"
    r"/ordering_(?P<ordering>trace|source|destination)$"
)


def new_path_for(old: Path) -> Path | None:
    match = OLD_PATTERN.match(str(old))
    if not match:
        return None
    g = match.groupdict()
    return Path(
        f"{g['prefix']}/ordering_{g['ordering']}"
        f"/tablesize_{g['tablesize']}"
        f"/agingfactor_{g['agingfactor']}"
        f"/trace_{g['trace']}"
        f"/seed_{g['seed']}"
    )


def find_leaf_dirs(root: Path) -> list[Path]:
    leaves = []
    for summary in root.rglob("summary.json"):
        leaf = summary.parent
        if leaf.name.startswith("ordering_"):
            leaves.append(leaf)
    return sorted(leaves)


def remove_empty_dirs(root: Path) -> int:
    removed = 0
    for path in sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if path.is_dir() and not any(path.iterdir()):
            path.rmdir()
            removed += 1
    return removed


def reorganize(root: Path, dry_run: bool = True) -> None:
    leaves = find_leaf_dirs(root)
    print(f"Found {len(leaves)} leaf ordering directories under {root}")

    moves: list[tuple[Path, Path]] = []
    skipped = 0
    for old in leaves:
        new = new_path_for(old)
        if new is None:
            print(f"SKIP (unmatched path): {old}")
            skipped += 1
            continue
        if old == new:
            skipped += 1
            continue
        if new.exists():
            print(f"SKIP (target exists): {old} -> {new}")
            skipped += 1
            continue
        moves.append((old, new))

    print(f"Planned moves: {len(moves)}, skipped: {skipped}")
    if moves[:3]:
        print("Sample moves:")
        for old, new in moves[:3]:
            print(f"  {old.relative_to(root)}")
            print(f"    -> {new.relative_to(root)}")

    if dry_run:
        print("\nDry run only — no changes made.")
        return

    for old, new in moves:
        new.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(old), str(new))

    removed = remove_empty_dirs(root)
    print(f"\nMoved {len(moves)} directories.")
    print(f"Removed {removed} empty directories.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="Experiment results root directory",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply moves (default is dry run)",
    )
    args = parser.parse_args()
    reorganize(args.root.resolve(), dry_run=not args.apply)


if __name__ == "__main__":
    main()
