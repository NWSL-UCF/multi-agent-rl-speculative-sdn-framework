#!/usr/bin/env python3
"""Reorganize flat DQN job folders into a hierarchical tree under results/grid/dqn.

Hierarchy:
  mode_{mode}/objective_{objective}/numberofFlowsPerAgent_{n}/gamma_{gamma}/
  dqn_lr_{lr}/rewardAgingFactor_{raf}/spatialReward_{sr}/hidden_layers_{layers}/
  ordering_{ordering}/trace_{trace}/seed_{seed}/

``agingfactor`` is not part of the path (it is determined by ``objective``).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

STEP0_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SRC = STEP0_DIR.parents[2] / "results" / "grid" / "algorithm_dqn"
DEFAULT_DST = STEP0_DIR.parents[2] / "results" / "grid" / "dqn"


def _fmt(value) -> str:
    if isinstance(value, float):
        return format(value, "g")
    return str(value)


def dest_dir_for_args(args: dict, dst_root: Path) -> Path:
    parts = [
        f"mode_{args['mode']}",
        f"objective_{args['objective']}",
        f"numberofFlowsPerAgent_{args['numberofFlowsPerAgent']}",
        f"gamma_{_fmt(args['gamma'])}",
        f"dqn_lr_{_fmt(args['dqn_lr'])}",
        f"rewardAgingFactor_{_fmt(args['rewardAgingFactor'])}",
        f"spatialReward_{_fmt(args['spatialReward'])}",
        f"hidden_layers_{args['hidden_layers']}",
        f"ordering_{args['ordering']}",
        f"trace_{args['trace']}",
        f"seed_{args['seed']}",
    ]
    return dst_root.joinpath(*parts)


def organize(src_root: Path, dst_root: Path, move: bool = False) -> None:
    dst_root.mkdir(parents=True, exist_ok=True)
    copied = skipped = errors = 0

    for src in sorted(src_root.iterdir(), key=lambda p: int(p.name)):
        if not src.is_dir() or not src.name.isdigit():
            continue
        args_path = src / "args.json"
        if not args_path.exists():
            print(f"skip {src.name}: missing args.json")
            errors += 1
            continue

        with open(args_path) as f:
            args = json.load(f)

        dst = dest_dir_for_args(args, dst_root)
        if dst.exists():
            skipped += 1
            continue

        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            if move:
                shutil.move(str(src), str(dst))
            else:
                shutil.copytree(src, dst)
            copied += 1
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR {src.name} -> {dst}: {exc}")
            errors += 1

        if copied % 5000 == 0 and copied:
            print(f"  progress: copied={copied} skipped={skipped} errors={errors}")

    print(
        f"Done. copied={copied} skipped={skipped} errors={errors} -> {dst_root}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--dst", type=Path, default=DEFAULT_DST)
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move folders instead of copying (default: copy).",
    )
    args = parser.parse_args()
    organize(args.src.resolve(), args.dst.resolve(), move=args.move)


if __name__ == "__main__":
    main()
