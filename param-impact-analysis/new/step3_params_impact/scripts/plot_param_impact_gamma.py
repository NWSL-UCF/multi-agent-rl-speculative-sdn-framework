#!/usr/bin/env python3
"""Plot gamma param-impact (wrapper around plot_param_impact)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from plot_param_impact import param_plots_dir, plot_param, print_param_summary

PARAM = "gamma"


def main() -> None:
    plot_param(PARAM)
    data = json.loads((param_plots_dir(PARAM) / f"{PARAM}_hitrate_spec_eff_data.json").read_text())
    print_param_summary(PARAM, data)


if __name__ == "__main__":
    main()
