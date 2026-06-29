#!/usr/bin/env python3
"""Generate LaTeX/PDF tables from code/best_grid_tablesize50.json."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

STEP0_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = STEP0_DIR.parents[2]
TABLES_DIR = STEP0_DIR / "tables"
DEFAULT_JSON = REPO_ROOT / "code" / "best_grid_tablesize50.json"

sys.path.insert(0, str(STEP0_DIR / "scripts"))

from generate_tablesize50_latex_tables import (  # noqa: E402
    ALGORITHMS,
    ALGORITHM_LABELS,
    ORDERINGS,
    TABLESIZE,
    build_unified_table_data,
    format_diff,
    format_efficiency,
    format_hit_rate,
    render_unified_pdf,
)

# Per-algorithm tunable columns (no placeholders).
ALGO_PARAM_KEYS: dict[str, list[str]] = {
    "bandit": ["agingfactor", "rewardAgingFactor", "spatialReward", "bandit_c"],
    "dqn": [
        "agingfactor",
        "rewardAgingFactor",
        "spatialReward",
        "numberofFlowsPerAgent",
        "gamma",
        "dqn_lr",
        "hidden_layers",
    ],
    "ppo": [
        "agingfactor",
        "rewardAgingFactor",
        "spatialReward",
        "gamma",
        "ppo_lr",
        "hidden_layers",
        "ppo_epochs",
    ],
}

PARAM_LABELS: dict[str, str] = {
    "agingfactor": "AF",
    "rewardAgingFactor": "RAF",
    "spatialReward": "SR",
    "bandit_c": "c",
    "numberofFlowsPerAgent": "F/A",
    "gamma": "γ",
    "dqn_lr": "LR",
    "ppo_lr": "LR",
    "hidden_layers": "HL",
    "ppo_epochs": "Ep",
}

TABLE_ORDERING_LABELS = {
    "source": "Src",
    "destination": "Dest",
    "trace": "Trace",
}

EFF_COL = "Eff."

# Union of all tunable columns across algorithms; "-" when not applicable.
COMBINED_PARAM_COLS: list[tuple[str, str]] = [
    ("agingfactor", "AF"),
    ("rewardAgingFactor", "RAF"),
    ("spatialReward", "SR"),
    ("bandit_c", "c"),
    ("numberofFlowsPerAgent", "F/A"),
    ("gamma", "γ"),
    ("learner_lr", "LR"),
    ("hidden_layers", "HL"),
    ("ppo_epochs", "Ep"),
]

N_PARAMS = len(COMBINED_PARAM_COLS)

# Match unified_tablesize50.pdf typography.
FONT_DATA = 10
FONT_HEADER = 10
FONT_GROUP = 10.5
FONT_ALGO = 10.5
ROW_GROUP_H = 0.140
ROW_HEAD_H = 0.150
ROW_DATA_H = 0.210
FIG_BASE_W = 11.0
CHAR_WIDTH_IN = 0.075
COL_PAD_IN = 0.06
FIG_MARGIN_IN = 0.40
HR_COL = "HR(%)"
IMPROV_COL = "Improv."
GAMMA_COL = "γ"
RL_COL = "RL"
GROUP_SPEC_HR = "Spec. HR"
GROUP_SR_HR = "Spec.+Reac. HR"
GROUP_SR_EFF = "Spec. Eff."


def _plain_text(text: str) -> str:
    return text.replace("$", "").replace(r"\gamma", "γ")


def _column_width_inches(header: str, values: list[str]) -> float:
    texts = [_plain_text(t) for t in [header, *values]]
    max_len = max(len(t) for t in texts)

    if header in {"Flow Order", RL_COL}:
        return max(max_len * CHAR_WIDTH_IN + COL_PAD_IN, 0.52 if header == "Flow Order" else 0.48)
    if header == IMPROV_COL:
        return max(max_len * CHAR_WIDTH_IN + COL_PAD_IN, 0.58)
    if header in {HR_COL, EFF_COL}:
        return max(max_len * CHAR_WIDTH_IN + COL_PAD_IN, 0.48)
    if header == "LR":
        return max(max_len * CHAR_WIDTH_IN + COL_PAD_IN, 0.56)
    if header == GAMMA_COL:
        return max(max_len * CHAR_WIDTH_IN + COL_PAD_IN, 0.40)
    if header in {"AF", "RAF", "SR"}:
        return max(max_len * CHAR_WIDTH_IN + COL_PAD_IN, 0.44)
    return max(max_len * CHAR_WIDTH_IN + COL_PAD_IN, 0.34)


def _column_widths_inches(headers: list[str], data: list[list[str]]) -> list[float]:
    return [
        _column_width_inches(headers[col], [row[col] for row in data])
        for col in range(len(headers))
    ]


def _col_widths_from_content(headers: list[str], data: list[list[str]]) -> list[float]:
    inches = _column_widths_inches(headers, data)
    total = sum(inches)
    return [w / total for w in inches]


def _figure_width(headers: list[str], data: list[list[str]]) -> float:
    return sum(_column_widths_inches(headers, data)) + FIG_MARGIN_IN


def format_param(key: str, value: float | int) -> str:
    if key in {"numberofFlowsPerAgent", "hidden_layers", "ppo_epochs"}:
        return str(int(value))
    if key in {"rewardAgingFactor", "spatialReward", "gamma"}:
        return f"{float(value):.2f}"
    if key == "ppo_lr":
        v = float(value)
        if v < 0.01:
            return f"{v:.4f}".rstrip("0").rstrip(".")
        return f"{v:g}"
    if key == "dqn_lr":
        return f"{float(value):.2f}"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def param_cell(algorithm: str, params: dict, col_key: str) -> str:
    if col_key == "learner_lr":
        if algorithm == "dqn":
            return format_param("dqn_lr", params["dqn_lr"])
        if algorithm == "ppo":
            return format_param("ppo_lr", params["ppo_lr"])
        return "-"
    if col_key == "bandit_c":
        if algorithm != "bandit":
            return "-"
        return format_param(col_key, params[col_key])
    if col_key == "numberofFlowsPerAgent":
        if algorithm != "dqn":
            return "-"
        return format_param(col_key, params[col_key])
    if col_key == "ppo_epochs":
        if algorithm != "ppo":
            return "-"
        return format_param(col_key, params[col_key])
    if col_key in {"gamma", "hidden_layers"}:
        if algorithm not in {"dqn", "ppo"}:
            return "-"
        return format_param(col_key, params[col_key])
    return format_param(col_key, params[col_key])


def param_cells(algorithm: str, params: dict) -> list[str]:
    return [param_cell(algorithm, params, key) for key, _ in COMBINED_PARAM_COLS]


def rows_from_best_json(best: dict, objective: str, metric_key: str) -> list[dict]:
    rows: list[dict] = []
    for algorithm in ALGORITHMS:
        for ordering in ORDERINGS:
            entry = best[objective][algorithm][ordering]
            rows.append(
                {
                    "algorithm": algorithm,
                    "ordering": ordering,
                    "aging_factor": float(entry["params"]["agingfactor"]),
                    "value": float(entry[metric_key]),
                }
            )
    return rows


def algo_param_cells(algorithm: str, params: dict) -> list[str]:
    return [
        format_param(key, params[key]) for key in ALGO_PARAM_KEYS[algorithm]
    ]


def build_algo_table_data(
    best: dict,
    algorithm: str,
    reactive_hr: float,
) -> tuple[list[str], list[list[str]]]:
    param_keys = ALGO_PARAM_KEYS[algorithm]
    param_headers = [PARAM_LABELS[k] for k in param_keys]

    headers = ["Flow Order"]
    headers.extend(param_headers)
    headers.extend([HR_COL, IMPROV_COL])
    headers.extend(param_headers)
    headers.extend([HR_COL, IMPROV_COL])
    headers.extend(param_headers)
    headers.append(EFF_COL)

    data: list[list[str]] = []
    for ordering in ORDERINGS:
        sp = best["speculative_hitrate"][algorithm][ordering]
        sr = best["speculativereactive_hitrate"][algorithm][ordering]
        eff = best["speculativereactive_speculation_efficiency"][algorithm][ordering]

        row = [TABLE_ORDERING_LABELS[ordering]]
        row.extend(algo_param_cells(algorithm, sp["params"]))
        row.extend(
            [
                format_hit_rate(sp["hitrate"]),
                format_diff(float(sp["hitrate"]) - reactive_hr),
            ]
        )
        row.extend(algo_param_cells(algorithm, sr["params"]))
        row.extend(
            [
                format_hit_rate(sr["hitrate"]),
                format_diff(float(sr["hitrate"]) - reactive_hr),
            ]
        )
        row.extend(algo_param_cells(algorithm, eff["params"]))
        row.append(format_efficiency(eff["speculation_efficiency"]))
        data.append(row)

    return headers, data


def _group_spans_from_col(start_col: int, n_params: int) -> list[tuple[int, int, str]]:
    sp_start = start_col
    sp_end = sp_start + n_params + 1
    sr_start = sp_end + 1
    sr_end = sr_start + n_params + 1
    eff_start = sr_end + 1
    eff_end = eff_start + n_params
    return [
        (sp_start, sp_end, GROUP_SPEC_HR),
        (sr_start, sr_end, GROUP_SR_HR),
        (eff_start, eff_end, GROUP_SR_EFF),
    ]


def render_algo_latex_table(
    algorithm: str,
    headers: list[str],
    data: list[list[str]],
    reactive_hr: float,
) -> str:
    n_params = len(ALGO_PARAM_KEYS[algorithm])
    n_cols = len(headers)
    col_spec = "l" + "c" * (n_cols - 1)
    groups = _group_spans_from_col(1, n_params)

    group_line_parts = [r"&"]
    cmidrules: list[str] = []
    for start, end, label in groups:
        span = end - start + 1
        group_line_parts.append(rf" \multicolumn{{{span}}}{{c}}{{{label}}} &")
        cmidrules.append(rf"\cmidrule(lr){{{start + 1}-{end + 1}}}")
    group_line = "".join(group_line_parts).rstrip(" &")

    lines = [
        r"% Requires: \usepackage{booktabs}",
        r"\begin{table*}[t]",
        r"\centering",
        rf"\caption{{Best grid-search configuration for {ALGORITHM_LABELS[algorithm]} "
        rf"(SFT size = {TABLESIZE}). Reactive baseline hit rate: {format_hit_rate(reactive_hr)}\%. "
        r"AF=Aging Factor, RAF=Reward Aging Factor, SR=Spatial Reward. "
        r"Improvement columns report hit-rate gain over reactive (\%). "
        r"Each metric group lists the hyperparameters that maximize that objective.}",
        rf"\label{{tab:grid-{algorithm}-tablesize50}}",
        r"\small",
        rf"\begin{{tabular}}{{@{{}}{col_spec}@{{}}}}",
        r"\toprule",
        group_line + r" \\",
        " ".join(cmidrules),
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    for row in data:
        lines.append(" & ".join(row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    return "\n".join(lines) + "\n"


def _render_wide_table_pdf(
    path: Path,
    headers: list[str],
    data: list[list[str]],
    groups: list[tuple[int, int, str]],
    col_widths: list[float],
    *,
    merge_col: int | None = None,
    merge_group_size: int = 3,
) -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
        }
    )

    n_cols = len(headers)
    bold_vcols = {start for start, _, _ in groups}
    n_data = len(data)
    total_h = ROW_GROUP_H + ROW_HEAD_H + n_data * ROW_DATA_H
    fig_h = total_h / 0.72
    fig_w = _figure_width(headers, data)

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    ax = fig.add_axes([0.01, 0.01, 0.98, 0.98])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    cum_x = [0.0]
    for w in col_widths:
        cum_x.append(cum_x[-1] + w)
    scale_x = 1.0 / cum_x[-1]

    def cx(col: int) -> float:
        return cum_x[col] * scale_x

    def cxmid(col: int) -> float:
        return (cum_x[col] + col_widths[col] / 2) * scale_x

    def cxr(col: int) -> float:
        return cum_x[col + 1] * scale_x

    top = 1.0
    row_tops = [top, top - ROW_GROUP_H / total_h, top - (ROW_GROUP_H + ROW_HEAD_H) / total_h]
    for _ in range(n_data):
        row_tops.append(row_tops[-1] - ROW_DATA_H / total_h)
    row_bottoms = row_tops[1:] + [0.0]

    def cell_mid_y(row: int) -> float:
        return (row_tops[row] + row_bottoms[row]) / 2

    THICK = 0.9
    THIN = 0.2
    BLACK = "black"

    def hrule(y: float, x0: float, x1: float, lw: float = THIN) -> None:
        ax.plot([x0, x1], [y, y], color=BLACK, lw=lw, clip_on=False)

    def vrule(x: float, y0: float, y1: float, lw: float = THIN) -> None:
        ax.plot([x, x], [y0, y1], color=BLACK, lw=lw, clip_on=False)

    GREY = "#d8d8d8"
    Rectangle = __import__("matplotlib.patches", fromlist=["Rectangle"]).Rectangle

    def cell_text(x: float, y: float, text: str, **kw) -> None:
        defaults = dict(ha="center", va="center", fontsize=FONT_DATA, color="black", clip_on=False)
        defaults.update(kw)
        ax.text(x, y, text, **defaults)

    x0, x1 = cx(0), cxr(n_cols - 1)
    ax.add_patch(
        Rectangle(
            (x0, row_bottoms[0]),
            x1 - x0,
            row_tops[0] - row_bottoms[0],
            facecolor=GREY,
            edgecolor="none",
            zorder=0,
            clip_on=False,
        )
    )
    ax.add_patch(
        Rectangle(
            (x0, row_bottoms[1]),
            x1 - x0,
            row_tops[1] - row_bottoms[1],
            facecolor=GREY,
            edgecolor="none",
            zorder=0,
            clip_on=False,
        )
    )

    hrule(row_tops[0], x0, x1, THICK)
    hrule(row_bottoms[0], x0, x1, THICK)
    hrule(row_bottoms[1], x0, x1, THICK)
    hrule(row_bottoms[-1], x0, x1, THICK)
    vrule(x0, 0.0, row_tops[0], THICK)
    vrule(x1, 0.0, row_tops[0], THICK)
    if merge_col is not None:
        vrule(x0, row_bottoms[0], row_tops[0], THICK)

    for start, end, label in groups:
        gx0, gx1 = cx(start), cxr(end)
        hrule(row_tops[0], gx0, gx1, THICK)
        hrule(row_bottoms[0], gx0, gx1, THIN)
        vrule(gx0, row_bottoms[0], row_tops[0], THICK)
        vrule(gx1, row_bottoms[0], row_tops[0], THICK)
        cell_text((gx0 + gx1) / 2, cell_mid_y(0), label, fontweight="bold", fontsize=FONT_GROUP)

    for col in range(n_cols):
        lw = THICK if col in bold_vcols else THIN
        vrule(cx(col), row_bottoms[1], row_tops[1], lw)
    vrule(cxr(n_cols - 1), row_bottoms[1], row_tops[1], THICK)
    for col, label in enumerate(headers):
        cell_text(cxmid(col), cell_mid_y(1), label, fontweight="bold", fontsize=FONT_HEADER)

    text_start_col = 2 if merge_col is not None else 1
    for r_idx in range(n_data):
        row = r_idx + 2
        yt, yb = row_tops[row], row_bottoms[row]
        is_group_boundary = (
            merge_col is not None
            and (r_idx % merge_group_size == merge_group_size - 1)
            and (r_idx < n_data - 1)
        )
        is_last = r_idx == n_data - 1
        row_lw = THICK if is_group_boundary or is_last else THIN
        if merge_col is not None and not is_group_boundary and not is_last:
            hrule(yb, cx(1), x1, row_lw)
        else:
            hrule(yb, x0, x1, row_lw)

        vrule(x0, yb, yt, THICK)
        for col in range(1, n_cols):
            lw = THICK if col in bold_vcols else THIN
            vrule(cx(col), yb, yt, lw)
        vrule(cxr(n_cols - 1), yb, yt, THICK)

        row_vals = data[r_idx]
        for col in range(n_cols):
            if merge_col is not None and col == merge_col:
                g_start = (r_idx // merge_group_size) * merge_group_size
                if r_idx == g_start:
                    ymid3 = (
                        row_tops[g_start + 2]
                        + row_bottoms[g_start + 2 + merge_group_size - 1]
                    ) / 2
                    cell_text(
                        cxmid(col),
                        ymid3,
                        row_vals[col],
                        fontweight="bold",
                        fontsize=FONT_ALGO,
                    )
            else:
                ha = "center" if col >= text_start_col else "left"
                x_pos = cxmid(col) if ha == "center" else (cx(col) + 0.004)
                cell_text(x_pos, cell_mid_y(row), row_vals[col], ha=ha)

    fig.savefig(path, bbox_inches="tight", pad_inches=0.05, facecolor="white", dpi=200)
    plt.close(fig)


def render_algo_pdf(
    path: Path,
    algorithm: str,
    headers: list[str],
    data: list[list[str]],
    reactive_hr: float,
) -> None:
    del reactive_hr
    n_params = len(ALGO_PARAM_KEYS[algorithm])
    _render_wide_table_pdf(
        path,
        headers,
        data,
        _group_spans_from_col(1, n_params),
        _col_widths_from_content(headers, data),
    )


def build_combined_table_data(
    best: dict,
    reactive_hr: float,
) -> tuple[list[str], list[list[str]]]:
    param_headers = [label for _, label in COMBINED_PARAM_COLS]
    headers = [RL_COL, "Flow Order"]
    headers.extend(param_headers)
    headers.extend([HR_COL, IMPROV_COL])
    headers.extend(param_headers)
    headers.extend([HR_COL, IMPROV_COL])
    headers.extend(param_headers)
    headers.append(EFF_COL)

    data: list[list[str]] = []
    for algorithm in ALGORITHMS:
        for ordering in ORDERINGS:
            sp = best["speculative_hitrate"][algorithm][ordering]
            sr = best["speculativereactive_hitrate"][algorithm][ordering]
            eff = best["speculativereactive_speculation_efficiency"][algorithm][ordering]

            row = [ALGORITHM_LABELS[algorithm], TABLE_ORDERING_LABELS[ordering]]
            row.extend(param_cells(algorithm, sp["params"]))
            row.extend(
                [
                    format_hit_rate(sp["hitrate"]),
                    format_diff(float(sp["hitrate"]) - reactive_hr),
                ]
            )
            row.extend(param_cells(algorithm, sr["params"]))
            row.extend(
                [
                    format_hit_rate(sr["hitrate"]),
                    format_diff(float(sr["hitrate"]) - reactive_hr),
                ]
            )
            row.extend(param_cells(algorithm, eff["params"]))
            row.append(format_efficiency(eff["speculation_efficiency"]))
            data.append(row)

    return headers, data


def _combined_group_spans() -> list[tuple[int, int, str]]:
    return _group_spans_from_col(2, N_PARAMS)


def _combined_col_widths(headers: list[str], data: list[list[str]]) -> list[float]:
    return _col_widths_from_content(headers, data)


def _latex_combined_rows(data: list[list[str]]) -> list[str]:
    lines: list[str] = []
    for i, row in enumerate(data):
        if i % len(ORDERINGS) == 0:
            lines.append(
                rf"\multirow{{{len(ORDERINGS)}}}{{*}}{{{row[0]}}} & "
                + " & ".join(row[1:])
                + r" \\"
            )
        else:
            lines.append(" & ".join(row[1:]) + r" \\")
    return lines


def render_combined_full_latex_table(
    headers: list[str],
    data: list[list[str]],
    reactive_hr: float,
) -> str:
    n_cols = len(headers)
    col_spec = "ll" + "c" * (n_cols - 2)
    groups = _combined_group_spans()

    group_line_parts = [r"& &"]
    cmidrules: list[str] = []
    for start, end, label in groups:
        span = end - start + 1
        group_line_parts.append(rf" \multicolumn{{{span}}}{{c}}{{{label}}} &")
        cmidrules.append(rf"\cmidrule(lr){{{start + 1}-{end + 1}}}")
    group_line = "".join(group_line_parts).rstrip(" &")

    lines = [
        r"% Requires: \usepackage{booktabs,multirow}",
        r"\begin{table*}[t]",
        r"\centering",
        rf"\caption{{Best grid-search configuration per algorithm and flow ordering (SFT size = {TABLESIZE}). "
        rf"Reactive baseline hit rate: {format_hit_rate(reactive_hr)}\%. "
        r"AF=Aging Factor, RAF=Reward Aging Factor, SR=Spatial Reward. "
        r"Improvement columns report hit-rate gain over reactive (\%). "
        r"Each metric group lists the hyperparameters that maximize that objective; "
        r"``--'' marks parameters not used by the algorithm.}",
        r"\label{tab:grid-full-tablesize50}",
        r"\scriptsize",
        rf"\begin{{tabular}}{{@{{}}{col_spec}@{{}}}}",
        r"\toprule",
        group_line + r" \\",
        " ".join(cmidrules),
        " & ".join(headers) + r" \\",
        r"\midrule",
        *_latex_combined_rows(data),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]
    return "\n".join(lines) + "\n"


def render_combined_full_pdf(
    path: Path,
    headers: list[str],
    data: list[list[str]],
    reactive_hr: float,
) -> None:
    del reactive_hr
    _render_wide_table_pdf(
        path,
        headers,
        data,
        _combined_group_spans(),
        _combined_col_widths(headers, data),
        merge_col=0,
        merge_group_size=len(ORDERINGS),
    )


def _latex_unified_rows(data: list[list[str]]) -> list[str]:
    lines: list[str] = []
    for i, row in enumerate(data):
        if i % len(ORDERINGS) == 0:
            lines.append(
                rf"\multirow{{{len(ORDERINGS)}}}{{*}}{{{row[0]}}} & "
                + " & ".join(row[1:])
                + r" \\"
            )
        else:
            lines.append(" & ".join(row[1:]) + r" \\")
    return lines


def render_unified_grid_latex_table(
    headers: list[str],
    data: list[list[str]],
    reactive_hr: float,
) -> str:
    col_spec = "ll" + "c" * (len(headers) - 2)
    lines = [
        r"% Requires: \usepackage{booktabs,multirow}",
        r"\begin{table*}[t]",
        r"\centering",
        rf"\caption{{Best grid-search metrics per algorithm and flow ordering (SFT size = {TABLESIZE}). "
        rf"Reactive baseline hit rate: {format_hit_rate(reactive_hr)}\%. "
        r"Spec.=Speculative, S+R=Speculative+Reactive. "
        r"AF=Aging Factor, Eff.=Speculation Efficiency. "
        r"Improvement columns report hit-rate gain over reactive (\%). "
        r"Each row uses the best hyperparameter configuration from the grid search.}",
        r"\label{tab:unified-grid-metrics-tablesize50}",
        r"\small",
        rf"\begin{{tabular}}{{@{{}}{col_spec}@{{}}}}",
        r"\toprule",
        rf"& & \multicolumn{{3}}{{c}}{{{GROUP_SPEC_HR}}} & \multicolumn{{3}}{{c}}{{{GROUP_SR_HR}}} & \multicolumn{{2}}{{c}}{{{GROUP_SR_EFF}}} \\",
        r"\cmidrule(lr){3-5} \cmidrule(lr){6-8} \cmidrule(lr){9-10}",
        rf"{RL_COL} & Flow Order & AF & HR(\%) & Improv. & AF & HR(\%) & Improv. & AF & Eff. \\",
        r"\midrule",
        *_latex_unified_rows(data),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]
    return "\n".join(lines) + "\n"


def _apply_short_metric_headers(headers: list[str]) -> list[str]:
    return [
        HR_COL if h == "Hit Rate (%)" else IMPROV_COL if h == "Improvement" else h
        for h in headers
    ]


def generate_grid_tables(json_path: Path, tables_dir: Path) -> list[Path]:
    with open(json_path) as f:
        best = json.load(f)

    reactive_hr = float(best["reactive_hitrate"])
    speculative_rows = rows_from_best_json(best, "speculative_hitrate", "hitrate")
    speculativereactive_rows = rows_from_best_json(
        best, "speculativereactive_hitrate", "hitrate"
    )
    efficiency_rows = rows_from_best_json(
        best, "speculativereactive_speculation_efficiency", "speculation_efficiency"
    )
    unified_headers, unified_data = build_unified_table_data(
        speculative_rows,
        speculativereactive_rows,
        efficiency_rows,
        reactive_hr,
    )
    unified_headers = _apply_short_metric_headers(unified_headers)
    combined_headers, combined_data = build_combined_table_data(best, reactive_hr)

    tables_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    tex_path = tables_dir / "unified_grid_tablesize50.tex"
    pdf_path = tables_dir / "unified_grid_tablesize50.pdf"
    tex_path.write_text(
        render_unified_grid_latex_table(unified_headers, unified_data, reactive_hr)
    )
    render_unified_pdf(pdf_path, unified_headers, unified_data, reactive_hr)
    written.extend([tex_path, pdf_path])

    full_tex = tables_dir / "grid_full_tablesize50.tex"
    full_pdf = tables_dir / "grid_full_tablesize50.pdf"
    full_tex.write_text(
        render_combined_full_latex_table(combined_headers, combined_data, reactive_hr)
    )
    render_combined_full_pdf(full_pdf, combined_headers, combined_data, reactive_hr)
    written.extend([full_tex, full_pdf])

    for algorithm in ALGORITHMS:
        algo_headers, algo_data = build_algo_table_data(best, algorithm, reactive_hr)
        algo_tex = tables_dir / f"grid_{algorithm}_tablesize50.tex"
        algo_pdf = tables_dir / f"grid_{algorithm}_tablesize50.pdf"
        algo_tex.write_text(
            render_algo_latex_table(algorithm, algo_headers, algo_data, reactive_hr)
        )
        render_algo_pdf(algo_pdf, algorithm, algo_headers, algo_data, reactive_hr)
        written.extend([algo_tex, algo_pdf])

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--tables-dir", type=Path, default=TABLES_DIR)
    args = parser.parse_args()

    json_path = args.json.resolve()
    tables_dir = args.tables_dir.resolve()

    with open(json_path) as f:
        reactive_hr = float(json.load(f)["reactive_hitrate"])
    print(f"reactive hit rate (tablesize {TABLESIZE}): {reactive_hr:.2f}%")
    print(f"source json: {json_path}")

    for path in generate_grid_tables(json_path, tables_dir):
        print(f"saved {path}")


if __name__ == "__main__":
    main()
