#!/usr/bin/env python3
"""Generate LaTeX and PDF tables for best metrics per algorithm×ordering at tablesize 50."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.table import Table

STEP0_DIR = Path(__file__).resolve().parents[1]
ROOT = STEP0_DIR.parents[2] / "results" / "agingfactor_tablesize_experiment_data"
TABLES_DIR = STEP0_DIR / "tables"

TABLESIZE = 50
ALGORITHMS = ["bandit", "ppo", "dqn"]
ORDERINGS = ["source", "destination", "trace"]

ALGORITHM_LABELS = {
    "bandit": "Bandit",
    "ppo": "PPO",
    "dqn": "DQN",
}
ORDERING_LABELS = {
    "source": "Source",
    "destination": "Destination",
    "trace": "Trace",
}
TABLE_ORDERING_LABELS = {
    "source": "Src",
    "destination": "Dest",
    "trace": "Trace",
}
AF_COL = "AF"
HR_COL = "HR(%)"
IMPROV_COL = "Improv."
EFF_COL = "Eff."
RL_COL = "RL"
ORDER_COL = "Order"
GROUP_SPEC_HR = "Spec. HR"
GROUP_SR_HR = "Spec.+Reac. HR"
GROUP_SR_EFF = "Spec. Eff."
UNIFIED_GROUP_LABELS = [
    (2, 4, GROUP_SPEC_HR),
    (5, 7, GROUP_SR_HR),
    (8, 9, GROUP_SR_EFF),
]


def read_summary(summary_path: Path) -> dict:
    with open(summary_path) as f:
        return json.load(f)


def load_reactive_hit_rate(root: Path, tablesize: int) -> float:
    summary = read_summary(root / "mode_reactive" / f"tablesize_{tablesize}" / "summary.json")
    return float(summary["average_hitrate_per_lti"])


def best_per_algo_ordering(
    root: Path,
    mode: str,
    tablesize: int,
    metric_key: str,
) -> list[dict]:
    rows: list[dict] = []
    for algorithm in ALGORITHMS:
        for ordering in ORDERINGS:
            base = (
                root
                / f"mode_{mode}"
                / f"algorithm_{algorithm}"
                / f"ordering_{ordering}"
                / f"tablesize_{tablesize}"
            )
            best_value: float | None = None
            best_af: float | None = None
            for summary_path in base.glob("agingfactor_*/summary.json"):
                af_match = re.search(r"agingfactor_([\d.]+)/summary\.json$", str(summary_path))
                if not af_match:
                    continue
                value = read_summary(summary_path).get(metric_key)
                if value is None:
                    continue
                value = float(value)
                if best_value is None or value > best_value:
                    best_value = value
                    best_af = float(af_match.group(1))
            if best_value is None or best_af is None:
                continue
            rows.append(
                {
                    "algorithm": algorithm,
                    "ordering": ordering,
                    "aging_factor": best_af,
                    "value": best_value,
                }
            )
    return rows


def format_af(af: float) -> str:
    return f"{af:g}"


def format_hit_rate(value: float) -> str:
    return f"{value:.2f}"


def format_diff(diff: float) -> str:
    sign = "+" if diff >= 0 else "-"
    return f"{sign}{abs(diff):.2f}"


def format_efficiency(value: float) -> str:
    return f"{value:.2f}"


def render_hit_rate_table(
    caption: str,
    label: str,
    rows: list[dict],
    reactive_hr: float,
) -> str:
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Algorithm & Order & Aging Factor & Hit Rate (\%) & Difference with Reactive \\",
        r"\midrule",
    ]
    for row in rows:
        diff = row["value"] - reactive_hr
        lines.append(
            " & ".join(
                [
                    ALGORITHM_LABELS[row["algorithm"]],
                    ORDERING_LABELS[row["ordering"]],
                    format_af(row["aging_factor"]),
                    format_hit_rate(row["value"]),
                    format_diff(diff),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines) + "\n"


def render_efficiency_table(caption: str, label: str, rows: list[dict]) -> str:
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{llcc}",
        r"\toprule",
        r"Algorithm & Order & Aging Factor & Speculation Efficiency \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                [
                    ALGORITHM_LABELS[row["algorithm"]],
                    ORDERING_LABELS[row["ordering"]],
                    format_af(row["aging_factor"]),
                    format_efficiency(row["value"]),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines) + "\n"


def _hit_rate_rows_latex(rows: list[dict], reactive_hr: float) -> list[str]:
    lines: list[str] = []
    for row in rows:
        diff = row["value"] - reactive_hr
        lines.append(
            " & ".join(
                [
                    ALGORITHM_LABELS[row["algorithm"]],
                    ORDERING_LABELS[row["ordering"]],
                    format_af(row["aging_factor"]),
                    format_hit_rate(row["value"]),
                    format_diff(diff),
                ]
            )
            + r" \\"
        )
    return lines


def _efficiency_rows_latex(rows: list[dict]) -> list[str]:
    lines: list[str] = []
    for row in rows:
        lines.append(
            " & ".join(
                [
                    ALGORITHM_LABELS[row["algorithm"]],
                    ORDERING_LABELS[row["ordering"]],
                    format_af(row["aging_factor"]),
                    format_efficiency(row["value"]),
                ]
            )
            + r" \\"
        )
    return lines


def render_combined_ieee_table(
    speculative_rows: list[dict],
    speculativereactive_rows: list[dict],
    efficiency_rows: list[dict],
    reactive_hr: float,
) -> str:
    """Single table* spanning both IEEE columns with all three sub-tables."""
    lines = [
        r"% Requires: \usepackage{booktabs}",
        r"\begin{table*}[t]",
        r"\centering",
        rf"\caption{{Best configuration per algorithm and flow ordering (SFT size = {TABLESIZE}). "
        rf"Reactive baseline hit rate: {format_hit_rate(reactive_hr)}\%. "
        r"Each row reports the aging factor that maximizes the reported metric.}",
        r"\label{tab:combined-metrics-tablesize50}",
        r"\small",
        r"\textbf{(a) Speculative Hit Rate}\\[0.4em]",
        r"\begin{tabular}{@{}llccc@{}}",
        r"\toprule",
        r"Algorithm & Order & Aging Factor & Hit Rate (\%) & $\Delta$ Reactive \\",
        r"\midrule",
        *_hit_rate_rows_latex(speculative_rows, reactive_hr),
        r"\bottomrule",
        r"\end{tabular}",
        r"",
        r"\vspace{0.8em}",
        r"\textbf{(b) Speculative+Reactive Hit Rate}\\[0.4em]",
        r"\begin{tabular}{@{}llccc@{}}",
        r"\toprule",
        r"Algorithm & Order & Aging Factor & Hit Rate (\%) & $\Delta$ Reactive \\",
        r"\midrule",
        *_hit_rate_rows_latex(speculativereactive_rows, reactive_hr),
        r"\bottomrule",
        r"\end{tabular}",
        r"",
        r"\vspace{0.8em}",
        r"\textbf{(c) Speculative+Reactive Speculation Efficiency}\\[0.4em]",
        r"\begin{tabular}{@{}llcc@{}}",
        r"\toprule",
        r"Algorithm & Order & Aging Factor & Speculation Efficiency \\",
        r"\midrule",
        *_efficiency_rows_latex(efficiency_rows),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]
    return "\n".join(lines) + "\n"


def _index_rows(rows: list[dict]) -> dict[tuple[str, str], dict]:
    return {(row["algorithm"], row["ordering"]): row for row in rows}


def build_best_agingfactor_json(
    speculative_rows: list[dict],
    speculativereactive_rows: list[dict],
    efficiency_rows: list[dict],
    reactive_hr: float,
) -> dict:
    def rows_to_ordering_map(
        rows: list[dict],
        value_key: str,
    ) -> dict[str, dict[str, dict[str, float]]]:
        by_algo: dict[str, dict[str, dict[str, float]]] = {
            algorithm: {} for algorithm in ALGORITHMS
        }
        for row in rows:
            by_algo[row["algorithm"]][row["ordering"]] = {
                "params": {"agingfactor": row["aging_factor"]},
                value_key: round(row["value"], 2),
            }
        return {
            algorithm: {ordering: by_algo[algorithm][ordering] for ordering in ORDERINGS}
            for algorithm in ALGORITHMS
        }

    return {
        "reactive_hitrate": round(reactive_hr, 2),
        "speculative_hitrate": rows_to_ordering_map(speculative_rows, "hitrate"),
        "speculativereactive_hitrate": rows_to_ordering_map(speculativereactive_rows, "hitrate"),
        "speculativereactive_speculation_efficiency": rows_to_ordering_map(
            efficiency_rows, "speculation_efficiency"
        ),
    }


def build_unified_table_data(
    speculative_rows: list[dict],
    speculativereactive_rows: list[dict],
    efficiency_rows: list[dict],
    reactive_hr: float,
) -> tuple[list[str], list[list[str]]]:
    sp_by_key = _index_rows(speculative_rows)
    sr_by_key = _index_rows(speculativereactive_rows)
    eff_by_key = _index_rows(efficiency_rows)

    headers = [
        RL_COL,
        ORDER_COL,
        AF_COL,
        HR_COL,
        IMPROV_COL,
        AF_COL,
        HR_COL,
        IMPROV_COL,
        AF_COL,
        EFF_COL,
    ]
    data: list[list[str]] = []
    for algorithm in ALGORITHMS:
        for ordering in ORDERINGS:
            key = (algorithm, ordering)
            sp = sp_by_key[key]
            sr = sr_by_key[key]
            eff = eff_by_key[key]
            data.append(
                [
                    ALGORITHM_LABELS[algorithm],
                    TABLE_ORDERING_LABELS[ordering],
                    format_af(sp["aging_factor"]),
                    format_hit_rate(sp["value"]),
                    format_diff(sp["value"] - reactive_hr),
                    format_af(sr["aging_factor"]),
                    format_hit_rate(sr["value"]),
                    format_diff(sr["value"] - reactive_hr),
                    format_af(eff["aging_factor"]),
                    format_efficiency(eff["value"]),
                ]
            )
    return headers, data


def _latex_unified_rows(data: list[list[str]]) -> list[str]:
    lines: list[str] = []
    for i, row in enumerate(data):
        if i % len(ORDERINGS) == 0:
            lines.append(rf"\multirow{{{len(ORDERINGS)}}}{{*}}{{{row[0]}}} & " + " & ".join(row[1:]) + r" \\")
        else:
            lines.append(" & ".join(row[1:]) + r" \\")
    return lines


def _pdf_display_data_with_merged_algorithm(data: list[list[str]]) -> list[list[str]]:
    display = [row[:] for row in data]
    for group_start in range(0, len(data), len(ORDERINGS)):
        algorithm = data[group_start][0]
        for offset in range(len(ORDERINGS)):
            display[group_start + offset][0] = algorithm if offset == 1 else ""
    return display


UNIFIED_GROUP_HEADERS = list(UNIFIED_GROUP_LABELS)

UNIFIED_CHAR_WIDTH_IN = 0.068
UNIFIED_COL_PAD_IN = 0.035
UNIFIED_FIG_MARGIN_IN = 0.28


def _unified_column_width_inches(header: str, values: list[str]) -> float:
    texts = [header, *[v for v in values if v]]
    max_len = max(len(t) for t in texts)
    width = max_len * UNIFIED_CHAR_WIDTH_IN + UNIFIED_COL_PAD_IN
    if header == HR_COL:
        width = max(width, 0.43)
    return width


def _unified_col_widths_normalized(headers: list[str], data: list[list[str]]) -> list[float]:
    inches = [
        _unified_column_width_inches(header, [row[col] for row in data])
        for col, header in enumerate(headers)
    ]
    total = sum(inches)
    return [w / total for w in inches]


def _unified_figure_width(headers: list[str], data: list[list[str]]) -> float:
    inches = [
        _unified_column_width_inches(header, [row[col] for row in data])
        for col, header in enumerate(headers)
    ]
    return sum(inches) + UNIFIED_FIG_MARGIN_IN


def _pdf_group_header_row(n_cols: int) -> list[str]:
    row = [""] * n_cols
    for start, end, label in UNIFIED_GROUP_HEADERS:
        row[(start + end) // 2] = label
    return row


def _style_horizontal_group_headers(table: Table, row_idx: int, n_cols: int) -> None:
    for col in range(n_cols):
        cell = table[(row_idx, col)]
        cell.set_facecolor("white")
        cell.set_edgecolor("black")
        cell.set_linewidth(0.8)
        cell.set_text_props(weight="bold", ha="center", va="center")

    for start, end, _ in UNIFIED_GROUP_HEADERS:
        mid = (start + end) // 2
        for col in range(start, end + 1):
            cell = table[(row_idx, col)]
            if col == start:
                cell.visible_edges = "LRT"
            elif col == end:
                cell.visible_edges = "RT"
            else:
                cell.visible_edges = "T"
            if col != mid:
                cell.get_text().set_text("")


def _style_unified_pdf_table(table: Table, headers: list[str], n_data_rows: int) -> None:
    n_cols = len(headers)
    n_rows = 2 + n_data_rows
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1.0, 1.35)

    for row in range(n_rows):
        for col in range(n_cols):
            cell = table[(row, col)]
            cell.set_facecolor("white")
            cell.set_edgecolor("black")
            cell.set_linewidth(0.8)

    _style_horizontal_group_headers(table, 0, n_cols)
    for col in range(2, n_cols):
        table[(0, col)].get_text().set_fontsize(8)

    for col in range(n_cols):
        cell = table[(1, col)]
        cell.set_text_props(weight="bold", ha="center")

    header_offset = 2
    for row in range(header_offset, header_offset + n_data_rows):
        for col in range(n_cols):
            cell = table[(row, col)]
            ha = "center" if col >= 2 else "left"
            cell.get_text().set_ha(ha)

    _style_merged_algorithm_column(table, n_data_rows, header_offset=header_offset)


def _style_merged_algorithm_column(table: Table, n_data_rows: int, *, header_offset: int = 1) -> None:
    n_orderings = len(ORDERINGS)
    for group_start in range(0, n_data_rows, n_orderings):
        for offset in range(n_orderings):
            table_row = group_start + offset + header_offset
            cell = table[(table_row, 0)]
            if offset == 0:
                cell.visible_edges = "LRT"
            elif offset == 1:
                cell.visible_edges = "LR"
                cell.get_text().set_va("center")
                cell.get_text().set_fontweight("bold")
            else:
                cell.visible_edges = "LRB"


def render_unified_latex_table(
    headers: list[str],
    data: list[list[str]],
    reactive_hr: float,
) -> str:
    col_spec = "ll" + "c" * (len(headers) - 2)
    lines = [
        r"% Requires: \usepackage{booktabs,multirow}",
        r"\begin{table*}[t]",
        r"\centering",
        rf"\caption{{Best metrics per algorithm and flow ordering (SFT size = {TABLESIZE}). "
        rf"Reactive baseline hit rate: {format_hit_rate(reactive_hr)}\%. "
        r"Spec.=Speculative, S+R=Speculative+Reactive. "
        r"AF=Aging Factor, Eff.=Speculation Efficiency. "
        r"Each metric uses the aging factor that maximizes it.}",
        r"\label{tab:unified-metrics-tablesize50}",
        r"\small",
        rf"\begin{{tabular}}{{@{{}}{col_spec}@{{}}}}",
        r"\toprule",
        rf"& & \multicolumn{{3}}{{c}}{{{GROUP_SPEC_HR}}} & \multicolumn{{3}}{{c}}{{{GROUP_SR_HR}}} & \multicolumn{{2}}{{c}}{{{GROUP_SR_EFF}}} \\",
        r"\cmidrule(lr){3-5} \cmidrule(lr){6-8} \cmidrule(lr){9-10}",
        rf"{RL_COL} & {ORDER_COL} & {AF_COL} & HR(\%) & Improv. & {AF_COL} & HR(\%) & Improv. & {AF_COL} & Eff. \\",
        r"\midrule",
        *_latex_unified_rows(data),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]
    return "\n".join(lines) + "\n"


def render_unified_pdf(path: Path, headers: list[str], data: list[list[str]], reactive_hr: float) -> None:
    del reactive_hr

    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
    })

    # --- layout constants ---
    col_widths = _unified_col_widths_normalized(headers, data)
    row_group_h = 0.140
    row_head_h  = 0.150
    row_data_h  = 0.210
    n_data = len(data)    # 9
    total_h = row_group_h + row_head_h + n_data * row_data_h
    fig_h = total_h / 0.72  # leave margins
    fig_w = _unified_figure_width(headers, data)

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    ax = fig.add_axes([0.01, 0.01, 0.98, 0.98])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # coordinate helpers
    cum_x = [0.0]
    for w in col_widths:
        cum_x.append(cum_x[-1] + w)
    total_w = cum_x[-1]
    scale_x = 1.0 / total_w

    def cx(col: int) -> float:          # left edge of column (axes coords)
        return cum_x[col] * scale_x
    def cxmid(col: int) -> float:       # center of column
        return (cum_x[col] + col_widths[col] / 2) * scale_x
    def cxr(col: int) -> float:         # right edge of column
        return cum_x[col + 1] * scale_x

    # row top edges (axes coords, top-down)
    top = 1.0
    row_tops = [
        top,
        top - row_group_h / total_h,
        top - (row_group_h + row_head_h) / total_h,
    ]
    for i in range(n_data):
        row_tops.append(row_tops[-1] - row_data_h / total_h)
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

    def cell_text(x: float, y: float, text: str, **kw) -> None:
        defaults = dict(ha="center", va="center", fontsize=10, color="black", clip_on=False)
        defaults.update(kw)
        ax.text(x, y, text, **defaults)

    x0, x1 = cx(0), cxr(9)

    # ── header row backgrounds (light grey) ───────────────────────────────────
    GREY = "#d8d8d8"
    ax.add_patch(__import__("matplotlib.patches", fromlist=["Rectangle"]).Rectangle(
        (x0, row_bottoms[0]), x1 - x0, row_tops[0] - row_bottoms[0],
        facecolor=GREY, edgecolor="none", zorder=0, clip_on=False,
    ))
    ax.add_patch(__import__("matplotlib.patches", fromlist=["Rectangle"]).Rectangle(
        (x0, row_bottoms[1]), x1 - x0, row_tops[1] - row_bottoms[1],
        facecolor=GREY, edgecolor="none", zorder=0, clip_on=False,
    ))

    # ── outer border ──────────────────────────────────────────────────────────
    hrule(row_tops[0],    x0, x1, THICK)
    hrule(row_bottoms[0], x0, x1, THICK)   # below group header
    hrule(row_bottoms[1], x0, x1, THICK)   # below col header
    hrule(row_bottoms[-1], x0, x1, THICK)  # bottom
    vrule(x0, 0.0, row_tops[0], THICK)
    vrule(x1, 0.0, row_tops[0], THICK)

    # ── row 0: group headers ──────────────────────────────────────────────────
    # cols 0-1: blank, just left border
    vrule(x0, row_bottoms[0], row_tops[0], THICK)

    groups = list(UNIFIED_GROUP_LABELS)
    for c_start, c_end, label in groups:
        gx0, gx1 = cx(c_start), cxr(c_end)
        gmid = (gx0 + gx1) / 2
        y_mid = cell_mid_y(0)
        hrule(row_tops[0],    gx0, gx1, THICK)
        hrule(row_bottoms[0], gx0, gx1, THIN)
        vrule(gx0, row_bottoms[0], row_tops[0], THICK)
        vrule(gx1, row_bottoms[0], row_tops[0], THICK)
        cell_text(gmid, y_mid, label, fontweight="bold", fontsize=10.5)

    # bold vertical dividers after col 1, col 4, col 7 (group boundaries)
    BOLD_VCOLS = {2, 5, 8}   # cx(col) positions that get THICK

    # ── row 1: column headers ─────────────────────────────────────────────────
    for col in range(10):
        lw = THICK if col in BOLD_VCOLS else THIN
        vrule(cx(col), row_bottoms[1], row_tops[1], lw)
    vrule(cxr(9), row_bottoms[1], row_tops[1], THICK)
    for col, label in enumerate(headers):
        cell_text(cxmid(col), cell_mid_y(1), label, fontweight="bold", fontsize=10)

    # ── data rows ─────────────────────────────────────────────────────────────
    algo_group = len(ORDERINGS)     # 3
    for r_idx in range(n_data):
        row = r_idx + 2
        yt, yb = row_tops[row], row_bottoms[row]
        is_group_boundary = (r_idx % algo_group == algo_group - 1) and (r_idx < n_data - 1)
        is_last = r_idx == n_data - 1
        row_lw = THICK if is_group_boundary else THIN
        # For merged col-0 cells: only draw hrule across col 0 at group boundaries
        if is_group_boundary or is_last:
            hrule(yb, x0, x1, row_lw)
        else:
            hrule(yb, cx(1), x1, row_lw)   # skip col-0 interior dividers

        # vertical lines between columns
        vrule(x0, yb, yt, THICK)
        for col in range(1, 10):
            lw = THICK if col in BOLD_VCOLS else THIN
            vrule(cx(col), yb, yt, lw)
        vrule(cxr(9), yb, yt, THICK)

        row_vals = data[r_idx]
        for col in range(10):
            if col == 0:
                g_start = (r_idx // algo_group) * algo_group
                if r_idx == g_start:     # top row of group — write algo centered over 3 rows
                    y_top3 = row_tops[g_start + 2]
                    y_bot3 = row_bottoms[g_start + 2 + 2]  # skip; use direct
                    # center of 3 rows
                    y_top3 = row_tops[g_start + 2]
                    y_bot3 = row_bottoms[g_start + 2 + 2 - 1]
                    ymid3 = (row_tops[g_start + 2] + row_bottoms[g_start + 2 + algo_group - 1]) / 2
                    cell_text(cxmid(0), ymid3, row_vals[0], fontweight="bold", fontsize=10.5)
            else:
                ha = "center" if col >= 2 else "left"
                x_pos = cxmid(col) if ha == "center" else (cx(col) + 0.008)
                cell_text(x_pos, cell_mid_y(row), row_vals[col], ha=ha)

    fig.savefig(path, bbox_inches="tight", pad_inches=0.1, facecolor="white", dpi=200)
    plt.close(fig)


def _hit_rate_table_data(rows: list[dict], reactive_hr: float) -> tuple[list[str], list[list[str]]]:
    headers = [
        "RL",
        ORDER_COL,
        "Aging Factor",
        "Hit Rate (%)",
        "Δ Reactive",
    ]
    data: list[list[str]] = []
    for row in rows:
        diff = row["value"] - reactive_hr
        data.append(
            [
                ALGORITHM_LABELS[row["algorithm"]],
                ORDERING_LABELS[row["ordering"]],
                format_af(row["aging_factor"]),
                format_hit_rate(row["value"]),
                format_diff(diff),
            ]
        )
    return headers, data


def _efficiency_table_data(rows: list[dict]) -> tuple[list[str], list[list[str]]]:
    headers = ["RL", ORDER_COL, "Aging Factor", "Speculation Efficiency"]
    data = [
        [
            ALGORITHM_LABELS[row["algorithm"]],
            ORDERING_LABELS[row["ordering"]],
            format_af(row["aging_factor"]),
            format_efficiency(row["value"]),
        ]
        for row in rows
    ]
    return headers, data


def _style_table(table: Table, headers: list[str], n_rows: int) -> None:
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.35)
    for col in range(len(headers)):
        cell = table[(0, col)]
        cell.set_facecolor("#f0f0f0")
        cell.set_text_props(weight="bold", ha="center")
        cell.set_edgecolor("#cccccc")
    for row in range(1, n_rows + 1):
        for col in range(len(headers)):
            cell = table[(row, col)]
            cell.set_edgecolor("#dddddd")
            ha = "center" if col >= 2 else "left"
            cell.get_text().set_ha(ha)
            if row % 2 == 0:
                cell.set_facecolor("#fafafa")


def _add_table_to_axes(
    ax: plt.Axes,
    title: str,
    headers: list[str],
    data: list[list[str]],
    bbox: tuple[float, float, float, float],
) -> None:
    ax.text(
        bbox[0],
        bbox[1] + bbox[3],
        title,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="bottom",
        ha="left",
    )
    table = ax.table(
        cellText=data,
        colLabels=headers,
        cellLoc="center",
        colLoc="center",
        bbox=(bbox[0], bbox[1], bbox[2], bbox[3]),
    )
    _style_table(table, headers, len(data))


def render_hit_rate_pdf(
    path: Path,
    title: str,
    rows: list[dict],
    reactive_hr: float,
) -> None:
    headers, data = _hit_rate_table_data(rows, reactive_hr)
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.axis("off")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=12)
    _add_table_to_axes(ax, "", headers, data, (0.0, 0.05, 1.0, 0.82))
    fig.savefig(path, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def render_efficiency_pdf(path: Path, title: str, rows: list[dict]) -> None:
    headers, data = _efficiency_table_data(rows)
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    ax.axis("off")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=12)
    _add_table_to_axes(ax, "", headers, data, (0.0, 0.05, 1.0, 0.82))
    fig.savefig(path, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def render_combined_pdf(
    path: Path,
    speculative_rows: list[dict],
    speculativereactive_rows: list[dict],
    efficiency_rows: list[dict],
    reactive_hr: float,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 10.5))
    ax.axis("off")
    caption = (
        f"Best configuration per algorithm and flow ordering (SFT size = {TABLESIZE}). "
        f"Reactive baseline hit rate: {format_hit_rate(reactive_hr)}%. "
        "Each row reports the aging factor that maximizes the reported metric."
    )
    ax.text(0.0, 0.995, caption, transform=ax.transAxes, fontsize=9, va="top", ha="left", wrap=True)

    sp_headers, sp_data = _hit_rate_table_data(speculative_rows, reactive_hr)
    sr_headers, sr_data = _hit_rate_table_data(speculativereactive_rows, reactive_hr)
    eff_headers, eff_data = _efficiency_table_data(efficiency_rows)

    _add_table_to_axes(ax, "(a) Speculative Hit Rate", sp_headers, sp_data, (0.0, 0.72, 1.0, 0.22))
    _add_table_to_axes(
        ax, "(b) Speculative+Reactive Hit Rate", sr_headers, sr_data, (0.0, 0.39, 1.0, 0.22)
    )
    _add_table_to_axes(
        ax,
        "(c) Speculative+Reactive Speculation Efficiency",
        eff_headers,
        eff_data,
        (0.0, 0.06, 1.0, 0.22),
    )
    fig.savefig(path, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def generate_tables(root: Path, tables_dir: Path) -> list[Path]:
    tables_dir.mkdir(parents=True, exist_ok=True)
    reactive_hr = load_reactive_hit_rate(root, TABLESIZE)

    speculative_rows = best_per_algo_ordering(
        root, "speculative", TABLESIZE, "average_hitrate_per_lti"
    )
    speculativereactive_rows = best_per_algo_ordering(
        root, "speculativereactive", TABLESIZE, "average_hitrate_per_lti"
    )
    efficiency_rows = best_per_algo_ordering(
        root, "speculativereactive", TABLESIZE, "average_speculation_efficiency_per_lti"
    )
    unified_headers, unified_data = build_unified_table_data(
        speculative_rows, speculativereactive_rows, efficiency_rows, reactive_hr
    )

    outputs = [
        (
            tables_dir / "speculative_hitrate_tablesize50.tex",
            render_hit_rate_table(
                f"Speculative Hit Rate (SFT Size = {TABLESIZE})",
                "tab:speculative-hitrate-50",
                speculative_rows,
                reactive_hr,
            ),
        ),
        (
            tables_dir / "speculativereactive_hitrate_tablesize50.tex",
            render_hit_rate_table(
                f"Speculative+Reactive Hit Rate (SFT Size = {TABLESIZE})",
                "tab:speculativereactive-hitrate-50",
                speculativereactive_rows,
                reactive_hr,
            ),
        ),
        (
            tables_dir / "speculativereactive_speculation_efficiency_tablesize50.tex",
            render_efficiency_table(
                f"Speculative+Reactive Speculation Efficiency (SFT Size = {TABLESIZE})",
                "tab:speculativereactive-speculation-efficiency-50",
                efficiency_rows,
            ),
        ),
        (
            tables_dir / "combined_tablesize50_ieee.tex",
            render_combined_ieee_table(
                speculative_rows,
                speculativereactive_rows,
                efficiency_rows,
                reactive_hr,
            ),
        ),
        (
            tables_dir / "unified_tablesize50.tex",
            render_unified_latex_table(unified_headers, unified_data, reactive_hr),
        ),
    ]

    written: list[Path] = []
    for path, content in outputs:
        path.write_text(content)
        written.append(path)

    pdf_outputs = [
        (
            tables_dir / "speculative_hitrate_tablesize50.pdf",
            lambda: render_hit_rate_pdf(
                tables_dir / "speculative_hitrate_tablesize50.pdf",
                f"Speculative Hit Rate (SFT Size = {TABLESIZE})",
                speculative_rows,
                reactive_hr,
            ),
        ),
        (
            tables_dir / "speculativereactive_hitrate_tablesize50.pdf",
            lambda: render_hit_rate_pdf(
                tables_dir / "speculativereactive_hitrate_tablesize50.pdf",
                f"Speculative+Reactive Hit Rate (SFT Size = {TABLESIZE})",
                speculativereactive_rows,
                reactive_hr,
            ),
        ),
        (
            tables_dir / "speculativereactive_speculation_efficiency_tablesize50.pdf",
            lambda: render_efficiency_pdf(
                tables_dir / "speculativereactive_speculation_efficiency_tablesize50.pdf",
                f"Speculative+Reactive Speculation Efficiency (SFT Size = {TABLESIZE})",
                efficiency_rows,
            ),
        ),
        (
            tables_dir / "combined_tablesize50_ieee.pdf",
            lambda: render_combined_pdf(
                tables_dir / "combined_tablesize50_ieee.pdf",
                speculative_rows,
                speculativereactive_rows,
                efficiency_rows,
                reactive_hr,
            ),
        ),
        (
            tables_dir / "unified_tablesize50.pdf",
            lambda: render_unified_pdf(
                tables_dir / "unified_tablesize50.pdf",
                unified_headers,
                unified_data,
                reactive_hr,
            ),
        ),
    ]
    for path, render_fn in pdf_outputs:
        render_fn()
        written.append(path)

    json_path = tables_dir / "best_agingfactor_tablesize50.json"
    best_af_json = build_best_agingfactor_json(
        speculative_rows, speculativereactive_rows, efficiency_rows, reactive_hr
    )
    json_path.write_text(json.dumps(best_af_json, indent=4) + "\n")
    written.append(json_path)

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--tables-dir", type=Path, default=TABLES_DIR)
    args = parser.parse_args()

    root = args.root.resolve()
    tables_dir = args.tables_dir.resolve()
    reactive_hr = load_reactive_hit_rate(root, TABLESIZE)
    print(f"reactive hit rate (tablesize {TABLESIZE}): {reactive_hr:.2f}%")

    for path in generate_tables(root, tables_dir):
        print(f"saved {path}")


if __name__ == "__main__":
    main()
