"""
Compile per-judge JSON metrics from step 3 into a colour-coded HTML file.
Reads benchmarks/RM-Bench-train/4-results/*.json and writes
benchmarks/RM-Bench-train/overall.html.

Produces two tables:
  - Domain average accuracy: chat, math, code, safety, total_avg_acc
  - Style matrix: hard_acc, normal_acc, easy_acc

Each data column is coloured independently, centred on the column median:
green = column maximum, red = column minimum.

Example (from the 08a-judge-evaluation repo root):

    python -m benchmarks.RM-Bench-train.src.4-compile-results-html
"""

import glob
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.utils import JUDGE_MAPPING

DOMAINS = ("chat", "math", "code", "safety")
DOMAIN_COLS = ["Judge #", "Description", "chat", "math", "code", "safety", "total_avg_acc"]
STYLE_COLS = ["Judge #", "Description", "hard_acc", "normal_acc", "easy_acc"]


def _row_from_json(filepath: str, judge_number: str) -> Dict[str, Any]:
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    row: Dict[str, Any] = {
        "Judge #": judge_number,
        "Description": JUDGE_MAPPING.get(judge_number, "—"),
    }
    for domain in DOMAINS:
        raw = data.get(domain)
        if isinstance(raw, dict):
            row[domain] = raw.get("acc")
            nr = raw.get("none_rate")
            # none_rate stored as 0-100 in RM-Bench JSONs
            row[f"{domain}_none_rate"] = nr if nr is not None else float("nan")
        elif raw is not None:
            row[domain] = float(raw)
            row[f"{domain}_none_rate"] = float("nan")

    for k, v in data.items():
        if k in DOMAINS:
            continue
        if not isinstance(v, dict):
            row[k] = v

    return row


def _score_to_colour(value: float, col_min: float, col_median: float, col_max: float) -> str:
    if value <= col_median:
        denom = col_median - col_min
        s = (value - col_min) / denom if denom > 0 else 1.0
        r = int(224 + (255 - 224) * s)
        g = int(85 + (255 - 85) * s)
        b = int(85 + (255 - 85) * s)
    else:
        denom = col_max - col_median
        s = (value - col_median) / denom if denom > 0 else 1.0
        r = int(255 + (76 - 255) * s)
        g = int(255 + (175 - 255) * s)
        b = int(255 + (115 - 255) * s)
    return f"#{r:02x}{g:02x}{b:02x}"


def _build_table(
    df: pd.DataFrame,
    col_order: List[str],
    meta_cols: List[str],
    none_rate_cols: Optional[Dict[str, str]] = None,
) -> str:
    """
    Render one HTML <table>.
    none_rate_cols: mapping from data column name → DataFrame column holding its none_rate (0-100).
    """
    cols_present = [c for c in col_order if c in df.columns]
    data_cols = [c for c in cols_present if c not in meta_cols]

    col_stats: Dict[str, Tuple[float, float, float]] = {}
    for col in data_cols:
        vals = df[col].dropna().astype(float)
        if len(vals):
            col_stats[col] = (vals.min(), vals.median(), vals.max())

    rows_html = []
    for _, row in df[cols_present].iterrows():
        cells = [f"<td>{row[c]}</td>" for c in meta_cols if c in cols_present]
        for col in data_cols:
            score = row[col]
            if pd.isna(score):
                cells.append("<td></td>")
                continue
            score_f = float(score)
            col_min, col_median, col_max = col_stats.get(col, (score_f, score_f, score_f))
            bg = _score_to_colour(score_f, col_min, col_median, col_max)
            is_best = abs(score_f - col_max) <= 1e-9

            none_rate_pct = 0.0
            if none_rate_cols and col in none_rate_cols:
                nr_col = none_rate_cols[col]
                if nr_col in df.columns:
                    nr_val = df.loc[row.name, nr_col] if row.name in df.index else float("nan")
                    if not pd.isna(nr_val):
                        none_rate_pct = float(nr_val)

            text = f"{score_f:.4f}"
            sup_html = f"<sup>{none_rate_pct:.1f}%</sup>" if none_rate_pct > 0 else ""
            cells.append(
                f'<td class="num{"  best" if is_best else ""}" style="background:{bg}">'
                f"{text}{sup_html}</td>"
            )
        rows_html.append("<tr>" + "".join(cells) + "</tr>")

    header_cells = "".join(
        f'<th class="num">{c}</th>' if c not in meta_cols else f"<th>{c}</th>"
        for c in cols_present
    )

    return (
        f"<table>\n<thead><tr>{header_cells}</tr></thead>\n"
        f"<tbody>{''.join(rows_html)}</tbody>\n</table>"
    )


if __name__ == "__main__":
    results_root = "benchmarks/RM-Bench-train/4-results"
    rows: List[Dict[str, Any]] = []

    for filepath in sorted(glob.glob(os.path.join(results_root, "*.json"))):
        judge_number = os.path.splitext(os.path.basename(filepath))[0]
        rows.append(_row_from_json(filepath, judge_number))

    if not rows:
        print(f"No JSON files under {results_root}")
        raise SystemExit(1)

    df = pd.DataFrame(rows)
    if "Judge #" in df.columns:
        df = df.sort_values("Judge #").reset_index(drop=True)

    none_rate_map = {d: f"{d}_none_rate" for d in DOMAINS}

    domain_table = _build_table(df, DOMAIN_COLS, ["Judge #", "Description"], none_rate_map)
    style_table = _build_table(df, STYLE_COLS, ["Judge #", "Description"], none_rate_cols=None)

    css = """
    body { font-family: system-ui, sans-serif; font-size: 13px; padding: 1em 2em; color: #000; }
    h3 { margin-bottom: 0.4em; }
    table { border-collapse: collapse; white-space: nowrap; margin-bottom: 2em; }
    th { background: #2d3748; color: #edf2f7; padding: 6px 10px; text-align: left; }
    th.num { text-align: right; }
    td { padding: 5px 10px; border: 1px solid #e2e8f0; }
    td.num { text-align: right; font-variant-numeric: tabular-nums; }
    td.best { font-weight: 700; }
    sup { font-size: 9px; color: #000; margin-left: 2px; }
    """

    legend = """<p style="font-size:11px;color:#000">
  Colour scale is per-column, centred on the column median:
  <span style="background:#4caf73;padding:1px 6px;border-radius:3px">green = column max</span>
  &nbsp;<span style="background:#ffffff;border:1px solid #ccc;padding:1px 6px;border-radius:3px">white = column median</span>
  &nbsp;<span style="background:#e05555;padding:1px 6px;border-radius:3px">red = column min</span>.
  Superscript = none-rate %.  <strong>Bold</strong> = column best.
</p>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>RM-Bench results</title>
<style>{css}</style>
</head>
<body>
{legend}
<h3>RM-Bench — domain average accuracy</h3>
{domain_table}
<h3>RM-Bench — style matrix (hard / normal / easy)</h3>
{style_table}
</body>
</html>
"""

    out_path = "benchmarks/RM-Bench-train/overall.html"
    with open(out_path, "w", encoding="utf-8") as f_out:
        f_out.write(html)

    print(f"Wrote {out_path}")
