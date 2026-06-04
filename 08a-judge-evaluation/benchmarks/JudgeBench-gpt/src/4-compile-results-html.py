"""
Compile per-judge JSON metrics from step 3 into a colour-coded HTML table.
Reads benchmarks/JudgeBench-gpt/4-results/*.json and writes
benchmarks/JudgeBench-gpt/overall.html.

Each data column is coloured independently, centred on the column median:
green = column maximum, red = column minimum.

Example (from the 08a-judge-evaluation repo root):

    python -m benchmarks.JudgeBench-gpt.src.4-compile-results-html
"""

import glob
import json
import os
from typing import Any, Dict, List, Tuple

import pandas as pd


SCORE_COLS = ["mmlu-pro", "livebench-reasoning", "livebench-math", "livecodebench", "overall"]


def _flatten_metrics(data: Dict[str, Any]) -> Tuple[Dict[str, float], Dict[str, float]]:
    scores: Dict[str, float] = {}
    none_rates: Dict[str, float] = {}
    for key, val in data.items():
        if isinstance(val, dict):
            if "score" in val:
                scores[key] = val["score"]
            if "none_rate" in val:
                none_rates[key] = val["none_rate"]
        else:
            scores[key] = float(val)
    return scores, none_rates


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


def _build_html(
    df_scores: pd.DataFrame,
    df_none: pd.DataFrame,
    meta_cols: List[str],
) -> str:
    data_cols = [c for c in df_scores.columns if c not in meta_cols]

    col_stats: Dict[str, Tuple[float, float, float]] = {}
    for col in data_cols:
        vals = df_scores[col].dropna().astype(float)
        col_stats[col] = (vals.min(), vals.median(), vals.max())

    css = """
    body { font-family: system-ui, sans-serif; font-size: 13px; padding: 1em 2em; color: #000; }
    h3 { margin-bottom: 0.4em; }
    table { border-collapse: collapse; white-space: nowrap; }
    th { background: #2d3748; color: #edf2f7; padding: 6px 10px; text-align: left; }
    th.num { text-align: right; }
    td { padding: 5px 10px; border: 1px solid #e2e8f0; }
    td.num { text-align: right; font-variant-numeric: tabular-nums; }
    td.best { font-weight: 700; }
    sup { font-size: 9px; color: #000; margin-left: 2px; }
    """

    rows_html = []
    for _, row in df_scores.iterrows():
        judge = row["Judge"]
        none_row = df_none[df_none["Judge"] == judge].iloc[0] if judge in df_none["Judge"].values else None

        cells = [f"<td>{judge}</td>"]
        for col in data_cols:
            score = row[col]
            if pd.isna(score):
                cells.append("<td></td>")
                continue
            score_f = float(score)
            col_min, col_median, col_max = col_stats[col]
            bg = _score_to_colour(score_f, col_min, col_median, col_max)

            none_rate_pct = 0.0
            if none_row is not None and col in none_row.index and not pd.isna(none_row[col]):
                # none_rate stored as 0-1 fraction in JudgeBench JSONs
                none_rate_pct = float(none_row[col]) * 100

            text = f"{score_f:.2f}"
            is_best = abs(score_f - col_max) <= 1e-9
            sup_html = f"<sup>{none_rate_pct:.0f}%</sup>" if none_rate_pct > 0 else ""
            cells.append(
                f'<td class="num{"  best" if is_best else ""}" style="background:{bg}">'
                f"{text}{sup_html}</td>"
            )
        rows_html.append("<tr>" + "".join(cells) + "</tr>")

    header_cells = "".join(
        f'<th class="num">{c}</th>' if c not in meta_cols else f"<th>{c}</th>"
        for c in meta_cols + data_cols
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>JudgeBench results</title>
<style>{css}</style>
</head>
<body>
<h3>JudgeBench — accuracy (higher is better)</h3>
<p style="font-size:11px;color:#000">
  Colour scale is per-column, centred on the column median:
  <span style="background:#4caf73;padding:1px 6px;border-radius:3px">green = column max</span>
  &nbsp;<span style="background:#ffffff;border:1px solid #ccc;padding:1px 6px;border-radius:3px">white = column median</span>
  &nbsp;<span style="background:#e05555;padding:1px 6px;border-radius:3px">red = column min</span>.
  Superscript = none-rate %.  <strong>Bold</strong> = column best.
</p>
<table>
<thead><tr>{header_cells}</tr></thead>
<tbody>
{"".join(rows_html)}
</tbody>
</table>
</body>
</html>
"""


if __name__ == "__main__":
    folder = "benchmarks/JudgeBench-gpt/4-results"
    meta_cols = ["Judge"]
    score_rows: List[Dict[str, Any]] = []
    none_rate_rows: List[Dict[str, Any]] = []

    for filepath in sorted(glob.glob(os.path.join(folder, "*.json"))):
        judge_name = os.path.splitext(os.path.basename(filepath))[0]
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        meta = {"Judge": judge_name}
        scores, none_rates = _flatten_metrics(data)
        score_rows.append({**meta, **scores})
        none_rate_rows.append({**meta, **none_rates})

    if not score_rows:
        print(f"No JSON files under {folder}")
        raise SystemExit(1)

    def _build_df(rows: List[Dict[str, Any]]) -> pd.DataFrame:
        df = pd.DataFrame(rows).round(4)
        if "Judge" in df.columns:
            df = df.sort_values("Judge")
        data_cols = [c for c in df.columns if c not in meta_cols]
        ordered = [c for c in data_cols if c != "overall"] + (["overall"] if "overall" in data_cols else [])
        return df[meta_cols + ordered]

    df_scores = _build_df(score_rows)
    df_none = _build_df(none_rate_rows)

    out_path = "benchmarks/JudgeBench-gpt/overall.html"
    html = _build_html(df_scores, df_none, meta_cols)
    with open(out_path, "w", encoding="utf-8") as f_out:
        f_out.write(html)

    print(f"Wrote {out_path}")