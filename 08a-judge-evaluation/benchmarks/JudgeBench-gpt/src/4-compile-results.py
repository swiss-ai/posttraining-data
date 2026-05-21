"""
Compile per-judge JSON metrics from step 3 into a markdown table.
Reads benchmarks/JudgeBench-gpt/4-results/*.json and writes
benchmarks/JudgeBench-gpt/overall.md.

Example (from the 08a-judge-evaluation repo root):

    python -m benchmarks.JudgeBench-gpt.src.4-compile-results
"""

import glob
import json
import os
from typing import Any, Dict, List

import pandas as pd

from src.utils import JUDGE_MAPPING

SCORE_COLS = ["mmlu-pro", "livebench-reasoning", "livebench-math", "livecodebench", "overall"]


def _flatten_metrics(data: Dict[str, Any]) -> tuple[Dict[str, float], Dict[str, float]]:
    """Split nested {subset: {score, none_rate}} into two flat dicts."""
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


def _build_display(
    df_scores: pd.DataFrame, df_none: pd.DataFrame, meta_cols: List[str]
) -> pd.DataFrame:
    """
    Produce a string DataFrame where each score cell reads:
      "75.14"                  when none_rate == 0
      "75.14 (5% unscored)"    when none_rate > 0
    The best score per column is bolded.
    """
    display = df_scores[meta_cols].copy()
    for col in SCORE_COLS:
        if col not in df_scores.columns:
            continue
        scores = df_scores[col]
        nones = df_none[col] if col in df_none.columns else pd.Series([0.0] * len(scores))
        best_val = scores.max()

        def _cell(score: object, none_rate: object) -> str:
            if pd.isna(score):
                return ""
            s = f"{float(score):.2f}"
            nr = float(none_rate) if not pd.isna(none_rate) else 0.0
            if nr > 0:
                s += f" ({nr * 100:.0f}% random scored)"
            if not pd.isna(best_val) and abs(float(score) - float(best_val)) <= 1e-9:
                s = f"**{s}**"
            return s

        display[col] = [_cell(sc, nr) for sc, nr in zip(scores, nones)]

    return display


if __name__ == "__main__":
    folder = "benchmarks/JudgeBench-gpt/4-results"

    meta_cols = ["Judge #", "Description"]
    score_rows: List[Dict[str, Any]] = []
    none_rate_rows: List[Dict[str, Any]] = []

    for filepath in sorted(glob.glob(os.path.join(folder, "*.json"))):
        judge_number = os.path.splitext(os.path.basename(filepath))[0]
        with open(filepath, "r") as f:
            data = json.load(f)
        meta = {"Judge #": judge_number, "Description": JUDGE_MAPPING.get(judge_number, "—")}
        scores, none_rates = _flatten_metrics(data)
        score_rows.append({**meta, **scores})
        none_rate_rows.append({**meta, **none_rates})

    if not score_rows:
        print(f"No JSON files under {folder}")
        raise SystemExit(1)

    def _build_df(rows: List[Dict[str, Any]]) -> pd.DataFrame:
        df = pd.DataFrame(rows).round(4)
        if "Judge #" in df.columns:
            df = df.sort_values("Judge #")
        data_cols = [c for c in df.columns if c not in meta_cols]
        ordered = [c for c in data_cols if c != "overall"] + (["overall"] if "overall" in data_cols else [])
        return df[meta_cols + ordered]

    df_scores = _build_df(score_rows)
    df_none = _build_df(none_rate_rows)

    out_path = "benchmarks/JudgeBench-gpt/overall.md"
    with open(out_path, "w") as f_out:
        f_out.write("### JudgeBench — accuracy (higher is better; bracketed % = none rate)\n\n")
        display = _build_display(df_scores, df_none, meta_cols)
        print(display.to_markdown(index=False), file=f_out)
        f_out.write("\n")

    print(f"Wrote {out_path}")
