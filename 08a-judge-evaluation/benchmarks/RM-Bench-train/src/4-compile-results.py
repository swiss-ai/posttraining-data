"""
Compile per-judge JSON metrics from step 3 into markdown tables.
Reads ``benchmarks/RM-Bench-train/4-results/*.json`` and writes
``benchmarks/RM-Bench-train/4-results/overall.md``.

Example (from the ``08a-judge-evaluation`` repo root)::

    python -m benchmarks.RM-Bench-train.src.4-compile-results
"""

import argparse
import glob
import json
import os
from typing import Any, Dict, List

import pandas as pd

from src.utils import JUDGE_MAPPING

DOMAINS = ("chat", "math", "code", "safety")


def _row_from_metrics_json(filepath: str, judge_number: str) -> Dict[str, Any]:
    """Flatten nested per-domain metrics (acc + percentage_skipped) into dataframe columns."""
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
            pct = raw.get("percentage_comparisons_skipped")
            row[f"{domain}_skipped_pct"] = pct if pct is not None else float("nan")
        elif raw is not None:
            row[domain] = raw
            row[f"{domain}_skipped_pct"] = float("nan")

    for k, v in data.items():
        if k in DOMAINS:
            continue
        row[k] = v

    return row


def _bold_best_per_numeric_column(df: pd.DataFrame, best_type: str = "max") -> pd.DataFrame:
    display_df = df.copy()
    numeric_cols = display_df.select_dtypes(include=["number"]).columns

    for col in numeric_cols:
        s = display_df[col]
        if s.isna().all():
            continue

        if best_type == "min":
            best_val = s.min()
        else:
            best_val = s.max()

        def fmt(x: object) -> str:
            if pd.isna(x):
                return ""
            return f"{float(x):.2f}"

        def bold_if_best(x: object) -> str:
            if pd.isna(x):
                return ""
            if abs(float(x) - float(best_val)) <= 1e-9:
                return f"**{fmt(x)}**"
            return fmt(x)

        display_df[col] = display_df[col].apply(bold_if_best)

    return display_df


def _append_skipped_pct_to_domains(
    display_df: pd.DataFrame, source_df: pd.DataFrame
) -> pd.DataFrame:
    """After bolding scores, suffix each domain cell with skipped % when present."""
    out = display_df.copy()
    for domain in DOMAINS:
        if domain not in out.columns:
            continue
        sk_col = f"{domain}_skipped_pct"
        if sk_col not in source_df.columns:
            continue
        for i in range(len(out)):
            acc_str = out.at[i, domain]
            skip_val = source_df.at[i, sk_col]
            if acc_str != "" and not pd.isna(skip_val) and skip_val > 0:
                out.at[i, domain] = f"{acc_str} ({float(skip_val):.1f}% skipped)"
    return out


def _judge_eval_root() -> str:
    p = os.path.abspath(__file__)
    for _ in range(4):
        p = os.path.dirname(p)
    return p


def _default_results_dir() -> str:
    return os.path.join(_judge_eval_root(), "benchmarks", "RM-Bench-train", "4-results")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=str,
        default=_default_results_dir(),
        help="Directory containing one <judge>.json per run (default: …/4-results).",
    )
    args = parser.parse_args()
    folder = args.results_dir

    rows: List[Dict[str, Any]] = []
    for filepath in sorted(glob.glob(os.path.join(folder, "*.json"))):
        judge_number = os.path.splitext(os.path.basename(filepath))[0]
        rows.append(_row_from_metrics_json(filepath, judge_number))

    if not rows:
        print(f"No JSON files under {folder}")
        raise SystemExit(1)

    df = pd.DataFrame(rows)
    df = df.round(2)
    if "Judge #" in df.columns:
        df = df.sort_values("Judge #")

    out_path = "benchmarks/RM-Bench-train/overall.md"
    # Domain-level average accuracy (per-domain mean over the 3×3 score matrix), plus RM-Bench aggregate.
    domain_cols = ["Judge #", "Description", "chat", "math", "code", "safety", "total_avg_acc"]
    # hard / normal / easy: sub-aggregates of the style comparison matrix (see RM-Bench README).
    style_cols = ["Judge #", "Description", "hard_acc", "normal_acc", "easy_acc"]

    domain_use = [c for c in domain_cols if c in df.columns]
    style_use = [c for c in style_cols if c in df.columns]

    with open(out_path, "w", encoding="utf-8") as f_out:
        f_out.write("### RM-Bench — domain average accuracy\n\n")
        d1 = _bold_best_per_numeric_column(df[domain_use].copy(), best_type="max")
        d1 = _append_skipped_pct_to_domains(d1.reset_index(drop=True), df.reset_index(drop=True))
        print(d1.to_markdown(index=False), file=f_out)
        f_out.write("\n\n")

        f_out.write("### RM-Bench — style matrix (hard / normal / easy)\n\n")
        d2 = _bold_best_per_numeric_column(df[style_use].copy(), best_type="max")
        print(d2.to_markdown(index=False), file=f_out)
        f_out.write("\n")

    print(f"Wrote {out_path}")
