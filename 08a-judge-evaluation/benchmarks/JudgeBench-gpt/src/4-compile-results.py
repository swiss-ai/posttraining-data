"""
Example run command: 
python -m benchmarks.JudgeBench-gpt.src.4-compile-results
"""

import os
import glob
import json
import pandas as pd

from src.utils import JUDGE_MAPPING


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
            # values are expected to be already rounded to 2 decimals
            return f"{float(x):.2f}"

        def bold_if_best(x: object) -> str:
            if pd.isna(x):
                return ""
            # float-safe equality after rounding
            if abs(float(x) - float(best_val)) <= 1e-9:
                return f"**{fmt(x)}**"
            return fmt(x)

        display_df[col] = display_df[col].apply(bold_if_best)

    return display_df


if __name__ == "__main__":
    folder = "benchmarks/JudgeBench-gpt/4-results"

    rows = []

    filepaths = sorted(glob.glob(os.path.join(folder, "*.json")))
    for filepath in filepaths:
        with open(filepath, "r") as f:
            data = json.load(f)

        # Add filename as first column
        judge_number = os.path.splitext(os.path.basename(filepath))[0]
        row = {"Judge #": judge_number, "Description": JUDGE_MAPPING[judge_number]}
        row.update(data)
        rows.append(row)

    # Create DataFrame
    df = pd.DataFrame(rows)
    df = df.round(2)

    # Also print a markdown table with the best value per numeric column bolded.
    with open(os.path.join(folder, "overall.md"), "w") as f_out:
        display_df = _bold_best_per_numeric_column(df[["Judge #", "Description", "mmlu-pro", "livebench-reasoning", "livebench-math", "livecodebench"]], best_type="max")
        print(display_df.to_markdown(index=False), file=f_out)
        print("", file=f_out)
        display_df = _bold_best_per_numeric_column(df[["Judge #", "Description", "two-None rate", "one-None rate", "tie rate", "tie rate (among zero-None)"]], best_type="min")
        print(display_df.to_markdown(index=False), file=f_out)