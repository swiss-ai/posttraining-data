#!/usr/bin/env python3
"""Convert openlifescienceai/medmcqa to parquet with messages column."""

import pandas as pd
from pathlib import Path

CACHE = Path("/capstor/store/cscs/swissai/infra01/medical_text/hf_hub_cache")
SNAP = CACHE / "datasets--openlifescienceai--medmcqa/snapshots/91c6572c454088bf71b679ad90aa8dffcd0d5868/data"
DST = Path("/capstor/store/cscs/swissai/infra01/medical_text/pre-processed/medmcqa.parquet")

OPTION_LETTERS = ["A", "B", "C", "D"]


def format_options(row):
    opts = [row["opa"], row["opb"], row["opc"], row["opd"]]
    return "\n".join(f"{OPTION_LETTERS[i]}. {o}" for i, o in enumerate(opts))


def convert():
    splits = sorted(SNAP.glob("train-*.parquet"))
    print(f"Found {len(splits)} train split files: {[f.name for f in splits]}")
    df = pd.concat([pd.read_parquet(f) for f in splits], ignore_index=True)
    print(f"Loaded {len(df)} rows (train only)")

    cop_to_letter = {0: "A", 1: "B", 2: "C", 3: "D"}

    rows = []
    for _, r in df.iterrows():
        user_msg = f"{r['question']}\n\n{format_options(r)}"
        letter = cop_to_letter.get(r["cop"], "?")
        exp = r.get("exp") or ""
        assistant_msg = f"{letter}. {[r['opa'], r['opb'], r['opc'], r['opd']][r['cop']] if r['cop'] in cop_to_letter else '?'}"
        if exp:
            assistant_msg += f"\n\n{exp}"
        rows.append({
            "messages": [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg},
            ],
            "source_dataset": "openlifescienceai/medmcqa",
            "subject_name": r.get("subject_name", ""),
            "topic_name": r.get("topic_name", ""),
            "choice_type": r.get("choice_type", ""),
        })

    out = pd.DataFrame(rows)
    DST.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(DST, index=False)
    print(f"Saved {len(out)} rows to {DST}")


if __name__ == "__main__":
    convert()
