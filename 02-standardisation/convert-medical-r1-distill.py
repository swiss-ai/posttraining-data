#!/usr/bin/env python3
"""Convert FreedomIntelligence/Medical-R1-Distill-Data to parquet with messages column."""

import json
import pandas as pd
from pathlib import Path

CACHE = Path("/capstor/store/cscs/swissai/infra01/medical_text/hf_hub_cache")
SRC = CACHE / "datasets--FreedomIntelligence--Medical-R1-Distill-Data/snapshots/3491deecebe1973a2d7370b824f4b41be29dcf1a/medical_r1_distill_sft.json"
DST = Path("/capstor/store/cscs/swissai/infra01/medical_text/pre-processed/medical-r1-distill-data.parquet")


def convert():
    with open(SRC) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} records")

    rows = []
    for rec in data:
        rows.append({
            "messages": [
                {"role": "user", "content": rec.get("question", "")},
                {"role": "assistant", "content": rec.get("response (content)", rec.get("content", rec.get("response", "")))},
            ],
            "source_dataset": "FreedomIntelligence/Medical-R1-Distill-Data",
            "reasoning": rec.get("reasoning (reasoning_content)", rec.get("reasoning_content", rec.get("reasoning", ""))),
        })

    df = pd.DataFrame(rows)
    DST.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DST, index=False)
    print(f"Saved {len(df)} rows to {DST}")


if __name__ == "__main__":
    convert()
