#!/usr/bin/env python3
"""Convert FreedomIntelligence/medical-o1-verifiable-problem to parquet with messages column."""

import json
import pandas as pd
from pathlib import Path

CACHE = Path("/capstor/store/cscs/swissai/infra01/medical_text/hf_hub_cache")
SRC = CACHE / "datasets--FreedomIntelligence--medical-o1-verifiable-problem/snapshots/46d5175eb74fdef3516d51d52e8c40db04bbdf35/medical_o1_verifiable_problem.json"
DST = Path("/capstor/store/cscs/swissai/infra01/medical_text/pre-processed/medical-o1-verifiable-problem.parquet")


def convert():
    with open(SRC) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} records")

    rows = []
    for rec in data:
        rows.append({
            "messages": [
                {"role": "user", "content": rec.get("Open-ended Verifiable Question", "")},
                {"role": "assistant", "content": rec.get("Ground-True Answer", "")},
            ],
            "source_dataset": "FreedomIntelligence/medical-o1-verifiable-problem",
        })

    df = pd.DataFrame(rows)
    DST.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DST, index=False)
    print(f"Saved {len(df)} rows to {DST}")


if __name__ == "__main__":
    convert()
