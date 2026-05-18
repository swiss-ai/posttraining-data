#!/usr/bin/env python3
"""Convert UCSC-VLAA/MedReason to parquet with messages column."""

import json
import pandas as pd
from pathlib import Path

CACHE = Path("/capstor/store/cscs/swissai/infra01/medical_text/hf_hub_cache")
SRC = CACHE / "datasets--UCSC-VLAA--MedReason/snapshots/7fc0ddb80653c839be359ac4496ec6ee00bdd52f/ours_quality_33000.jsonl"
DST = Path("/capstor/store/cscs/swissai/infra01/medical_text/pre-processed/medreason.parquet")


def convert():
    rows = []
    with open(SRC) as f:
        for line in f:
            rec = json.loads(line)
            question = rec.get("question", "")
            options = rec.get("options", "")
            user_msg = question
            if options:
                user_msg += f"\n\n{options}"

            rows.append({
                "messages": [
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": rec.get("answer", "")},
                ],
                "source_dataset": "UCSC-VLAA/MedReason",
                "reasoning": rec.get("reasoning", ""),
                "dataset_name": rec.get("dataset_name", ""),
                "id_in_dataset": rec.get("id_in_dataset", ""),
            })

    df = pd.DataFrame(rows)
    DST.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DST, index=False)
    print(f"Saved {len(df)} rows to {DST}")


if __name__ == "__main__":
    convert()
