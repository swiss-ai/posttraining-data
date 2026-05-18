#!/usr/bin/env python3
"""Convert nvidia/Daring-Anteater to parquet with messages column."""

import json
import pandas as pd
from pathlib import Path

CACHE = Path("/capstor/store/cscs/swissai/infra01/medical_text/hf_hub_cache")
SRC = CACHE / "datasets--nvidia--Daring-Anteater/snapshots/ae79f8ac44cf185fbd3250dbe46057a7f7c4ec40/train.jsonl"
DST = Path("/capstor/store/cscs/swissai/infra01/medical_text/pre-processed/daring-anteater.parquet")

ROLE_MAP = {"User": "user", "Assistant": "assistant"}


def convert():
    rows = []
    with open(SRC) as f:
        for line in f:
            rec = json.loads(line)
            messages = []
            system = (rec.get("system") or "").strip()
            if system:
                messages.append({"role": "system", "content": system})
            for turn in rec.get("conversations", []):
                role = ROLE_MAP.get(turn["from"], turn["from"].lower())
                messages.append({"role": role, "content": turn["value"]})
            rows.append({
                "messages": messages,
                "source_dataset": "nvidia/Daring-Anteater",
                "dataset": rec.get("dataset", ""),
            })

    df = pd.DataFrame(rows)
    DST.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DST, index=False)
    print(f"Saved {len(df)} rows to {DST}")


if __name__ == "__main__":
    convert()
