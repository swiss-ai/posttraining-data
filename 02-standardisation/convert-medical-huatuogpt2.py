#!/usr/bin/env python3
"""Convert FreedomIntelligence/HuatuoGPT2-SFT-GPT4-140K to parquet with messages column."""

import json
import pandas as pd
from pathlib import Path

CACHE = Path("/capstor/store/cscs/swissai/infra01/medical_text/hf_hub_cache")
SRC = CACHE / "datasets--FreedomIntelligence--HuatuoGPT2-SFT-GPT4-140K/snapshots/4077ffaeb123e49b8b8a0283f42957a5570a52ce/HuatuoGPT2-GPT4-SFT-140K.json"
DST = Path("/capstor/store/cscs/swissai/infra01/medical_text/pre-processed/huatuogpt2.parquet")

ROLE_MAP = {"human": "user", "gpt": "assistant"}


def convert():
    with open(SRC) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} records")

    rows = []
    for rec in data:
        messages = []
        for turn in rec.get("conversations", []):
            role = ROLE_MAP.get(turn["from"], turn["from"].lower())
            messages.append({"role": role, "content": turn["value"]})
        rows.append({
            "messages": messages,
            "source_dataset": "FreedomIntelligence/HuatuoGPT2-SFT-GPT4-140K",
            "id": rec.get("id", ""),
        })

    df = pd.DataFrame(rows)
    DST.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DST, index=False)
    print(f"Saved {len(df)} rows to {DST}")


if __name__ == "__main__":
    convert()
