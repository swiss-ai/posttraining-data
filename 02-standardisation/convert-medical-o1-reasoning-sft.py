#!/usr/bin/env python3
"""Convert FreedomIntelligence/medical-o1-reasoning-SFT to parquet with messages column."""

import json
import pandas as pd
from pathlib import Path

CACHE = Path("/capstor/store/cscs/swissai/infra01/medical_text/hf_hub_cache")
SNAP = CACHE / "datasets--FreedomIntelligence--medical-o1-reasoning-SFT/snapshots/fc2c9e8a37b38f38da6d449564a8c350b244aef4"
DST = Path("/capstor/store/cscs/swissai/infra01/medical_text/pre-processed/medical-o1-reasoning-sft.parquet")

FILES = {
    "medical_o1_sft.json": "EN",
    "medical_o1_sft_mix.json": "EN_mix",
    "medical_o1_sft_Chinese.json": "CN",
    "medical_o1_sft_mix_Chinese.json": "CN_mix",
}


def convert():
    rows = []
    for fname, subset in FILES.items():
        src = SNAP / fname
        if not src.exists():
            print(f"Skipping missing file: {fname}")
            continue
        print(f"Reading {fname}...")
        with open(src) as f:
            data = json.load(f)
        print(f"  {len(data)} records")
        for rec in data:
            rows.append({
                "messages": [
                    {"role": "user", "content": rec.get("Question", "")},
                    {"role": "assistant", "content": rec.get("Response", "")},
                ],
                "source_dataset": "FreedomIntelligence/medical-o1-reasoning-SFT",
                "reasoning": rec.get("Complex_CoT", ""),
                "subset": subset,
            })

    df = pd.DataFrame(rows)
    DST.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DST, index=False)
    print(f"Saved {len(df)} rows to {DST}")


if __name__ == "__main__":
    convert()
