#!/usr/bin/env python3
"""Convert nvidia/HelpSteer2 to parquet with messages column."""

import gzip
import json
import pandas as pd
from pathlib import Path

CACHE = Path("/capstor/store/cscs/swissai/infra01/medical_text/hf_hub_cache")
SNAP = CACHE / "datasets--nvidia--HelpSteer2/snapshots/990b2711a36180dd19d9c94b8627844866f8982a"
DST = Path("/capstor/store/cscs/swissai/infra01/medical_text/pre-processed/helpsteer2.parquet")

SPLITS = ["train.jsonl.gz"]


def convert():
    rows = []
    for split_name in SPLITS:
        src = SNAP / split_name
        if not src.exists():
            print(f"Skipping missing file: {src}")
            continue
        print(f"Reading {src.name}...")
        with gzip.open(src, "rt") as f:
            for line in f:
                rec = json.loads(line)
                rows.append({
                    "messages": [
                        {"role": "user", "content": rec["prompt"]},
                        {"role": "assistant", "content": rec["response"]},
                    ],
                    "source_dataset": "nvidia/HelpSteer2",
                    "helpfulness": rec.get("helpfulness"),
                    "correctness": rec.get("correctness"),
                    "coherence": rec.get("coherence"),
                    "complexity": rec.get("complexity"),
                    "verbosity": rec.get("verbosity"),
                })

    df = pd.DataFrame(rows)
    DST.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DST, index=False)
    print(f"Saved {len(df)} rows to {DST}")


if __name__ == "__main__":
    convert()
