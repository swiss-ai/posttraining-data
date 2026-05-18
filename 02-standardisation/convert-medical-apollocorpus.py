#!/usr/bin/env python3
"""Convert FreedomIntelligence/ApolloCorpus to parquet with messages column.

Iterates over all JSON files in pretrain/ and sft/ directories.
Only includes QA-style files (lists of [question, answer] pairs), skips
plain-text files (lists of strings used for pretraining).
"""

import json
import pandas as pd
from pathlib import Path

CACHE = Path("/capstor/store/cscs/swissai/infra01/medical_text/hf_hub_cache")
SNAP = CACHE / "datasets--FreedomIntelligence--ApolloCorpus/snapshots/c1ee58d62a26a2422d01749517435fb0d716d449"
DST = Path("/capstor/store/cscs/swissai/infra01/medical_text/pre-processed/apollocorpus.parquet")


def convert():
    rows = []
    json_files = sorted(SNAP.rglob("*.json"))
    print(f"Found {len(json_files)} JSON files")

    for jf in json_files:
        subset = jf.stem
        print(f"  Processing {jf.relative_to(SNAP)} ...", end=" ")
        with open(jf) as f:
            data = json.load(f)

        if not data:
            print("empty, skipping")
            continue

        # Skip plain-text files (list of strings for pretraining)
        if isinstance(data[0], str):
            print(f"text-only ({len(data)} strings), skipping")
            continue

        # QA-style: list of [question, answer] pairs
        count = 0
        for item in data:
            if isinstance(item, list) and len(item) >= 2:
                q, a = str(item[0]).strip(), str(item[1]).strip()
            elif isinstance(item, dict):
                q = str(item.get("0", item.get("instruction", ""))).strip()
                a = str(item.get("1", item.get("output", ""))).strip()
            else:
                continue
            if not q or not a:
                continue
            rows.append({
                "messages": [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": a},
                ],
                "source_dataset": "FreedomIntelligence/ApolloCorpus",
                "subset": subset,
            })
            count += 1
        print(f"{count} QA pairs")

    df = pd.DataFrame(rows)
    DST.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DST, index=False)
    print(f"\nSaved {len(df)} rows to {DST}")


if __name__ == "__main__":
    convert()
