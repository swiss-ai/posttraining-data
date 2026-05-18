#!/usr/bin/env python3
"""Convert miriad/miriad-4.4M to parquet with messages column."""

import pandas as pd
from pathlib import Path

CACHE = Path("/capstor/store/cscs/swissai/infra01/medical_text/hf_hub_cache")
SNAP = CACHE / "datasets--miriad--miriad-4.4M/snapshots/ba958e430c08e3291aed5ffd3292f0962ade02e1/data"
DST = Path("/capstor/store/cscs/swissai/infra01/medical_text/pre-processed/miriad.parquet")


def convert():
    shards = sorted(SNAP.glob("*.parquet"))
    print(f"Found {len(shards)} shard files")

    chunks = []
    for i, shard in enumerate(shards):
        df = pd.read_parquet(shard)
        print(f"  Shard {i+1}/{len(shards)}: {len(df)} rows")

        records = []
        for _, r in df.iterrows():
            records.append({
                "messages": [
                    {"role": "user", "content": str(r["question"])},
                    {"role": "assistant", "content": str(r["answer"])},
                ],
                "source_dataset": "miriad/miriad-4.4M",
                "paper_title": r.get("paper_title", ""),
                "passage_text": r.get("passage_text", ""),
                "year": r.get("year"),
                "venue": r.get("venue", ""),
                "specialty": r.get("specialty", ""),
            })
        chunks.append(pd.DataFrame(records))

    out = pd.concat(chunks, ignore_index=True)
    DST.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(DST, index=False)
    print(f"Saved {len(out)} rows to {DST}")


if __name__ == "__main__":
    convert()
