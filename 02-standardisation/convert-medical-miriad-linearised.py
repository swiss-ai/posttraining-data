#!/usr/bin/env python3
"""Convert miriad.parquet to Apertus linearised format.

No reasoning — response block only.
"""

import pandas as pd
from pathlib import Path
from tqdm import tqdm

from linearise_medical_utils import build_linearised_messages

SRC = Path("/capstor/store/cscs/swissai/infra01/medical_text/pre-processed/miriad.parquet")
DST = Path("/capstor/store/cscs/swissai/infra01/medical_text/linearised/miriad.parquet")


def convert():
    print(f"Reading {SRC} ...")
    df = pd.read_parquet(SRC)
    print(f"Loaded {len(df)} rows")

    rows = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc="Linearising"):
        msgs = r["messages"]
        user_content = msgs[0]["content"]
        assistant_content = msgs[1]["content"]

        rows.append({
            "messages": build_linearised_messages(user_content, assistant_content),
            "source_dataset": r.get("source_dataset", ""),
            "paper_title": r.get("paper_title", ""),
            "passage_text": r.get("passage_text", ""),
            "year": r.get("year"),
            "venue": r.get("venue", ""),
            "specialty": r.get("specialty", ""),
        })

    out = pd.DataFrame(rows)
    DST.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(DST, index=False)
    print(f"Saved {len(out)} rows to {DST}")


if __name__ == "__main__":
    convert()
