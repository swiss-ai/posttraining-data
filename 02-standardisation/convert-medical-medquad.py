#!/usr/bin/env python3
"""Convert lavita/MedQuAD to parquet with messages column."""

import pandas as pd
from pathlib import Path

CACHE = Path("/capstor/store/cscs/swissai/infra01/medical_text/hf_hub_cache")
SRC = CACHE / "datasets--lavita--MedQuAD/snapshots/84ea67f83cec9692ad254eaa02c9731b24ecfe4c/data/train-00000-of-00001-e36383d177026d53.parquet"
DST = Path("/capstor/store/cscs/swissai/infra01/medical_text/pre-processed/medquad.parquet")


def convert():
    df = pd.read_parquet(SRC)
    print(f"Loaded {len(df)} rows")

    rows = []
    for _, r in df.iterrows():
        question = r.get("question", "")
        answer = r.get("answer", "")
        if not question or not answer:
            continue
        rows.append({
            "messages": [
                {"role": "user", "content": str(question)},
                {"role": "assistant", "content": str(answer)},
            ],
            "source_dataset": "lavita/MedQuAD",
            "document_source": r.get("document_source", ""),
            "question_type": r.get("question_type", ""),
            "question_focus": r.get("question_focus", ""),
            "umls_cui": r.get("umls_cui", ""),
            "umls_semantic_group": r.get("umls_semantic_group", ""),
        })

    out = pd.DataFrame(rows)
    DST.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(DST, index=False)
    print(f"Saved {len(out)} rows to {DST}")


if __name__ == "__main__":
    convert()
