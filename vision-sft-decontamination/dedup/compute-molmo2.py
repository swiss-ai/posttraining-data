"""Molmo2-MultiImageQA: parquet with qa_pairs (lists of questions and answers per row).

Emits one output row per Q-A pair (so a multi-question Molmo row produces N
records, each with its own id derived from the image_sha256 + qa index)."""
import os
import sys

import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _hash_utils import write_shard


def iter_rows(src):
    base = os.path.splitext(os.path.basename(src))[0]
    tbl = pq.read_table(src, columns=["image_sha256s", "qa_pairs"])
    for i, row in enumerate(tbl.to_pylist()):
        qa = row.get("qa_pairs") or {}
        questions = qa.get("question") or []
        answers = qa.get("answer") or []
        shas = row.get("image_sha256s") or []
        first_sha = shas[0] if shas else f"row_{i:08d}"
        for j, (q, a) in enumerate(zip(questions, answers)):
            yield f"{first_sha}_qa{j}", q or "", a or ""


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    n = write_shard(iter_rows(src), dst)
    print(f"{src} -> {dst} ({n} rows)")
