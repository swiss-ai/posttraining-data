"""pixmo-cap-qa: parquet (image_url, question, answer, messages) — no sha; derive id from url+row."""
import os
import sys
import pyarrow.parquet as pq
from _hash_utils import write_shard, h64


def iter_rows(src):
    base = os.path.splitext(os.path.basename(src))[0]
    tbl = pq.read_table(src, columns=["image_url", "question", "answer"])
    for i, row in enumerate(tbl.to_pylist()):
        url = row.get("image_url") or ""
        rid = f"{base}_{i:08d}_{h64(url):016x}" if url else f"{base}_{i:08d}"
        yield rid, row["question"] or "", row["answer"] or ""


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    n = write_shard(iter_rows(src), dst)
    print(f"{src} -> {dst} ({n} rows)")
