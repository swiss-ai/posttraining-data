"""pixmo-ask-model-anything: parquet (image_url, image_sha256, question, answer)."""
import sys
import pyarrow.parquet as pq
from _hash_utils import write_shard


def iter_rows(src):
    tbl = pq.read_table(src, columns=["image_sha256", "question", "answer"])
    for i, row in enumerate(tbl.to_pylist()):
        rid = row.get("image_sha256") or f"row_{i:08d}"
        yield rid, row["question"] or "", row["answer"] or ""


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    n = write_shard(iter_rows(src), dst)
    print(f"{src} -> {dst} ({n} rows)")
