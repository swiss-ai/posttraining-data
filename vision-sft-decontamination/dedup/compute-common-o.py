"""Common-O: parquet (image_1, image_2, question, answer, ...). No id; synthesize."""
import os
import sys
import pyarrow.parquet as pq
from _hash_utils import write_shard


def iter_rows(src):
    base = os.path.splitext(os.path.basename(src))[0]
    tbl = pq.read_table(src, columns=["question", "answer"])
    for i, row in enumerate(tbl.to_pylist()):
        yield f"{base}_{i:08d}", row["question"] or "", row["answer"] or ""


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    n = write_shard(iter_rows(src), dst)
    print(f"{src} -> {dst} ({n} rows)")
