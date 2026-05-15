"""ChartVerse-SFT-1.8M: parquet rows (id, images, code, question, answer, ...)."""
import sys
import pyarrow.parquet as pq
from _hash_utils import write_shard


def iter_rows(src):
    tbl = pq.read_table(src, columns=["id", "question", "answer"])
    for row in tbl.to_pylist():
        yield row["id"], row["question"] or "", row["answer"] or ""


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    n = write_shard(iter_rows(src), dst)
    print(f"{src} -> {dst} ({n} rows)")
