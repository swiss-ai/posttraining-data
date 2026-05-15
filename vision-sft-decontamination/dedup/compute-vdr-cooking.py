"""VDR_Cooking_Recipes: parquet rows (id, query, image, answer, language)."""
import sys
import pyarrow.parquet as pq
from _hash_utils import write_shard


def iter_rows(src):
    tbl = pq.read_table(src, columns=["id", "query", "answer"])
    for row in tbl.to_pylist():
        yield row["id"], row["query"] or "", row["answer"] or ""


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    n = write_shard(iter_rows(src), dst)
    print(f"{src} -> {dst} ({n} rows)")
