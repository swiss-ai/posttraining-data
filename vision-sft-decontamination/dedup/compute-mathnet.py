"""MathNet: parquet with id + problem_markdown + solutions_markdown[]."""
import sys
import pyarrow.parquet as pq
from _hash_utils import write_shard


def iter_rows(src):
    tbl = pq.read_table(src, columns=["id", "problem_markdown", "solutions_markdown", "final_answer"])
    for i, row in enumerate(tbl.to_pylist()):
        rid = row.get("id") or f"row_{i:08d}"
        prompt = row.get("problem_markdown") or ""
        sols = row.get("solutions_markdown") or []
        resp = sols[0] if sols else (row.get("final_answer") or "")
        yield rid, prompt, resp


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    n = write_shard(iter_rows(src), dst)
    print(f"{src} -> {dst} ({n} rows)")
