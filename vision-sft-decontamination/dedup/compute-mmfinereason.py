"""MMFineReason: parquet with question + id + answer + qwen3vl_235b_thinking_response."""
import sys
import pyarrow.parquet as pq
from _hash_utils import write_shard


def iter_rows(src):
    tbl = pq.read_table(src, columns=["id", "question", "answer", "qwen3vl_235b_thinking_response"])
    for row in tbl.to_pylist():
        rid = str(row.get("id"))
        prompt = row.get("question") or ""
        resp = row.get("qwen3vl_235b_thinking_response") or row.get("answer") or ""
        yield rid, prompt, resp


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    n = write_shard(iter_rows(src), dst)
    print(f"{src} -> {dst} ({n} rows)")
