"""DRIM-VisualReasonHard: JSON list of {images, doc_id, problem, solution, data_source}."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _hash_utils import write_shard


def iter_rows(src):
    with open(src) as f:
        data = json.load(f)
    base = os.path.splitext(os.path.basename(src))[0]
    for i, row in enumerate(data):
        rid = row.get("doc_id") or f"{base}_{i:08d}"
        prompt = row.get("problem") or ""
        resp = row.get("solution") or ""
        yield rid, prompt, resp


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    n = write_shard(iter_rows(src), dst)
    print(f"{src} -> {dst} ({n} rows)")
