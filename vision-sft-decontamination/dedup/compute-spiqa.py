"""SPIQA: JSON list of {image, question, thinking, ...}."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _hash_utils import write_shard


def iter_rows(src):
    base = os.path.splitext(os.path.basename(src))[0]
    import ijson
    with open(src, "rb") as f:
        for i, row in enumerate(ijson.items(f, "item")):
            rid = str(row.get("id") or f"{base}_{i:08d}")
            prompt = row.get("question") or ""
            resp = row.get("thinking") or row.get("answer") or ""
            yield rid, prompt, resp


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    n = write_shard(iter_rows(src), dst)
    print(f"{src} -> {dst} ({n} rows)")
