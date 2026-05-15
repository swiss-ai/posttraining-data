"""OneThinker: JSON list with {problem_id, problem, answer, images, ...}."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _hash_utils import write_shard


def iter_rows(src):
    base = os.path.splitext(os.path.basename(src))[0]
    # Files can be large (~300 MB); stream via ijson to avoid the whole-file load
    import ijson
    with open(src, "rb") as f:
        for i, row in enumerate(ijson.items(f, "item")):
            rid = str(row.get("problem_id") or row.get("id") or f"{base}_{i:08d}")
            prompt = row.get("problem") or ""
            resp = row.get("answer") or ""
            yield rid, prompt, resp


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    n = write_shard(iter_rows(src), dst)
    print(f"{src} -> {dst} ({n} rows)")
