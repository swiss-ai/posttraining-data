"""RSRCC: metadata.csv per split (train/val/test) + image dirs.

Columns inferred from the metadata.csv header at runtime; we extract image path,
prompt and response when they exist (most VQA-style CSVs use `question`/`answer`
or `caption`)."""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _hash_utils import write_shard


_PROMPT_KEYS = ("question", "instruction", "prompt", "query")
_RESPONSE_KEYS = ("answer", "response", "caption", "target", "completion")
_ID_KEYS = ("id", "image_id", "filename", "image")


def _pick(row, keys):
    for k in keys:
        v = row.get(k)
        if v:
            return v
    return ""


def iter_rows(src):
    base = os.path.basename(os.path.dirname(src)) + "_" + os.path.splitext(os.path.basename(src))[0]
    with open(src, newline="") as f:
        for i, row in enumerate(csv.DictReader(f)):
            rid = _pick(row, _ID_KEYS) or f"{base}_{i:08d}"
            prompt = _pick(row, _PROMPT_KEYS)
            resp = _pick(row, _RESPONSE_KEYS)
            yield rid, prompt, resp


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    n = write_shard(iter_rows(src), dst)
    print(f"{src} -> {dst} ({n} rows)")
