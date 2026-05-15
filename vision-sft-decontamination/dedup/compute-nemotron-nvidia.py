"""nemotron hf___nvidia___ jsonl files: same {id, messages:[{role, content: [...]}]}
structure as nemotron archive parquets, but in line-delimited JSON."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _hash_utils import write_shard


def _first_text(content):
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    for part in content:
        if isinstance(part, str):
            return part
        if isinstance(part, dict):
            if part.get("type") == "text" and "text" in part:
                return part["text"]
            if "text" in part:
                return part["text"]
    return ""


def iter_rows(src):
    base = os.path.splitext(os.path.basename(src))[0]
    with open(src) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rid = row.get("id") or f"{base}_{i:08d}"
            msgs = row.get("messages") or []
            prompt, resp = "", ""
            for m in msgs:
                role = m.get("role", "")
                content = m.get("content", "")
                if not prompt and role in ("user", "human"):
                    prompt = _first_text(content)
                elif not resp and role == "assistant":
                    resp = _first_text(content)
                if prompt and resp:
                    break
            yield str(rid), prompt, resp


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    n = write_shard(iter_rows(src), dst)
    print(f"{src} -> {dst} ({n} rows)")
