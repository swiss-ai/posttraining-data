"""nemotron archive/*.parquet: (id, messages:string [JSON list], images:map<str,binary>).

The messages column is a JSON-encoded list of {role, content} dicts where each
content is either a string or a list of mixed text/image parts. We pull the
first text snippet from the user role and the first text from the assistant."""
import json
import os
import sys

import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _hash_utils import write_shard


def _first_text(content):
    """content is either a string or a list of mixed text/dicts."""
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
    pf = pq.ParquetFile(src)
    for batch in pf.iter_batches(columns=["id", "messages"], batch_size=2000):
        ids = batch.column("id").to_pylist()
        msgs_list = batch.column("messages").to_pylist()
        for rid, msgs_str in zip(ids, msgs_list):
            try:
                msgs = json.loads(msgs_str) if msgs_str else []
            except (ValueError, TypeError):
                msgs = []
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
