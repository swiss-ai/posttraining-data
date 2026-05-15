"""PersonaVLM: JSON list (sft.json) + JSONL (rl.jsonl), conversations[{from,value}]."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _hash_utils import write_shard, first_human_assistant_fromvalue


def iter_rows(src):
    base = os.path.splitext(os.path.basename(src))[0]
    if src.endswith(".jsonl"):
        with open(src) as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                rid = str(row.get("id", f"{base}_{i:08d}"))
                convs = row.get("conversations") or []
                prompt, resp = first_human_assistant_fromvalue(convs)
                yield rid, prompt, resp
    else:
        with open(src) as f:
            data = json.load(f)
        for i, row in enumerate(data):
            rid = str(row.get("id", f"{base}_{i:08d}"))
            convs = row.get("conversations") or []
            prompt, resp = first_human_assistant_fromvalue(convs)
            yield rid, prompt, resp


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    n = write_shard(iter_rows(src), dst)
    print(f"{src} -> {dst} ({n} rows)")
