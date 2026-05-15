"""PangeaInstruct master JSON: list of {id, image, conversations[{from,value}], language}.

The PangeaIns.json file is 12 GB so we stream it with ijson rather than loading
the whole list into memory."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _hash_utils import write_shard, first_human_assistant_fromvalue


def iter_rows(src):
    import ijson
    base = os.path.splitext(os.path.basename(src))[0]
    with open(src, "rb") as f:
        for i, row in enumerate(ijson.items(f, "item")):
            rid = row.get("id") or f"{base}_{i:08d}"
            convs = row.get("conversations") or []
            prompt, resp = first_human_assistant_fromvalue(convs)
            yield str(rid), prompt, resp


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    n = write_shard(iter_rows(src), dst)
    print(f"{src} -> {dst} ({n} rows)")
