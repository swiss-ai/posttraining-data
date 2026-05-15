"""RadImageNet-VQA: parquet (image, conversations[{from,value}], metadata)."""
import os
import sys
import pyarrow.parquet as pq
from _hash_utils import write_shard, first_human_assistant_fromvalue


def iter_rows(src):
    base = os.path.splitext(os.path.basename(src))[0]
    tbl = pq.read_table(src, columns=["conversations"])
    for i, row in enumerate(tbl.to_pylist()):
        convs = row.get("conversations") or []
        prompt, resp = first_human_assistant_fromvalue(convs)
        yield f"{base}_{i:08d}", prompt, resp


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    n = write_shard(iter_rows(src), dst)
    print(f"{src} -> {dst} ({n} rows)")
