"""EO-Data1.5M: per-task parquet with conversation:list<{from,value}>, image, ..."""
import os
import sys

import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _hash_utils import write_shard, first_human_assistant_fromvalue


def iter_rows(src):
    base = os.path.splitext(os.path.basename(src))[0]
    pf = pq.ParquetFile(src)
    for batch in pf.iter_batches(columns=["conversation"], batch_size=2000):
        convs_list = batch.column("conversation").to_pylist()
        for i, convs in enumerate(convs_list):
            prompt, resp = first_human_assistant_fromvalue(convs or [])
            yield f"{base}_{i:08d}", prompt, resp


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    n = write_shard(iter_rows(src), dst)
    print(f"{src} -> {dst} ({n} rows)")
