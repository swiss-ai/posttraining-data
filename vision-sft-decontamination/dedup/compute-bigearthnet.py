"""BigEarthNet: single parquet with (ID, input, output, ...)."""
import sys
import pyarrow.parquet as pq

import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _hash_utils import write_shard


def iter_rows(src):
    pf = pq.ParquetFile(src)
    for batch in pf.iter_batches(columns=["ID", "input", "output"], batch_size=20000):
        ids = batch.column("ID").to_pylist()
        ins = batch.column("input").to_pylist()
        outs = batch.column("output").to_pylist()
        for rid, p, r in zip(ids, ins, outs):
            yield str(rid), p or "", r or ""


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    n = write_shard(iter_rows(src), dst)
    print(f"{src} -> {dst} ({n} rows)")
