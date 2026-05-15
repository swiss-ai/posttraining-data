"""Filter representative prompts from source shards into a single
decontamination-input parquet that decontaminate.py can consume via the
sharegpt-vision adapter.

Args:
    --cluster-table  /path/to/clusters_prompt_prefix8.parquet
    --source-glob    "/capstor/.../*.parquet" (or arrow / json / jsonl ...)
    --compute-script /path/to/compute-<dataset>.py
    --output         /iopsstor/.../decontam-input.parquet
    [--num-proc 16]

The compute script's iter_rows(src) is used as the per-shard extractor —
that way prepare uses exactly the same id/prompt extraction as the dedup
hashing pipeline that produced the cluster table.
"""

import argparse
import importlib.util
import os
from glob import glob
from multiprocessing import Pool

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq


def _load_compute_module(path: str):
    spec = importlib.util.spec_from_file_location("_compute_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_MOD = None
_REP_SET = None


def _init(rep_set, compute_path):
    global _MOD, _REP_SET
    _MOD = _load_compute_module(compute_path)
    _REP_SET = rep_set


def _scan(src):
    out = []
    try:
        for rid, prompt, _response in _MOD.iter_rows(src):
            sid = str(rid)
            if sid in _REP_SET:
                out.append((sid, prompt or ""))
    except Exception as e:
        return f"!! ERROR scanning {src}: {e}"
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cluster-table", required=True)
    p.add_argument("--source-glob", required=True)
    p.add_argument("--compute-script", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--num-proc", type=int, default=16)
    args = p.parse_args()

    ct = pl.read_parquet(args.cluster_table, columns=["representative_id"])
    rep_set = set(str(r) for r in ct["representative_id"].to_list())
    print(f"loaded {len(rep_set)} representatives from {args.cluster_table}")

    files = []
    for pat in args.source_glob.split():
        files.extend(glob(pat))
    src_files = sorted(set(files))
    print(f"source glob matched {len(src_files)} files")
    if not src_files:
        raise SystemExit("no source files")

    pairs = []
    done = 0
    with Pool(args.num_proc, initializer=_init, initargs=(rep_set, args.compute_script)) as pool:
        for result in pool.imap_unordered(_scan, src_files, chunksize=2):
            if isinstance(result, str):
                print(result)
                continue
            pairs.extend(result)
            done += 1
            if done % 50 == 0 or done == len(src_files):
                print(f"  scanned {done}/{len(src_files)} files, collected {len(pairs)} prompts")

    print(f"writing {len(pairs)} rows to {args.output}")
    ids = [p[0] for p in pairs]
    convs = [[{"from": "human", "value": p[1]}] for p in pairs]
    out_tbl = pa.table({
        "id": pa.array(ids, type=pa.string()),
        "conversations": pa.array(convs),
    })
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    pq.write_table(out_tbl, args.output, compression="zstd")
    print("done")


if __name__ == "__main__":
    main()
