"""Smoke test: read one Innovator-VL parquet shard, drop image bytes, strip
<image> tokens from conversations, write a text-only parquet to scratch.
Measures wallclock per stage and extrapolates corpus-wide cost."""

import os
import re
import time

import pyarrow as pa
import pyarrow.parquet as pq

SRC = "/capstor/store/cscs/swissai/infra01/vision-datasets/raw/sft/hf___InnovatorLab___Innovator-VL-Instruct-46M/data1/SFT_000001.parquet"
DST_DIR = "/iopsstor/scratch/cscs/schlag/apertus1p5-decontam/smoke"
DST = os.path.join(DST_DIR, "SFT_000001.txt.parquet")
TOTAL_SHARDS = 15568

os.makedirs(DST_DIR, exist_ok=True)

t0 = time.time()
tbl = pq.read_table(SRC, columns=["id", "conversations"])
t1 = time.time()

img_pat = re.compile(r"<image>\n?")
id_col = tbl.column("id")
new_rows = []
stripped = 0
for conv_list in tbl.column("conversations").to_pylist():
    new_list = []
    for turn in conv_list:
        v = turn["value"]
        new_v, n = img_pat.subn("", v)
        if n:
            stripped += n
        new_list.append({"from": turn["from"], "value": new_v})
    new_rows.append(new_list)
new_tbl = pa.table({"id": id_col, "conversations": pa.array(new_rows)})
t2 = time.time()

pq.write_table(new_tbl, DST, compression="zstd")
t3 = time.time()

in_size = os.path.getsize(SRC)
out_size = os.path.getsize(DST)

print(f"src           : {SRC}")
print(f"dst           : {DST}")
print(f"rows          : {tbl.num_rows}")
print(f"read          : {t1 - t0:.2f}s")
print(f"strip         : {t2 - t1:.2f}s  ({stripped} <image> tokens stripped)")
print(f"write         : {t3 - t2:.2f}s")
print(f"total         : {t3 - t0:.2f}s")
print(f"input file    : {in_size / 1e9:.2f} GB")
print(f"output file   : {out_size / 1e6:.2f} MB")
print(f"compression   : {in_size / out_size:.0f}x")

per = t3 - t0
seq_h = TOTAL_SHARDS * per / 3600
print(f"\nExtrapolation across {TOTAL_SHARDS} shards (single-process baseline)")
print(f"  sequential        : {seq_h:.1f} h")
for n in (50, 100, 200, 500):
    print(f"  {n:3d}-wide parallel: {seq_h / n * 60:6.1f} min wallclock")
print(f"  est text corpus   : {out_size * TOTAL_SHARDS / 1e12:.2f} TB")
