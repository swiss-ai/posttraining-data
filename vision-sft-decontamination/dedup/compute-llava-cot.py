"""llava_cot_100k: HF arrow files with (id, image, conversations[{from,value}], data_source)."""
import os
import sys

import pyarrow as pa
import pyarrow.ipc as ipc

# Re-add for import resolution within batch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _hash_utils import write_shard, first_human_assistant_fromvalue


def iter_rows(src):
    # HF arrow files are IPC streams.
    try:
        with pa.memory_map(src, "r") as mm:
            tbl = ipc.open_stream(mm).read_all()
    except pa.ArrowInvalid:
        tbl = ipc.open_file(src).read_all()
    cols = tbl.column_names
    convs_idx = cols.index("conversations")
    id_idx = cols.index("id")
    for i in range(tbl.num_rows):
        rid = tbl.column(id_idx)[i].as_py()
        convs = tbl.column(convs_idx)[i].as_py() or []
        prompt, resp = first_human_assistant_fromvalue(convs)
        yield rid, prompt, resp


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    n = write_shard(iter_rows(src), dst)
    print(f"{src} -> {dst} ({n} rows)")
