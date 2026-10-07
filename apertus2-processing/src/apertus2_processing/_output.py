"""Bounded Arrow writers and readers, with explicit schemas."""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from pyarrow import ipc

from ._util import digest


def token_schema(weights):
    fields = [
        pa.field("source", pa.string(), nullable=False),
        pa.field("split", pa.string(), nullable=False),
        pa.field("row", pa.int64(), nullable=False),
        pa.field("record_id", pa.string()),
        pa.field("token_ids", pa.list_(pa.int32()), nullable=False),
        pa.field("token_count", pa.int64(), nullable=False),
    ]
    if weights:
        fields.append(pa.field("loss_weights", pa.list_(pa.float64()), nullable=False))
    return pa.schema(fields)


class TokenWriter:
    def __init__(self, directory, kind, weights):
        self.directory = Path(directory)
        self.kind = kind
        self.schema = token_schema(weights)
        self.writers = {}
        self.outputs = {}

    def write(self, rows):
        by_split = {}
        for row in rows:
            by_split.setdefault(row["split"], []).append(row)
        for split, records in by_split.items():
            if split not in self.writers:
                name = f"tokens-{digest(split)[:16]}." + (
                    "arrow" if self.kind == "hf" else "parquet"
                )
                path = self.directory / name
                if self.kind == "hf":
                    sink = pa.OSFile(str(path), "wb")
                    writer = ipc.new_stream(sink, self.schema)
                    self.writers[split] = (writer, sink)
                else:
                    self.writers[split] = (pq.ParquetWriter(path, self.schema), None)
                self.outputs[split] = {"split": split, "path": name, "rows": 0}
            self.writers[split][0].write_table(pa.Table.from_pylist(records, schema=self.schema))
            self.outputs[split]["rows"] += len(records)

    def close(self):
        for writer, sink in self.writers.values():
            writer.close()
            if sink is not None:
                sink.close()
        return list(self.outputs.values())


def iter_token_rows(run_dir, manifests):
    for manifest in manifests:
        directory = Path(run_dir) / "shards" / f"{manifest['index']:06d}"
        for entry in manifest["token_files"]:
            path = directory / entry["path"]
            if path.suffix == ".parquet":
                for batch in pq.ParquetFile(path).iter_batches(batch_size=128):
                    yield from batch.to_pylist()
            else:
                with pa.memory_map(str(path), "r") as source:
                    for batch in ipc.open_stream(source):
                        yield from batch.to_pylist()
