"""Native-format inputs (Parquet files or a saved HF dataset): planning and reads."""

import functools
import hashlib
from pathlib import Path

import pyarrow.parquet as pq

_READ_BATCH = 1024
_FINGERPRINT_BYTES = 64 * 1024


def _visible(path, root):
    return not any(part.startswith(".") for part in path.relative_to(root).parts)


def _files(root):
    root = Path(root)
    if root.is_file():
        return [root]
    return sorted(p for p in root.rglob("*") if p.is_file() and _visible(p, root))


def _is_hf(path):
    return path.is_dir() and any((path / n).exists() for n in ("dataset_dict.json", "state.json"))


@functools.cache
def _load_hf(path, split):
    from datasets import DatasetDict, load_from_disk

    data = load_from_disk(str(path))
    if isinstance(data, DatasetDict):
        if split not in data:
            raise ValueError(f"split {split!r} not in {sorted(data)}; use --split")
        data = data[split]
    return data


def snapshot(*roots):
    """Metadata plus bounded content fingerprints; source files must stay immutable.

    Lustre can report only whole-second mtimes. Sampling both ends also detects
    same-size rewrites within that second without reading multi-terabyte inputs.
    This hashes entire small files, but is not a full checksum of large files.
    """
    result = []
    for root in roots:
        for path in _files(root):
            stat = path.stat()
            with path.open("rb") as stream:
                digest = hashlib.sha256(stream.read(_FINGERPRINT_BYTES))
                if stat.st_size > _FINGERPRINT_BYTES:
                    stream.seek(max(_FINGERPRINT_BYTES, stat.st_size - _FINGERPRINT_BYTES))
                    digest.update(stream.read(_FINGERPRINT_BYTES))
            result.append(
                {
                    "path": str(path),
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "sample_sha256": digest.hexdigest(),
                }
            )
    return result


def _row_group_shards(sizes, shard_rows):
    """Runs of consecutive whole row groups holding at least `shard_rows` rows."""
    shards, start, groups, rows = [], 0, [], 0
    for group, size in enumerate(sizes):
        groups.append(group)
        rows += size
        if rows >= shard_rows:
            shards.append({"start": start, "end": start + rows, "row_groups": groups})
            start, groups, rows = start + rows, [], 0
    if rows:
        shards.append({"start": start, "end": start + rows, "row_groups": groups})
    return shards


def plan(path, column, split, shard_rows):
    """Return the inputs in processing order and their shards, from metadata only.

    HF shards hold `shard_rows` rows; Parquet shards are runs of whole row groups
    holding at least `shard_rows` rows, so no row group is read twice.
    """
    path = Path(path).resolve()
    if not path.exists():
        raise ValueError(f"input does not exist: {path}")
    if _is_hf(path):
        data = _load_hf(path, split)
        if column not in data.column_names:
            raise ValueError(f"column {column!r} not in {data.column_names}; use --column")
        sources = [{"kind": "hf", "path": str(path), "split": split, "rows": len(data)}]
        shards = [
            {"source": 0, "start": start, "end": min(start + shard_rows, len(data))}
            for start in range(0, len(data), shard_rows)
        ]
        return sources, shards
    files = [p for p in _files(path) if p.suffix == ".parquet"]
    if not files:
        raise ValueError(f"no .parquet files or saved HF dataset at {path}")
    sources, shards = [], []
    for index, file in enumerate(files):
        parquet = pq.ParquetFile(file)
        if column not in parquet.schema_arrow.names:
            raise ValueError(f"column {column!r} not in {file}; use --column")
        sizes = [parquet.metadata.row_group(g).num_rows for g in range(parquet.num_row_groups)]
        sources.append({"kind": "parquet", "path": str(file), "rows": sum(sizes)})
        shards += [{"source": index} | part for part in _row_group_shards(sizes, shard_rows)]
    return sources, shards


def rows(source, shard, column):
    """Yield the raw native JSON of each row in the shard, in order."""
    if source["kind"] == "parquet":
        parquet = pq.ParquetFile(source["path"])
        for batch in parquet.iter_batches(
            _READ_BATCH, row_groups=shard["row_groups"], columns=[column]
        ):
            yield from batch.column(0).to_pylist()
    else:
        data = _load_hf(source["path"], source["split"]).select_columns([column])
        for batch in data.select(range(shard["start"], shard["end"])).iter(_READ_BATCH):
            yield from batch[column]
