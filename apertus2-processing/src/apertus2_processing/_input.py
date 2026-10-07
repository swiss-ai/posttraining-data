"""Disk-backed input discovery and deterministic contiguous row partitions."""

import json
from pathlib import Path

from ._guards import snapshot, verify_snapshot


def discover(path, kind="auto", split=None):
    path = Path(path).resolve()
    if not path.exists():
        raise ValueError(f"input does not exist: {path}")
    if kind == "auto":
        kind = "jsonl" if path.is_file() or any(path.glob("*.jsonl")) else "hf"
    sources = []
    if kind == "jsonl":
        files = [path] if path.is_file() else sorted(path.glob("*.jsonl"))
        if not files:
            raise ValueError("no JSONL files found")
        for file in files:
            file_snapshot = snapshot(file)
            with file.open("rb") as stream:
                count = sum(1 for _ in stream)
            verify_snapshot(file, file_snapshot)
            sources.append(
                {
                    "path": str(file),
                    "split": split or "train",
                    "count": count,
                    "kind": kind,
                    "files": file_snapshot,
                }
            )
    else:
        from datasets import DatasetDict, load_from_disk

        files = snapshot(path)
        data = load_from_disk(str(path))
        splits = data if isinstance(data, DatasetDict) else {"train": data}
        if split is not None and split not in splits:
            raise ValueError(f"unknown split: {split}")
        verify_snapshot(path, files)
        for name in sorted(splits):
            if split is None or split == name:
                sources.append(
                    {
                        "path": str(path),
                        "split": name,
                        "count": len(splits[name]),
                        "kind": kind,
                        "files": files,
                    }
                )
    return sources


def make_plan(sources, num_shards):
    total = sum(source["count"] for source in sources)
    shards = [
        {
            "index": i,
            "start": total * i // num_shards,
            "end": total * (i + 1) // num_shards,
            "slices": [],
        }
        for i in range(num_shards)
    ]
    base = 0
    for source_index, source in enumerate(sources):
        slices = []
        for shard in shards:
            start = max(shard["start"], base) - base
            end = min(shard["end"], base + source["count"]) - base
            if start < end:
                part = {"source": source_index, "start": start, "end": end}
                shard["slices"].append(part)
                slices.append(part)
        if source["kind"] == "jsonl" and slices:
            wanted = {p[edge] for p in slices for edge in ("start", "end")}
            offsets = {}
            with Path(source["path"]).open("rb") as stream:
                row = 0
                while True:
                    if row in wanted:
                        offsets[row] = stream.tell()
                    if not stream.readline():
                        break
                    row += 1
            for part in slices:
                part["byte_start"] = offsets[part["start"]]
                part["byte_end"] = offsets[part["end"]]
        base += source["count"]
    return shards


def iter_rows(sources, shard, column, context_column):
    """Yield record locations and raw JSON; invalid records remain reportable."""
    for part in shard["slices"]:
        source = sources[part["source"]]
        if source["kind"] == "jsonl":
            with Path(source["path"]).open("rb") as stream:
                stream.seek(part["byte_start"])
                for row in range(part["start"], part["end"]):
                    raw = stream.readline()
                    yield unpack(raw, source, row, column, context_column)
                if stream.tell() != part["byte_end"]:
                    raise ValueError("JSONL changed after planning")
        else:
            from datasets import DatasetDict, load_from_disk

            data = load_from_disk(source["path"])
            if isinstance(data, DatasetDict):
                data = data[source["split"]]
            selected = data.select(range(part["start"], part["end"]))
            row = part["start"]
            # Sequential iteration avoids repeating random-access Arrow lookups.
            # One input row at a time also keeps oversized records isolated.
            for batch in selected.iter(batch_size=1):
                value = {key: values[0] for key, values in batch.items()}
                yield unpack(value, source, row, column or "conversation_json", context_column)
                row += 1


def unpack(value, source, row, column, context_column):
    location = {"source": source["path"], "split": source["split"], "row": row, "record_id": None}
    try:
        if source["kind"] == "jsonl" and column is None:
            return location, value, None, None
        if isinstance(value, bytes):
            value = json.loads(value)
        if not isinstance(value, dict):
            raise TypeError("wrapped record must be an object")
        record_id = value.get("record_id", value.get("conversation_id", value.get("id")))
        if record_id is not None:
            record_id = str(record_id)
            record_id.encode("utf-8")
            location["record_id"] = record_id
        raw = value[column]
        if not isinstance(raw, (str, bytes)):
            raise TypeError(f"{column} must be a native JSON string")
        context = value[context_column] if context_column else None
        return location, raw, context, None
    except (KeyError, ValueError, TypeError, UnicodeError) as error:
        return location, b"", None, str(error)
