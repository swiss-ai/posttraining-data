"""Loss-preserving schema v2 ingestion; all rows, explicit provenance, file-level resume."""

import argparse
import concurrent.futures
import gzip
import hashlib
import json
import os
from pathlib import Path
import time

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from standardize import (
    SCHEMA as V1_SCHEMA,
    atomic_json,
    dumps,
    first_value,
    normalize,
    safe_name,
)
from preprocessors import get_processor

SCHEMA = pa.schema(
    list(V1_SCHEMA)
    + [
        ("schema_version", pa.int16()),
        ("source_file", pa.string()),
        ("license", pa.large_string()),
        ("license_status", pa.string()),
        ("license_notes", pa.large_string()),
        ("source_license_json", pa.large_string()),
        ("is_synthetic", pa.bool_()),
        ("synthetic_status", pa.string()),
        ("synthetic_notes", pa.large_string()),
        ("metadata_source", pa.string()),
        ("normalization_status", pa.string()),
        ("normalization_notes", pa.large_string()),
        ("source_record_json", pa.large_string()),
    ],
    metadata={b"schema_name": b"swissai_unified", b"schema_version": b"2"},
)


def normalize_v2(row, entry, file, index):
    return get_processor(entry["repo"]).normalize(row, entry, file, index)


def normalize_v2_shared(row, entry, file, index, processor):
    original = dumps(row)
    row = dict(row)
    notes = []
    row = processor.prepare(row, entry, file)
    for key in ("messages", "conversations", "conversation", "trajectory"):
        value = row.get(key)
        if isinstance(value, str):
            try:
                decoded = json.loads(value)
                if isinstance(decoded, list):
                    row[key] = decoded
            except (ValueError, TypeError):
                notes.append(f"{key} is serialized text, not parsed JSON messages")
        if isinstance(row.get(key), list):
            messages = []
            for msg in row[key]:
                if not isinstance(msg, dict):
                    notes.append(
                        "Non-object conversation element preserved in source_record_json"
                    )
                    continue
                msg = dict(msg)
                role = msg.get("role", msg.get("from", msg.get("speaker")))
                if not isinstance(role, (str, type(None))):
                    notes.append("Non-string role preserved in source_record_json")
                    role = None
                msg["role"] = {
                    "human": "user",
                    "gpt": "assistant",
                    "bot": "assistant",
                }.get(role, role)
                if "content" not in msg:
                    msg["content"] = msg.get("value", msg.get("text"))
                messages.append(msg)
            row["messages"] = messages
            break
    result = normalize(
        row,
        repo=entry["repo"],
        revision=entry["revision"],
        config=file["source_config"],
        split=file["source_split"],
        index=index,
        category=entry["category"],
    )
    if isinstance(row.get("messages"), list):
        users = [m.get("content") for m in row["messages"] if m.get("role") == "user"]
        assistants = [
            m.get("content") for m in row["messages"] if m.get("role") == "assistant"
        ]
        if result["prompt"] is None and users:
            result["prompt"] = (
                users[0] if isinstance(users[0], str) else dumps(users[0])
            )
        if result["response"] is None and assistants:
            result["response"] = (
                assistants[-1]
                if isinstance(assistants[-1], str)
                else dumps(assistants[-1])
            )
    for key in ("prompt", "response"):
        if result[key] in ("-", "", None):
            result[key] = None
            notes.append(
                f"{key} unavailable; source identifiers preserved for reconstruction"
            )
    if result["tests_json"] is None and "testcase" in row:
        result["tests_json"] = dumps(row["testcase"])
    synth = entry.get("synthetic_status", "unknown")
    result.update(
        schema_version=2,
        source_file=file["path"],
        license=entry["license"],
        license_status=entry["license_status"],
        license_notes=entry["license_notes"],
        source_license_json=dumps(row.get("license", row.get("licenses"))),
        synthetic_status=synth,
        is_synthetic={"synthetic": True, "non_synthetic": False}.get(synth),
        synthetic_notes=entry["synthetic_notes"],
        metadata_source=entry["metadata_source"],
        normalization_status="needs_review" if notes else "mapped",
        normalization_notes=dumps(notes),
        source_record_json=original,
    )
    result["id"] = hashlib.sha256(
        f"{entry['repo']}\0{entry['revision']}\0{file['path']}\0{index}".encode()
    ).hexdigest()
    return result


def read_rows(path):
    if str(path).endswith(".parquet"):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=512):
            yield from batch.to_pylist()
        return
    if str(path).endswith(".arrow"):
        with pa.memory_map(str(path), "r") as handle:
            try:
                reader = pa.ipc.open_file(handle)
                for i in range(reader.num_record_batches):
                    yield from reader.get_batch(i).to_pylist()
            except pa.ArrowInvalid:
                handle.seek(0)
                for batch in pa.ipc.open_stream(handle):
                    yield from batch.to_pylist()
        return
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8-sig") as handle:
        if str(path).endswith((".jsonl", ".jsonl.gz")):
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    yield row if isinstance(row, dict) else {"value": row}
        else:
            try:
                obj = json.load(handle)
            except json.JSONDecodeError:
                # Some publishers store JSONL using a .json filename.
                handle.seek(0)
                for line in handle:
                    if line.strip():
                        row = json.loads(line)
                        yield row if isinstance(row, dict) else {"value": row}
                return
            if isinstance(obj, dict):
                for key in ("data", "train", "examples"):
                    if isinstance(obj.get(key), list):
                        obj = obj[key]
                        break
            if isinstance(obj, list):
                for row in obj:
                    yield row if isinstance(row, dict) else {"value": row}
            else:
                yield obj if isinstance(obj, dict) else {"value": obj}


def file_partition(root, entry, file):
    return (
        Path(root)
        / entry["category"]
        / entry["repo"]
        / ("file=" + hashlib.sha256(file["path"].encode()).hexdigest()[:24])
    )


def process_file(root, entry, file):
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    partition = file_partition(root, entry, file)
    partition.mkdir(parents=True, exist_ok=True)
    success = partition / "_SUCCESS.json"
    if success.exists():
        state = json.loads(success.read_text())
        if (
            state["revision"] != entry["revision"]
            or state["schema_version"] != 2
            or state.get("code_policy_version") != entry.get("code_policy_version")
        ):
            raise ValueError("Existing partition revision/schema mismatch")
        for shard in state["files"]:
            pqfile = pq.ParquetFile(partition / shard["name"])
            if (
                not pqfile.schema_arrow.equals(SCHEMA)
                or pqfile.metadata.num_rows != shard["rows"]
            ):
                raise ValueError("Completed shard validation failed")
        return state
    checkpoint = partition / "_progress.json"
    state = (
        json.loads(checkpoint.read_text())
        if checkpoint.exists()
        else dict(
            rows=0,
            files=[],
            source_rows=0,
            excluded_noncode=0,
            excluded_unknown=0,
            code_policy_version=entry.get("code_policy_version"),
        )
    )
    if state.get("code_policy_version") != entry.get("code_policy_version"):
        raise ValueError("Checkpoint filter policy mismatch")
    resume = state.get("source_rows", state["rows"])
    token = os.environ.get("HF_TOKEN")
    path = hf_hub_download(
        entry["repo"],
        filename=file["path"],
        revision=entry["revision"],
        repo_type="dataset",
        token=token,
    )
    rows, size = [], 0

    def flush():
        nonlocal rows, size
        if not rows:
            return
        name = f"part-{len(state['files']):06d}.parquet"
        tmp = partition / (name + ".tmp")
        pq.write_table(
            pa.Table.from_pylist(rows, schema=SCHEMA),
            tmp,
            compression="zstd",
            compression_level=3,
        )
        if pq.ParquetFile(tmp).metadata.num_rows != len(rows):
            raise ValueError("Row count mismatch")
        tmp.replace(partition / name)
        state["files"].append(dict(name=name, rows=len(rows)))
        state["rows"] += len(rows)
        atomic_json(checkpoint, state)
        rows, size = [], 0

    seen = 0
    for index, row in enumerate(read_rows(path)):
        seen = index + 1
        if index < resume:
            continue
        state["source_rows"] = seen
        selection = get_processor(entry["repo"]).select(row, entry, file)
        if selection != "keep":
            state[selection] += 1
            if seen % 2000 == 0:
                flush()
                atomic_json(checkpoint, state)
            continue
        normalized = normalize_v2(row, entry, file, index)
        rows.append(normalized)
        size += len(normalized["source_record_json"]) * 3
        if len(rows) >= 2000 or size >= 64 * 1024 * 1024:
            flush()
    flush()
    if (
        state["rows"]
        + state.get("excluded_noncode", 0)
        + state.get("excluded_unknown", 0)
        != seen
    ):
        raise ValueError("Source/output row count mismatch")
    state.update(
        repo=entry["repo"],
        source_file=file["path"],
        revision=entry["revision"],
        schema_version=2,
    )
    atomic_json(success, state)
    checkpoint.unlink(missing_ok=True)
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--attempts", type=int, default=3)
    args = parser.parse_args()
    catalog = json.loads(args.catalog.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output / "catalog.json", catalog)
    pending = [(e, f) for e in catalog["datasets"] for f in e["files"]]
    failures = []
    for attempt in range(args.attempts):
        failed = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
            tasks = {
                pool.submit(process_file, str(args.output), e, f): (e, f)
                for e, f in pending
            }
            for future in concurrent.futures.as_completed(tasks):
                entry, file = tasks[future]
                try:
                    result = future.result()
                    print(
                        f"DONE {entry['repo']} {file['path']} rows={result['rows']}",
                        flush=True,
                    )
                except Exception as exc:
                    failed.append((entry, file))
                    print(
                        f"FAILED {entry['repo']} {file['path']} {type(exc).__name__}: {exc}",
                        flush=True,
                    )
        pending = failed
        if not pending:
            break
    for entry in catalog["datasets"]:
        root = args.output / entry["category"] / entry["repo"]
        root.mkdir(parents=True, exist_ok=True)
        atomic_json(root / "source.json", entry)
        missing, records, count = [], 0, 0
        excluded_noncode = excluded_unknown = 0
        for file in entry["files"]:
            marker = file_partition(args.output, entry, file) / "_SUCCESS.json"
            if not marker.exists():
                missing.append(file["path"])
            else:
                summary = json.loads(marker.read_text())
                records += summary["rows"]
                count += len(summary["files"])
                excluded_noncode += summary.get("excluded_noncode", 0)
                excluded_unknown += summary.get("excluded_unknown", 0)
        if missing:
            failures.append(
                dict(repo=entry["repo"], missing_files=missing, partial_rows=records)
            )
            atomic_json(root / "_FAILED.json", failures[-1])
        else:
            atomic_json(
                root / "_SUCCESS.json",
                dict(
                    repo=entry["repo"],
                    revision=entry["revision"],
                    schema_version=2,
                    rows=records,
                    shards=count,
                    source_files=len(entry["files"]),
                    code_policy_version=entry.get("code_policy_version"),
                    excluded_noncode=excluded_noncode,
                    excluded_unknown=excluded_unknown,
                ),
            )
            (root / "_FAILED.json").unlink(missing_ok=True)
    atomic_json(
        args.output / "run_summary.json",
        dict(
            failures=failures,
            unresolved=catalog["unresolved"],
            datasets=len(catalog["datasets"]),
            complete=len(catalog["datasets"]) - len(failures),
        ),
    )
    if failures or catalog["unresolved"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
