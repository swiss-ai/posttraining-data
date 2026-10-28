"""Standardise saved HF datasets, JSONL or Parquet into checked Apertus 2 native rows."""

from collections import Counter
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import DatasetDict, load_from_disk
from apertus_common import load_encoding

from chat_review import unresolved
from native_format import native_conversation
from openai_chat import convert, decode
from openai_training_payload import render_reasoning
from openai_validation import check_result
from standardize import atomic_json, dumps
from unified import SCHEMA as SOURCE_SCHEMA, normalize_v2
from unify_local import records, rename

SCHEMA = pa.schema(
    list(SOURCE_SCHEMA)
    + [
        ("dataset_source", pa.string()),
        ("conversation_json", pa.large_string()),
        ("native_text", pa.large_string()),
        ("provenance_json", pa.large_string()),
        ("tokens", pa.list_(pa.int64())),
        ("loss_weights", pa.list_(pa.float32())),
    ],
    metadata={b"schema_name": b"apertus2_native_code_sft", b"schema_version": b"2"},
)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def snapshot(path):
    if path.is_file():
        paths = [path]
    else:
        paths = []
        for root, dirs, files in os.walk(path):
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            paths.extend(
                Path(root) / f
                for f in files
                if f.endswith(
                    (".json", ".arrow", ".parquet", ".txt", ".model", ".tiktoken")
                )
            )
    return [
        dict(
            path=str(p.resolve()), size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns
        )
        for p in sorted(paths)
    ]


def sources(paths, split):
    for path in paths:
        if path.is_dir():
            dataset = load_from_disk(str(path))
            if isinstance(dataset, DatasetDict):
                if split is None and len(dataset) != 1:
                    raise ValueError(
                        "Saved DatasetDict has multiple splits; specify --split explicitly"
                    )
                name = split or next(iter(dataset))
                if name not in dataset:
                    raise ValueError(f"Missing split: {name}")
                yield path, name, iter(dataset[name])
            else:
                yield path, split or str(dataset.split or "train"), iter(dataset)
        else:
            yield path, split or "train", records(path)


def checked_row(original, entry, spec, index, mapping, convention, encoding=None):
    mapped = rename(original, mapping)
    if entry["repo"] == "open-alchemy/code-alchemy":
        from code_alchemy import adapt

        mapped = adapt(mapped, spec["source_config"], entry["revision"])
    row = normalize_v2(mapped, entry, spec, index)
    row["source_record_json"] = dumps(original)
    # Preserve explicitly stored instruction fields that are outside the message list.
    messages = decode(row["messages_json"]) or []
    for role, fields in [
        ("system", ("system", "system_prompt")),
        ("developer", ("developer", "developer_prompt")),
    ]:
        instructions = []
        for field in fields:
            if mapped.get(field):
                value = mapped[field]
                if isinstance(value, dict):
                    value = value.get("content")
                if not isinstance(value, str):
                    raise ValueError(f"Unsupported instruction field: {field}")
                if value not in instructions:
                    instructions.append(value)
        if instructions:
            if len(instructions) != 1:
                raise ValueError(f"Conflicting {role} instruction fields")
            existing = [
                m.get("content")
                for m in messages
                if isinstance(m, dict) and m.get("role") == role
            ]
            if existing and existing != instructions:
                raise ValueError(
                    f"Conflicting {role} instructions in fields and messages"
                )
            if not existing:
                insertion = (
                    0
                    if role == "system"
                    else next(
                        (
                            i
                            for i, m in enumerate(messages)
                            if m.get("role") != "system"
                        ),
                        len(messages),
                    )
                )
                messages.insert(insertion, dict(role=role, content=instructions[0]))
    row["messages_json"] = dumps(messages)
    result = convert(
        dict(
            dataset=entry["repo"],
            stored_sample=row,
            decoded_messages=decode(row["messages_json"]),
            decoded_tools=decode(row["tools_json"]),
            source_shard=spec["path"],
            row_offset_in_shard=index,
        )
    )
    if result.get("status") != "passed_sample":
        raise ValueError(
            json.dumps(
                dict(status=result.get("status"), issues=result.get("issues", []))
            )
        )
    render_reasoning(result, "sidecar")
    errors = unresolved(result) + check_result(result)
    if errors:
        raise ValueError(json.dumps(errors))
    conversation, text = native_conversation(result, convention)
    payload = json.dumps(result, ensure_ascii=False, allow_nan=False)
    payload.encode("utf-8")
    row.update(
        dataset_source=entry["repo"],
        conversation_json=conversation.to_json(),
        native_text=text,
        provenance_json=payload,
        messages_json=dumps(result["chat"]["messages"]),
        tools_json=dumps(result["chat"].get("tools")),
        reasoning=dumps(result.get("reasoning_by_message"))
        if result.get("reasoning_by_message")
        else None,
        normalization_status="native_checked",
        normalization_notes="[]",
        tokens=None,
        loss_weights=None,
    )
    if encoding is not None:
        encoded = encoding.encode_conversation(conversation, return_loss_weights=True)
        row.update(tokens=encoded.token_ids, loss_weights=encoded.loss_weights)
    return row


def standardise(
    inputs,
    output,
    repo=None,
    revision=None,
    split=None,
    mapping=None,
    category="code_sft",
    convention="think-first",
    tokenizer=None,
    rows_per_shard=50000,
    source_config=None,
):
    inputs = [p.resolve() for p in inputs]
    output = output.resolve()
    mapping = mapping or {}
    if not inputs or any(not p.exists() for p in inputs):
        raise ValueError("Input paths must exist")
    if len(set(inputs)) != len(inputs):
        raise ValueError("Duplicate input paths are not allowed")
    if category not in {"code_sft", "code_agentic"}:
        raise ValueError("Unknown training category")
    if rows_per_shard < 1:
        raise ValueError("rows_per_shard must be positive")
    if convention not in {"think-first", "source"}:
        raise ValueError("Unknown thinking convention")
    if not isinstance(mapping, dict) or any(
        not isinstance(k, str) or not isinstance(v, str) for k, v in mapping.items()
    ):
        raise ValueError("Column map must contain string source and target names")
    for path in inputs:
        if output == path or output.is_relative_to(path) or path.is_relative_to(output):
            raise ValueError("Input and output trees must be separate")
    metadata = {}
    if (
        len(inputs) == 1
        and inputs[0].is_dir()
        and (inputs[0] / "dataset_metadata.json").exists()
    ):
        metadata = json.loads((inputs[0] / "dataset_metadata.json").read_text())
    repo = repo or metadata.get("dataset_name")
    if (
        not isinstance(repo, str)
        or len(repo.split("/")) != 2
        or any(p in {"", ".", ".."} for p in repo.split("/"))
    ):
        raise ValueError(
            "Supply --repo owner/name or downloader metadata containing dataset_name"
        )
    revision = revision or metadata.get("commit_hash") or "unrecorded"
    split = split or metadata.get("split")
    directory = output / category / repo
    entry = dict(
        repo=repo,
        revision=revision,
        category=category,
        license="unknown",
        license_status="unknown",
        license_notes="Retained source declaration in source_record_json",
        synthetic_status="unknown",
        synthetic_notes="Not inferred",
        metadata_source="existing downloader output",
        code_policy="code_source",
    )
    code_root = Path(__file__).parent
    distribution = importlib.metadata.distribution("apertus-common")
    config = dict(
        inputs=[s for p in inputs for s in snapshot(p)],
        entry=entry,
        split=split,
        mapping=mapping,
        convention=convention,
        source_config=source_config,
        rows_per_shard=rows_per_shard,
        code={
            str(p.relative_to(code_root)): digest(p)
            for p in sorted(code_root.rglob("*.py"))
            if "tests" not in p.parts
        },
        resources={p.name: digest(p) for p in sorted(code_root.glob("*.json"))},
        library=distribution.read_text("direct_url.json") or distribution.version,
        tokenizer=snapshot(tokenizer.resolve()) if tokenizer else None,
    )
    encoding = load_encoding(tokenizer) if tokenizer else None
    directory.mkdir(parents=True, exist_ok=True)
    import fcntl

    with (directory / ".run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run = directory / "run.json"
        if run.exists() and json.loads(run.read_text()) != config:
            raise ValueError(
                "Inputs, code or settings changed; choose a fresh output directory"
            )
        atomic_json(run, config)
        atomic_json(directory / "source.json", entry)
        (directory / "_SUCCESS.json").unlink(missing_ok=True)
        totals = Counter(read=0, written=0, quarantined=0)
        for path, name, iterator in sources(inputs, split):
            spec = dict(
                path=str(path),
                source_config=source_config or metadata.get("subset") or "local",
                source_split=name,
            )
            partition = directory / (
                "file="
                + hashlib.sha256((str(path) + "\0" + name).encode()).hexdigest()[:16]
            )
            marker = partition / "_SUCCESS.json"
            if marker.exists():
                saved = json.loads(marker.read_text())
                for filename, checksum in saved["sha256"].items():
                    if digest(partition / filename) != checksum:
                        raise ValueError("Output checksum mismatch")
                totals.update(saved["counts"])
                continue
            if partition.exists():
                shutil.rmtree(
                    partition
                )  # Restart only this run's unpublished partition.
            partition.mkdir()
            counts = Counter(read=0, written=0, quarantined=0)
            batch = []
            shards = 0

            def flush():
                nonlocal shards
                if not batch:
                    return
                pq.write_table(
                    pa.Table.from_pylist(batch, schema=SCHEMA),
                    partition / f"part-{shards:06d}.parquet",
                    compression="zstd",
                )
                batch.clear()
                shards += 1

            with (partition / "rejected.jsonl").open("w") as rejected:
                for index, original in enumerate(iterator):
                    counts["read"] += 1
                    try:
                        row = checked_row(
                            original, entry, spec, index, mapping, convention, encoding
                        )
                        # Ensure malformed Unicode and non-finite provenance never reach a completed shard.
                        pa.Table.from_pylist([row], schema=SCHEMA)
                    except Exception as error:
                        counts["quarantined"] += 1
                        rejected.write(
                            json.dumps(
                                dict(
                                    source=str(path),
                                    split=name,
                                    row=index,
                                    error=f"{type(error).__name__}: {error}",
                                    source_record=json.loads(dumps(original)),
                                ),
                                ensure_ascii=True,
                            )
                            + "\n"
                        )
                        continue
                    batch.append(row)
                    counts["written"] += 1
                    if len(batch) >= rows_per_shard:
                        flush()
                flush()
            if not shards:
                pq.write_table(
                    pa.Table.from_pylist([], schema=SCHEMA),
                    partition / "part-000000.parquet",
                )
            readback = sum(
                batch.num_rows
                for p in partition.glob("part-*.parquet")
                for batch in pq.ParquetFile(p).iter_batches()
            )
            if (
                readback != counts["written"]
                or counts["read"] != counts["written"] + counts["quarantined"]
            ):
                raise ValueError("Output row accounting mismatch")
            atomic_json(
                marker,
                dict(
                    counts=dict(counts),
                    sha256={
                        p.name: digest(p) for p in partition.iterdir() if p.is_file()
                    },
                ),
            )
            totals.update(counts)
        report = dict(
            rows=totals["written"],
            revision=revision,
            counts=dict(totals),
            format="apertus2-native",
            validation="full reference-library checks including training rules",
        )
        atomic_json(directory / "_SUCCESS.json", report)
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo")
    parser.add_argument("--revision")
    parser.add_argument("--split")
    parser.add_argument("--source-config", help="Original dataset configuration/subset")
    parser.add_argument("--column-map", type=Path)
    parser.add_argument(
        "--category", choices=["code_sft", "code_agentic"], default="code_sft"
    )
    parser.add_argument(
        "--convention", choices=["think-first", "source"], default="think-first"
    )
    parser.add_argument(
        "--tokenizer", type=Path, help="Local verified Apertus 2 tokenizer artifact"
    )
    parser.add_argument("--rows-per-shard", type=int, default=50000)
    args = parser.parse_args()
    mapping = json.loads(args.column_map.read_text()) if args.column_map else {}
    result = standardise(
        args.input,
        args.output,
        args.repo,
        args.revision,
        args.split,
        mapping,
        args.category,
        args.convention,
        args.tokenizer,
        args.rows_per_shard,
        args.source_config,
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
