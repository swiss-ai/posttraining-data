#!/usr/bin/env python3
"""Stream Hugging Face datasets into one stable, loss-preserving Parquet schema."""

from __future__ import annotations

import argparse
import base64
import fnmatch
import hashlib
import json
import os
import re
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq
import requests
import yaml
from datasets import get_dataset_config_names, load_dataset
from huggingface_hub import HfApi, hf_hub_url


SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("source_dataset", pa.string()),
        ("source_revision", pa.string()),
        ("source_config", pa.string()),
        ("source_split", pa.string()),
        ("source_row_index", pa.int64()),
        ("category", pa.string()),
        ("messages_json", pa.large_string()),
        ("prompt", pa.large_string()),
        ("response", pa.large_string()),
        ("reasoning", pa.large_string()),
        ("tools_json", pa.large_string()),
        ("tests_json", pa.large_string()),
        ("metadata_json", pa.large_string()),
    ],
    metadata={
        b"schema_name": b"swissai_hf_sft_standardized",
        b"schema_version": b"1",
        b"json_encoding": b"UTF-8 canonical JSON (sorted keys, compact separators)",
    },
)

MESSAGE_KEYS = ("messages", "conversations", "conversation", "trajectory")
PROMPT_KEYS = ("prompt", "instruction", "query", "question", "problem", "input")
RESPONSE_KEYS = ("response", "output", "answer", "solution", "completion")
REASONING_KEYS = ("reasoning_content", "reasoning", "thought", "analysis")
TOOLS_KEYS = ("tools", "functions", "available_tools")
TEST_KEYS = ("unit_tests", "tests", "test", "test_code", "test_cases")


def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return {"__bytes_base64__": base64.b64encode(value).decode("ascii")}
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        return jsonable(value.tolist())
    return str(value)


def dumps(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(
        jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def text_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return dumps(value)


def first_value(row: dict[str, Any], keys: tuple[str, ...]) -> tuple[str | None, Any]:
    for key in keys:
        if key in row and row[key] is not None:
            return key, row[key]
    return None, None


def normalized_messages(row: dict[str, Any]) -> tuple[str | None, set[str]]:
    key, raw = first_value(row, MESSAGE_KEYS)
    consumed: set[str] = set()
    if key:
        consumed.add(key)
        if isinstance(raw, list):
            messages = []
            for item in raw:
                if isinstance(item, dict):
                    msg = dict(item)
                    if "role" not in msg:
                        for role_key in ("from", "speaker", "author"):
                            if role_key in msg:
                                msg["role"] = msg[role_key]
                                break
                    if "content" not in msg:
                        for content_key in ("value", "text", "message", "observation"):
                            if content_key in msg:
                                msg["content"] = msg[content_key]
                                break
                    messages.append(msg)
                else:
                    messages.append({"content": item})
            return dumps(messages), consumed
        return dumps(raw), consumed

    prompt_key, prompt = first_value(row, PROMPT_KEYS)
    response_key, response = first_value(row, RESPONSE_KEYS)
    messages = []
    if prompt is not None:
        messages.append({"role": "user", "content": text_value(prompt)})
        consumed.add(prompt_key)  # type: ignore[arg-type]
    if response is not None:
        messages.append({"role": "assistant", "content": text_value(response)})
        consumed.add(response_key)  # type: ignore[arg-type]
    return (dumps(messages) if messages else None), consumed


def normalize(
    row: dict[str, Any],
    *,
    repo: str,
    revision: str,
    config: str,
    split: str,
    index: int,
    category: str,
) -> dict[str, Any]:
    messages_json, consumed = normalized_messages(row)
    prompt_key, prompt = first_value(row, PROMPT_KEYS)
    response_key, response = first_value(row, RESPONSE_KEYS)
    reasoning_key, reasoning = first_value(row, REASONING_KEYS)
    tools_key, tools = first_value(row, TOOLS_KEYS)
    tests_key, tests = first_value(row, TEST_KEYS)
    consumed.update(
        k for k in (prompt_key, response_key, reasoning_key, tools_key, tests_key) if k
    )

    source_identity = None
    for key in ("id", "uuid", "task_id", "instance_id", "problem_id"):
        if row.get(key) is not None:
            source_identity = f"{key}:{row[key]}"
            break
    identity = source_identity or f"row:{index}"
    if category == "benchmarks" and row.get("benchmark_source_file"):
        identity = f"file:{row['benchmark_source_file']}\0row:{index}"
    record_id = hashlib.sha256(
        f"{repo}\0{revision}\0{config}\0{split}\0{identity}".encode()
    ).hexdigest()
    metadata = {key: value for key, value in row.items() if key not in consumed}
    return {
        "id": record_id,
        "source_dataset": repo,
        "source_revision": revision,
        "source_config": config,
        "source_split": split,
        "source_row_index": index,
        "category": category,
        "messages_json": messages_json,
        "prompt": text_value(prompt),
        "response": text_value(response),
        "reasoning": text_value(reasoning),
        "tools_json": dumps(tools),
        "tests_json": dumps(tests),
        "metadata_json": dumps(metadata),
    }


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "default"


def atomic_json(path: Path, value: Any) -> None:
    temp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def manifest_entries(path: Path, category: str | None) -> list[dict[str, Any]]:
    document = yaml.safe_load(path.read_text())
    entries = document["datasets"]
    if category:
        entries = [entry for entry in entries if entry["category"] == category]
    return entries


def remote_data_files(
    entry: dict[str, Any], token: str | None
) -> list[tuple[str, str]]:
    patterns = entry["data_files"]
    if all(not any(char in pattern for char in "*?[") for pattern in patterns):
        selected = sorted(set(patterns))
    else:
        api = HfApi(token=token)
        files = api.list_repo_files(
            entry["repo"], repo_type="dataset", revision=entry["revision"], token=token
        )
        selected = sorted(
            {
                name
                for pattern in patterns
                for name in files
                if fnmatch.fnmatch(name, pattern)
            }
        )
    if not selected:
        raise RuntimeError(f"No repository files matched {patterns!r}")
    return [
        (
            name,
            hf_hub_url(
                entry["repo"], name, repo_type="dataset", revision=entry["revision"]
            ),
        )
        for name in selected
    ]


def raw_split_name(name: str) -> str:
    leaf = Path(name).name
    for suffix in (".jsonl.gz", ".json.gz", ".jsonl", ".json"):
        if leaf.endswith(suffix):
            leaf = leaf[: -len(suffix)]
            break
    return leaf.replace(".", "_").replace("_part_", "_part")


def iter_remote_jsonl(url: str, token: str | None) -> Iterable[dict[str, Any]]:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    with requests.get(url, headers=headers, stream=True, timeout=(30, 300)) as response:
        response.raise_for_status()
        for line_number, line in enumerate(response.iter_lines(), 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL at line {line_number} of {url}: {exc}"
                ) from exc
            if not isinstance(value, dict):
                value = {"value": value}
            yield value


def iter_sources(
    entry: dict[str, Any], token: str | None
) -> Iterable[tuple[str, str, Iterable]]:
    split_filter = set(entry.get("splits", []))
    if entry.get("data_files"):
        files = remote_data_files(entry, token)
        if entry.get("raw_jsonl"):
            for name, url in files:
                split = raw_split_name(name)
                if not split_filter or split in split_filter:
                    yield "default", split, iter_remote_jsonl(url, token)
            return
        urls = [url for _, url in files]
        suffixes = {Path(url.split("?")[0]).suffix.lower() for url in urls}
        loader = "parquet" if suffixes == {".parquet"} else "json"
        loaded = load_dataset(
            loader, data_files={"train": urls}, streaming=True, token=token
        )
        for split, dataset in loaded.items():
            if not split_filter or split in split_filter:
                yield "selected_files", split, dataset
        return

    configs = entry.get("configs")
    if not configs:
        configs = get_dataset_config_names(
            entry["repo"], revision=entry["revision"], token=token
        )
    for config in configs:
        loaded = load_dataset(
            entry["repo"],
            name=config,
            revision=entry["revision"],
            streaming=True,
            token=token,
        )
        if hasattr(loaded, "items"):
            sources = loaded.items()
        else:
            sources = [("train", loaded)]
        for split, dataset in sources:
            if not split_filter or split in split_filter:
                yield str(config), str(split), dataset


def write_partition(
    dataset: Iterable,
    partition: Path,
    entry: dict[str, Any],
    config: str,
    split: str,
    rows_per_shard: int,
) -> dict[str, Any]:
    success = partition / "_SUCCESS.json"
    if success.exists():
        return json.loads(success.read_text())
    partition.mkdir(parents=True, exist_ok=True)
    progress_path = partition / "_progress.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    resume_at = int(progress.get("rows", 0))
    shard_index = int(progress.get("next_shard", 0))
    rows: list[dict[str, Any]] = []
    total = resume_at
    started = time.time()

    def flush() -> None:
        nonlocal rows, shard_index
        if not rows:
            return
        output = partition / f"part-{shard_index:05d}.parquet"
        temp = output.with_suffix(".parquet.tmp")
        table = pa.Table.from_pylist(rows, schema=SCHEMA)
        pq.write_table(
            table, temp, compression="zstd", compression_level=6, use_dictionary=True
        )
        temp.replace(output)
        shard_index += 1
        rows = []
        atomic_json(progress_path, {"rows": total, "next_shard": shard_index})

    for index, row in enumerate(dataset):
        if index < resume_at:
            continue
        rows.append(
            normalize(
                dict(row),
                repo=entry["repo"],
                revision=entry["revision"],
                config=config,
                split=split,
                index=index,
                category=entry["category"],
            )
        )
        total = index + 1
        if len(rows) >= rows_per_shard:
            flush()
    flush()
    summary = {
        "dataset": entry["repo"],
        "revision": entry["revision"],
        "config": config,
        "split": split,
        "rows": total,
        "shards": shard_index,
        "elapsed_seconds": round(time.time() - started, 3),
        "schema_version": 1,
    }
    atomic_json(success, summary)
    progress_path.unlink(missing_ok=True)
    return summary


def run_entry(
    entry: dict[str, Any],
    output_root: Path,
    rows_per_shard: int,
    mark_repo_success: bool = True,
) -> None:
    token = os.environ.get("HF_TOKEN")
    repo_dir = output_root / entry["category"] / entry["repo"]
    repo_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(repo_dir / "source.json", entry)
    failure = repo_dir / "_FAILED.json"
    failure.unlink(missing_ok=True)
    summaries = []
    try:
        for config, split, dataset in iter_sources(entry, token):
            partition = (
                repo_dir / f"config={safe_name(config)}" / f"split={safe_name(split)}"
            )
            summaries.append(
                write_partition(
                    dataset, partition, entry, config, split, rows_per_shard
                )
            )
        if not summaries:
            raise RuntimeError("No config/split partitions selected")
        if mark_repo_success:
            atomic_json(
                repo_dir / "_SUCCESS.json",
                {
                    "dataset": entry["repo"],
                    "revision": entry["revision"],
                    "rows": sum(item["rows"] for item in summaries),
                    "partitions": summaries,
                },
            )
    except Exception as exc:
        atomic_json(
            failure,
            {
                "dataset": entry["repo"],
                "revision": entry["revision"],
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "gated_in_manifest": entry.get("gated", False),
            },
        )
        raise


def main() -> None:
    worker_threads = int(os.environ.get("DATASET_WORKER_THREADS", "1"))
    pa.set_cpu_count(worker_threads)
    pa.set_io_thread_count(max(2, worker_threads))
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--category")
    parser.add_argument("--index", type=int)
    parser.add_argument("--repo")
    parser.add_argument("--source-split")
    parser.add_argument("--source-file")
    parser.add_argument("--partition-only", action="store_true")
    parser.add_argument("--rows-per-shard", type=int, default=50_000)
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()
    entries = manifest_entries(args.manifest, args.category)
    if args.repo:
        entries = [entry for entry in entries if entry["repo"] == args.repo]
    if args.source_split:
        if len(entries) != 1:
            raise SystemExit("--source-split requires exactly one selected --repo")
        entries[0] = dict(entries[0])
        entries[0]["splits"] = [args.source_split]
    if args.source_file:
        if len(entries) != 1:
            raise SystemExit("--source-file requires exactly one selected --repo")
        entries[0] = dict(entries[0])
        entries[0]["data_files"] = [args.source_file]
    if args.index is not None:
        if not 0 <= args.index < len(entries):
            raise SystemExit(f"index {args.index} outside 0..{len(entries) - 1}")
        entries = [entries[args.index]]
    if not entries:
        raise SystemExit("No datasets selected")
    args.output.mkdir(parents=True, exist_ok=True)
    failed = []
    for entry in entries:
        print(f"START {entry['repo']} @ {entry['revision']}", flush=True)
        try:
            run_entry(
                entry,
                args.output,
                args.rows_per_shard,
                mark_repo_success=not args.partition_only,
            )
            print(f"DONE {entry['repo']}", flush=True)
        except Exception as exc:
            if not args.continue_on_error:
                raise
            failed.append({"dataset": entry["repo"], "error": str(exc)})
            print(f"FAILED {entry['repo']}: {exc}", file=sys.stderr, flush=True)
    if failed:
        atomic_json(
            args.output / f"_{safe_name(args.category or 'all')}_FAILURES.json", failed
        )
        print(
            f"Completed with {len(failed)} failed dataset(s)",
            file=sys.stderr,
            flush=True,
        )


if __name__ == "__main__":
    main()
