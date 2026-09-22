#!/usr/bin/env python3
"""
Remove linearised SFT samples with empty assistant responses.

For linearised datasets, assistant responses normally live in
messages[*].content.blocks where block["type"] == "response". This script
removes rows where no assistant response block contains non-whitespace text.
Rows with no assistant message are also removed.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.ipc as ipc
from datasets import DatasetDict, load_from_disk
from tqdm import tqdm


DEFAULT_INPUT = (
    "/capstor/store/cscs/swissai/infra01/reasoning/data/sft_1.1/"
    "mixtures/v1p5-mix-v1-26-04-cleaned"
)


@dataclass(frozen=True)
class EmptyAssistantMatch:
    split: str
    split_index: int
    shard: str
    conversation_id: str
    dataset_source: str
    reason: str
    user_snippet: str
    assistant_snippet: str


def compact(text: str, limit: int = 700) -> str:
    return re.sub(r"\s+", " ", text).strip()[:limit]


def content_text(content: dict[str, Any] | None) -> str:
    if not content:
        return ""

    chunks: list[str] = []
    if content.get("text"):
        chunks.append(content["text"])

    for part in content.get("parts") or []:
        if part and part.get("text"):
            chunks.append(part["text"])

    for block in content.get("blocks") or []:
        if block and block.get("text"):
            chunks.append(block["text"])

    return "\n".join(chunks)


def message_text(messages: list[dict[str, Any]], role: str) -> str:
    return "\n".join(
        content_text(message.get("content"))
        for message in messages
        if message.get("role") == role
    )


def assistant_response_texts(messages: list[dict[str, Any]]) -> list[str]:
    response_texts: list[str] = []

    for message in messages:
        if message.get("role") != "assistant":
            continue

        content = message.get("content") or {}
        blocks = content.get("blocks") or []

        for block in blocks:
            if block and block.get("type") == "response":
                response_texts.append(block.get("text") or "")

        # Robust fallback for non-standard linearised rows.
        if not blocks and content.get("text"):
            response_texts.append(content["text"])

    return response_texts


def row_match(
    row: dict[str, Any],
    split: str,
    split_index: int,
    shard: str,
) -> EmptyAssistantMatch | None:
    messages = row.get("messages") or []
    has_assistant = any(message.get("role") == "assistant" for message in messages)
    response_texts = assistant_response_texts(messages)

    if has_assistant and any(text.strip() for text in response_texts):
        return None

    reason = "no_assistant_message" if not has_assistant else "empty_assistant_response"
    return EmptyAssistantMatch(
        split=split,
        split_index=split_index,
        shard=shard,
        conversation_id=row.get("conversation_id") or "",
        dataset_source=row.get("dataset_source") or "",
        reason=reason,
        user_snippet=compact(message_text(messages, "user"), 1000),
        assistant_snippet=compact(message_text(messages, "assistant"), 1000),
    )


def dataset_splits(input_path: Path) -> list[str]:
    dict_file = input_path / "dataset_dict.json"
    if dict_file.exists():
        with dict_file.open() as f:
            data = json.load(f)
        return list(data["splits"])

    return [path.name for path in input_path.iterdir() if path.is_dir()]


def data_shards(input_path: Path, split: str) -> list[Path]:
    split_dir = input_path / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    return sorted(split_dir.glob("data-*-of-*.arrow"))


def find_matches(
    input_path: Path,
    audit_jsonl: Path | None,
) -> dict[str, set[int]]:
    bad_indices_by_split: dict[str, set[int]] = {}
    audit_handle = audit_jsonl.open("w") if audit_jsonl else None

    try:
        for split in dataset_splits(input_path):
            bad_indices_by_split[split] = set()
            split_index = 0

            for shard in tqdm(data_shards(input_path, split), desc=f"Scanning {split}", unit="shard"):
                with pa.memory_map(str(shard), "r") as source:
                    reader = ipc.open_stream(source)
                    for batch in reader:
                        for row in batch.to_pylist():
                            match = row_match(row, split, split_index, shard.name)
                            if match:
                                bad_indices_by_split[split].add(split_index)
                                if audit_handle:
                                    audit_handle.write(
                                        json.dumps(asdict(match), ensure_ascii=False)
                                        + "\n"
                                    )
                            split_index += 1
    finally:
        if audit_handle:
            audit_handle.close()

    return bad_indices_by_split


def load_metadata(input_path: Path) -> dict[str, Any]:
    metadata_path = input_path / "dataset_metadata.json"
    if not metadata_path.exists():
        return {}
    with metadata_path.open() as f:
        return json.load(f)


def save_metadata(
    input_path: Path,
    output_path: Path,
    removed_counts: dict[str, int],
    kept_counts: dict[str, int],
    audit_jsonl: Path | None,
) -> None:
    original = load_metadata(input_path)
    entry = {
        "operation": "clean_empty_assistant_responses",
        "script": "clean-empty-assistant-responses.py",
        "timestamp": datetime.now().isoformat(),
        "input_path": str(input_path),
        "output_path": str(output_path),
        "removed_counts": removed_counts,
        "kept_counts": kept_counts,
        "audit_jsonl": str(audit_jsonl) if audit_jsonl else None,
    }
    metadata = {
        **original,
        "processing_log": original.get("processing_log", []) + [entry],
    }
    with (output_path / "dataset_metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)


def write_clean_dataset(
    input_path: Path,
    output_path: Path,
    bad_indices_by_split: dict[str, set[int]],
    audit_jsonl: Path | None,
) -> None:
    print(f"Loading dataset from {input_path}")
    dataset = load_from_disk(str(input_path))
    if not isinstance(dataset, DatasetDict):
        dataset = DatasetDict({"train": dataset})

    clean_splits = {}
    removed_counts: dict[str, int] = {}
    kept_counts: dict[str, int] = {}

    for split, split_dataset in dataset.items():
        bad_indices = bad_indices_by_split.get(split, set())
        removed_counts[split] = len(bad_indices)
        keep_indices = [
            index for index in range(len(split_dataset)) if index not in bad_indices
        ]
        kept_counts[split] = len(keep_indices)
        print(
            f"{split}: keeping {len(keep_indices):,} / {len(split_dataset):,} "
            f"rows; removing {len(bad_indices):,}"
        )
        clean_splits[split] = split_dataset.select(keep_indices)

    output = DatasetDict(clean_splits)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving cleaned dataset to {output_path}")
    output.save_to_disk(str(output_path))
    save_metadata(input_path, output_path, removed_counts, kept_counts, audit_jsonl)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean samples with empty assistant responses."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path(DEFAULT_INPUT),
        help="Input HuggingFace dataset path.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        help="Output path for the cleaned dataset. Required unless --dry-run is set.",
    )
    parser.add_argument(
        "--audit-jsonl",
        type=Path,
        help=(
            "Path for removed-row audit JSONL. Defaults to "
            "<output-path>.empty_assistant_responses.jsonl, or a file in cwd for "
            "--dry-run."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only scan and write the audit file; do not save a cleaned dataset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input_path.resolve()

    if not args.dry_run and args.output_path is None:
        raise SystemExit("--output-path is required unless --dry-run is set")

    output_path = args.output_path.resolve() if args.output_path else None
    audit_jsonl = args.audit_jsonl
    if audit_jsonl is None:
        if output_path:
            audit_jsonl = output_path.with_name(
                f"{output_path.name}.empty_assistant_responses.jsonl"
            )
        else:
            audit_jsonl = Path("empty_assistant_responses.jsonl")
    audit_jsonl = audit_jsonl.resolve()
    audit_jsonl.parent.mkdir(parents=True, exist_ok=True)

    print(f"Input: {input_path}")
    print(f"Audit JSONL: {audit_jsonl}")
    bad_indices_by_split = find_matches(input_path, audit_jsonl)

    removed_counts = {
        split: len(indices) for split, indices in bad_indices_by_split.items()
    }
    print(f"Matched rows by split: {removed_counts}")
    print(f"Total matched rows: {sum(removed_counts.values()):,}")

    if args.dry_run:
        print("Dry run complete; no dataset was written.")
        return

    assert output_path is not None
    write_clean_dataset(input_path, output_path, bad_indices_by_split, audit_jsonl)


if __name__ == "__main__":
    main()
