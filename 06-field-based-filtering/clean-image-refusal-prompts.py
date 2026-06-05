#!/usr/bin/env python3
"""
Remove image-related prompts where the assistant refuses image access.

This is intended for linearised SFT datasets with a HuggingFace on-disk layout
and a `messages` column in the Apertus chat format.

The default `broad-114` mode matches the broad scan used to find the 114
candidate rows in v1p5-mix-v1-21-04-linearised:

  user mentions image/photo/picture/etc.
  AND assistant contains a refusal/capability phrase about viewing/accessing
      images, photos, screenshots, visuals, attachments, or files.

Use `strict-existing-image` if you only want clearer cases where the user asks
about an existing image/photo/picture and the assistant says it cannot view or
analyze images.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.ipc as ipc
from datasets import DatasetDict, load_from_disk
from tqdm import tqdm


DEFAULT_INPUT = (
    "/capstor/store/cscs/swissai/infra01/reasoning/data/sft_1.1/"
    "mixtures/v1p5-mix-v1-21-04-linearised"
)


@dataclass(frozen=True)
class Match:
    split: str
    split_index: int
    shard: str
    conversation_id: str
    dataset_source: str
    user_hit: str
    assistant_hit: str
    user_snippet: str
    assistant_snippet: str


def compact(text: str, limit: int = 700) -> str:
    return re.sub(r"\s+", " ", text).strip()[:limit]


def content_text(content: dict[str, Any] | None) -> str:
    """Extract text from all known linearised content fields."""
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


def broad_patterns() -> tuple[list[re.Pattern[str]], list[re.Pattern[str]]]:
    user_patterns = [
        re.compile(
            r"\b(image|picture|photo|photograph|screenshot|diagram|chart|figure|"
            r"visual|attached|uploaded|provided|shown|above|look at|see this|"
            r"describe this)\b",
            re.I,
        )
    ]
    assistant_patterns = [
        re.compile(
            r"\b(i\s+)?(can('|no)t|can not|am unable to|(?:do not|don't) have "
            r"(?:the )?ability to|(?:do not|don't) have access to|am not able to)"
            r"\b.{0,180}\b(see|view|access|analy[sz]e|inspect|look at|process|"
            r"interpret|describe)\b.{0,180}\b(image|picture|photo|screenshot|"
            r"visual|attachment|file|uploaded)",
            re.I | re.S,
        ),
        re.compile(
            r"\b(as an? (?:ai|language model|text[- ]based ai))\b.{0,220}"
            r"\b(see|view|access|analy[sz]e|inspect|look at|process|interpret|"
            r"describe)\b.{0,220}\b(image|picture|photo|screenshot|visual|"
            r"attachment|uploaded)",
            re.I | re.S,
        ),
        re.compile(
            r"\b(no image (?:was )?(?:provided|attached|uploaded)|(?:there is|"
            r"there's) no image|i (?:do not|don't) see an? image|please "
            r"(?:upload|attach|provide) (?:the |an? )?(?:image|picture|photo|"
            r"screenshot))\b",
            re.I | re.S,
        ),
    ]
    return user_patterns, assistant_patterns


def strict_patterns() -> tuple[list[re.Pattern[str]], list[re.Pattern[str]]]:
    user_patterns = [
        re.compile(
            r"\b(from|in|on|based on|using|about)\s+(this|the|my|provided|"
            r"attached|uploaded|given|above|below)\s+(image|picture|photo|"
            r"photograph|screenshot)\b",
            re.I,
        ),
        re.compile(
            r"\b(this|the|my|provided|attached|uploaded|given|above|below)\s+"
            r"(image|picture|photo|photograph|screenshot)\b",
            re.I,
        ),
        re.compile(
            r"\b(describe|identify|analy[sz]e|interpret|caption|explain|tell me|"
            r"what(?:\s+is|\s+are|\s+can you see)?)\b.{0,120}\b(image|"
            r"picture|photo|photograph|screenshot)\b",
            re.I | re.S,
        ),
        re.compile(
            r"\b(image|picture|photo|photograph|screenshot)\b.{0,120}\b"
            r"(describe|identify|analy[sz]e|interpret|caption|explain|tell me|"
            r"what(?:\s+is|\s+are|\s+can you see)?)\b",
            re.I | re.S,
        ),
    ]
    assistant_patterns = [
        re.compile(
            r"\b(i\s+)?(?:can('|no)t|can not|am unable to|(?:do not|don't) have "
            r"(?:the )?ability to|am not able to)\b.{0,120}\b(?:see|view|"
            r"access|analy[sz]e|inspect|look at|process|interpret|describe)"
            r"\b.{0,120}\b(?:image|images|picture|pictures|photo|photos|"
            r"photograph|screenshot|visual)",
            re.I | re.S,
        ),
        re.compile(
            r"\b(?:image|images|picture|pictures|photo|photos|photograph|"
            r"screenshot|visual)\b.{0,120}\b(?:can('|no)t|can not|unable to|"
            r"not able to)\b.{0,120}\b(?:see|view|access|analy[sz]e|inspect|"
            r"look at|process|interpret|describe)",
            re.I | re.S,
        ),
        re.compile(
            r"\b(?:unable to view images directly|cannot see the (?:image|"
            r"picture|photo)|can't see the (?:image|picture|photo)|can't view "
            r"images|cannot view images|don't have eyes to see)\b",
            re.I,
        ),
    ]
    return user_patterns, assistant_patterns


def first_match(patterns: Iterable[re.Pattern[str]], text: str) -> str | None:
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            return match.group(0)
    return None


def row_match(
    row: dict[str, Any],
    split: str,
    split_index: int,
    shard: str,
    user_patterns: list[re.Pattern[str]],
    assistant_patterns: list[re.Pattern[str]],
) -> Match | None:
    messages = row.get("messages") or []
    user_text = message_text(messages, "user")
    user_hit = first_match(user_patterns, user_text)
    if not user_hit:
        return None

    assistant_text = message_text(messages, "assistant")
    assistant_hit = first_match(assistant_patterns, assistant_text)
    if not assistant_hit:
        return None

    return Match(
        split=split,
        split_index=split_index,
        shard=shard,
        conversation_id=row.get("conversation_id") or "",
        dataset_source=row.get("dataset_source") or "",
        user_hit=compact(user_hit, 250),
        assistant_hit=compact(assistant_hit, 300),
        user_snippet=compact(user_text, 1000),
        assistant_snippet=compact(assistant_text, 1000),
    )


def data_shards(input_path: Path, split: str) -> list[Path]:
    split_dir = input_path / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    return sorted(split_dir.glob("data-*-of-*.arrow"))


def dataset_splits(input_path: Path) -> list[str]:
    dict_file = input_path / "dataset_dict.json"
    if dict_file.exists():
        with dict_file.open() as f:
            data = json.load(f)
        return list(data["splits"])

    return [path.name for path in input_path.iterdir() if path.is_dir()]


def find_matches(
    input_path: Path,
    mode: str,
    audit_jsonl: Path | None,
) -> dict[str, set[int]]:
    if mode == "broad-114":
        user_patterns, assistant_patterns = broad_patterns()
    elif mode == "strict-existing-image":
        user_patterns, assistant_patterns = strict_patterns()
    else:
        raise ValueError(f"Unknown mode: {mode}")

    bad_indices_by_split: dict[str, set[int]] = {}
    audit_handle = audit_jsonl.open("w") if audit_jsonl else None

    try:
        for split in dataset_splits(input_path):
            bad_indices_by_split[split] = set()
            split_index = 0
            shards = data_shards(input_path, split)

            for shard in tqdm(shards, desc=f"Scanning {split}", unit="shard"):
                with pa.memory_map(str(shard), "r") as source:
                    reader = ipc.open_stream(source)
                    for batch in reader:
                        rows = batch.to_pylist()
                        for row in rows:
                            match = row_match(
                                row=row,
                                split=split,
                                split_index=split_index,
                                shard=shard.name,
                                user_patterns=user_patterns,
                                assistant_patterns=assistant_patterns,
                            )
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
    mode: str,
    removed_counts: dict[str, int],
    kept_counts: dict[str, int],
    audit_jsonl: Path | None,
) -> None:
    original = load_metadata(input_path)
    entry = {
        "operation": "clean_image_refusal_prompts",
        "script": "clean-image-refusal-prompts.py",
        "timestamp": datetime.now().isoformat(),
        "input_path": str(input_path),
        "output_path": str(output_path),
        "mode": mode,
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
    mode: str,
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
    save_metadata(
        input_path=input_path,
        output_path=output_path,
        mode=mode,
        removed_counts=removed_counts,
        kept_counts=kept_counts,
        audit_jsonl=audit_jsonl,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean image-access refusal prompts from a linearised SFT dataset."
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
            "<output-path>/removed_image_refusal_prompts.jsonl, or a file in cwd for "
            "--dry-run."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["broad-114", "strict-existing-image"],
        default="broad-114",
        help="Matching mode. Default reproduces the broad 114-row candidate scan.",
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
                f"{output_path.name}.removed_image_refusal_prompts.jsonl"
            )
        else:
            audit_jsonl = Path(f"removed_image_refusal_prompts.{args.mode}.jsonl")
    audit_jsonl = audit_jsonl.resolve()
    audit_jsonl.parent.mkdir(parents=True, exist_ok=True)

    print(f"Input: {input_path}")
    print(f"Mode: {args.mode}")
    print(f"Audit JSONL: {audit_jsonl}")

    bad_indices_by_split = find_matches(
        input_path=input_path,
        mode=args.mode,
        audit_jsonl=audit_jsonl,
    )
    removed_counts = {
        split: len(indices) for split, indices in bad_indices_by_split.items()
    }
    total_removed = sum(removed_counts.values())
    print(f"Matched rows by split: {removed_counts}")
    print(f"Total matched rows: {total_removed:,}")

    if args.dry_run:
        print("Dry run complete; no dataset was written.")
        return

    assert output_path is not None
    write_clean_dataset(
        input_path=input_path,
        output_path=output_path,
        bad_indices_by_split=bad_indices_by_split,
        mode=args.mode,
        audit_jsonl=audit_jsonl,
    )


if __name__ == "__main__":
    main()
