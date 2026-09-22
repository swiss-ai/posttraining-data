#!/usr/bin/env python3
"""
Remove bad samples from a non-linearised standard SFT dataset.

This runs before `linearise-dataset.py` on datasets that still have
`initial_prompt` and `conversation_branches`. It removes:

- rows with no non-empty assistant response
- image/photo/screenshot prompts where the assistant refuses because it cannot
  see or access images
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from datasets import Dataset, DatasetDict, load_from_disk
from tqdm import tqdm


@dataclass(frozen=True)
class BadSampleMatch:
    split: str
    split_index: int
    conversation_id: str
    dataset_source: str
    reasons: list[str]
    user_hit: str
    assistant_hit: str
    user_snippet: str
    assistant_snippet: str


def compact(text: str, limit: int = 700) -> str:
    return re.sub(r"\s+", " ", text).strip()[:limit]


def parse_jsonish(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def any_text(value: Any) -> str:
    value = parse_jsonish(value)
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(any_text(item) for item in value if item is not None)
    if isinstance(value, dict):
        chunks: list[str] = []
        for key in ("text", "content", "output"):
            if value.get(key):
                chunks.append(any_text(value[key]))
        for part in value.get("parts") or []:
            chunks.append(any_text(part))
        for block in value.get("blocks") or []:
            chunks.append(any_text(block))
        if chunks:
            return "\n".join(chunk for chunk in chunks if chunk)
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def initial_prompt_text(row: dict[str, Any]) -> str:
    prompt = row.get("initial_prompt")
    if isinstance(prompt, dict):
        return any_text(prompt.get("content"))
    return any_text(prompt)


def branch_messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for branch in row.get("conversation_branches") or []:
        if isinstance(branch, dict):
            branch_items = branch.get("messages") or []
        elif isinstance(branch, list):
            branch_items = branch
        else:
            branch_items = []
        for message in branch_items:
            if isinstance(message, dict):
                messages.append(message)
    return messages


def part_text(part: dict[str, Any]) -> str:
    if part.get("content") is not None:
        return any_text(part.get("content"))
    if part.get("text") is not None:
        return any_text(part.get("text"))
    return any_text(part)


def message_all_text(message: dict[str, Any]) -> str:
    chunks: list[str] = []
    if message.get("content") is not None:
        chunks.append(any_text(message.get("content")))
    for part in message.get("parts") or []:
        if isinstance(part, dict):
            chunks.append(part_text(part))
    return "\n".join(chunk for chunk in chunks if chunk)


def role_text(row: dict[str, Any], role: str) -> str:
    chunks: list[str] = []
    if role == "user":
        chunks.append(initial_prompt_text(row))
    for message in branch_messages(row):
        if message.get("role") == role:
            chunks.append(message_all_text(message))
    return "\n".join(chunk for chunk in chunks if chunk)


def assistant_response_texts(row: dict[str, Any]) -> list[str]:
    response_texts: list[str] = []

    for message in branch_messages(row):
        if message.get("role") != "assistant":
            continue

        content = parse_jsonish(message.get("content"))
        if isinstance(content, dict):
            blocks = content.get("blocks") or []
            parts = content.get("parts") or []

            for block in blocks:
                if isinstance(block, dict) and block.get("type") == "response":
                    response_texts.append(any_text(block.get("text") or block.get("content")))
            for part in parts:
                if isinstance(part, dict) and part.get("type") == "response":
                    response_texts.append(any_text(part.get("text") or part.get("content")))
            if not blocks and not parts and content.get("text"):
                response_texts.append(any_text(content.get("text")))
        elif isinstance(content, str):
            response_texts.append(content)

        for part in message.get("parts") or []:
            if isinstance(part, dict) and part.get("type") == "response":
                response_texts.append(part_text(part))

    return response_texts


def has_tool_activity_part(part: dict[str, Any]) -> bool:
    part_type = part.get("type")
    if part_type in {"function-call", "tool_call", "tool_calls"}:
        return bool(
            any_text(part.get("name")).strip()
            or any_text(part.get("args")).strip()
            or any_text(part.get("arguments")).strip()
            or any_text(part.get("calls")).strip()
            or any_text(part.get("content")).strip()
        )
    if part_type in {"function-output", "tool_output", "tool_outputs"}:
        return bool(
            any_text(part.get("content")).strip()
            or any_text(part.get("output")).strip()
            or any_text(part.get("outputs")).strip()
        )
    return False


def has_tool_activity_block(block: dict[str, Any]) -> bool:
    block_type = block.get("type")
    if block_type in {"tool_calls", "function-call"}:
        calls = block.get("calls")
        if isinstance(calls, list):
            return any(any_text(call).strip() for call in calls)
        return bool(
            any_text(block.get("name")).strip()
            or any_text(block.get("arguments")).strip()
            or any_text(block.get("args")).strip()
        )
    if block_type in {"tool_outputs", "function-output"}:
        outputs = block.get("outputs")
        if isinstance(outputs, list):
            return any(any_text(output).strip() for output in outputs)
        return bool(
            any_text(block.get("output")).strip()
            or any_text(block.get("content")).strip()
        )
    return False


def has_nonempty_assistant_tool_activity(row: dict[str, Any]) -> bool:
    for message in branch_messages(row):
        if message.get("role") != "assistant":
            continue

        content = parse_jsonish(message.get("content"))
        if isinstance(content, dict):
            for block in content.get("blocks") or []:
                if isinstance(block, dict) and has_tool_activity_block(block):
                    return True
            for part in content.get("parts") or []:
                if isinstance(part, dict) and has_tool_activity_part(part):
                    return True

        for part in message.get("parts") or []:
            if isinstance(part, dict) and has_tool_activity_part(part):
                return True

    return False


def has_nonempty_assistant_response(row: dict[str, Any]) -> bool:
    return any(text.strip() for text in assistant_response_texts(row)) or has_nonempty_assistant_tool_activity(row)



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


def first_match(patterns: Iterable[re.Pattern[str]], text: str) -> str:
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            return match.group(0)
    return ""


def image_refusal_hit(
    row: dict[str, Any],
    user_patterns: list[re.Pattern[str]],
    assistant_patterns: list[re.Pattern[str]],
) -> tuple[str, str]:
    user_text = role_text(row, "user")
    user_hit = first_match(user_patterns, user_text)
    if not user_hit:
        return "", ""
    assistant_text = role_text(row, "assistant")
    assistant_hit = first_match(assistant_patterns, assistant_text)
    if not assistant_hit:
        return "", ""
    return user_hit, assistant_hit


def row_match(
    row: dict[str, Any],
    split: str,
    split_index: int,
    user_patterns: list[re.Pattern[str]],
    assistant_patterns: list[re.Pattern[str]],
    check_empty_assistant: bool,
    check_image_refusal: bool,
) -> BadSampleMatch | None:
    reasons: list[str] = []
    user_hit = ""
    assistant_hit = ""

    if check_empty_assistant and not has_nonempty_assistant_response(row):
        has_assistant = any(message.get("role") == "assistant" for message in branch_messages(row))
        reasons.append("no_assistant_message" if not has_assistant else "empty_assistant_response")

    if check_image_refusal:
        user_hit, assistant_hit = image_refusal_hit(row, user_patterns, assistant_patterns)
        if user_hit and assistant_hit:
            reasons.append("image_refusal")

    if not reasons:
        return None

    return BadSampleMatch(
        split=split,
        split_index=split_index,
        conversation_id=row.get("conversation_id") or "",
        dataset_source=row.get("dataset_source") or "",
        reasons=reasons,
        user_hit=compact(user_hit, 250),
        assistant_hit=compact(assistant_hit, 300),
        user_snippet=compact(role_text(row, "user"), 1000),
        assistant_snippet=compact(role_text(row, "assistant"), 1000),
    )


def load_dataset_dict(input_path: Path) -> DatasetDict:
    dataset = load_from_disk(str(input_path))
    if isinstance(dataset, DatasetDict):
        return dataset
    if isinstance(dataset, Dataset):
        return DatasetDict({"train": dataset})
    raise TypeError(f"Unsupported dataset type: {type(dataset)}")


def find_matches(
    dataset: DatasetDict,
    image_mode: str,
    audit_jsonl: Path | None,
    check_empty_assistant: bool,
    check_image_refusal: bool,
) -> dict[str, set[int]]:
    if image_mode in {"broad", "broad-114"}:
        user_patterns, assistant_patterns = broad_patterns()
    elif image_mode == "strict-existing-image":
        user_patterns, assistant_patterns = strict_patterns()
    else:
        raise ValueError(f"Unknown image mode: {image_mode}")

    bad_indices_by_split: dict[str, set[int]] = {}
    audit_handle = audit_jsonl.open("w", encoding="utf-8") if audit_jsonl else None

    try:
        for split, split_dataset in dataset.items():
            bad_indices_by_split[split] = set()
            for index, row in enumerate(tqdm(split_dataset, desc=f"Scanning {split}", unit="row")):
                match = row_match(
                    row=row,
                    split=split,
                    split_index=index,
                    user_patterns=user_patterns,
                    assistant_patterns=assistant_patterns,
                    check_empty_assistant=check_empty_assistant,
                    check_image_refusal=check_image_refusal,
                )
                if match:
                    bad_indices_by_split[split].add(index)
                    if audit_handle:
                        audit_handle.write(json.dumps(asdict(match), ensure_ascii=False) + "\n")
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
    image_mode: str,
    removed_counts: dict[str, int],
    kept_counts: dict[str, int],
    audit_jsonl: Path | None,
    check_empty_assistant: bool,
    check_image_refusal: bool,
) -> None:
    original = load_metadata(input_path)
    entry = {
        "operation": "clean_standard_sft_samples",
        "script": "clean-standard-sft-samples.py",
        "timestamp": datetime.now().isoformat(),
        "input_path": str(input_path),
        "output_path": str(output_path),
        "image_mode": image_mode,
        "check_empty_assistant": check_empty_assistant,
        "check_image_refusal": check_image_refusal,
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
    dataset: DatasetDict,
    input_path: Path,
    output_path: Path,
    bad_indices_by_split: dict[str, set[int]],
    image_mode: str,
    audit_jsonl: Path | None,
    check_empty_assistant: bool,
    check_image_refusal: bool,
) -> None:
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
            f"rows; removing {len(bad_indices):,}",
            flush=True,
        )
        clean_splits[split] = split_dataset.select(keep_indices)

    output = DatasetDict(clean_splits)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving cleaned dataset to {output_path}", flush=True)
    output.save_to_disk(str(output_path))
    save_metadata(
        input_path=input_path,
        output_path=output_path,
        image_mode=image_mode,
        removed_counts=removed_counts,
        kept_counts=kept_counts,
        audit_jsonl=audit_jsonl,
        check_empty_assistant=check_empty_assistant,
        check_image_refusal=check_image_refusal,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean bad rows from a standard, non-linearised SFT dataset."
    )
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, help="Required unless --dry-run is set.")
    parser.add_argument("--audit-jsonl", type=Path)
    parser.add_argument(
        "--image-mode",
        choices=["broad", "broad-114", "strict-existing-image"],
        default="broad",
    )
    parser.add_argument("--skip-empty-assistant", action="store_true")
    parser.add_argument("--skip-image-refusal", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input_path.resolve()
    output_path = args.output_path.resolve() if args.output_path else None

    if not args.dry_run and output_path is None:
        raise SystemExit("--output-path is required unless --dry-run is set")

    audit_jsonl = args.audit_jsonl
    if audit_jsonl is None:
        if output_path:
            audit_jsonl = output_path.with_name(f"{output_path.name}.removed_bad_samples.jsonl")
        else:
            audit_jsonl = Path("removed_bad_standard_samples.jsonl")
    audit_jsonl = audit_jsonl.resolve()
    audit_jsonl.parent.mkdir(parents=True, exist_ok=True)

    check_empty_assistant = not args.skip_empty_assistant
    check_image_refusal = not args.skip_image_refusal
    if not check_empty_assistant and not check_image_refusal:
        raise SystemExit("Nothing to clean: both checks were skipped.")

    print(f"Input: {input_path}", flush=True)
    print(f"Audit JSONL: {audit_jsonl}", flush=True)
    print(f"Image mode: {args.image_mode}", flush=True)
    print(f"Check empty assistant: {check_empty_assistant}", flush=True)
    print(f"Check image refusal: {check_image_refusal}", flush=True)

    dataset = load_dataset_dict(input_path)
    bad_indices_by_split = find_matches(
        dataset=dataset,
        image_mode=args.image_mode,
        audit_jsonl=audit_jsonl,
        check_empty_assistant=check_empty_assistant,
        check_image_refusal=check_image_refusal,
    )

    removed_counts = {split: len(indices) for split, indices in bad_indices_by_split.items()}
    print(f"Matched rows by split: {removed_counts}", flush=True)
    print(f"Total matched rows: {sum(removed_counts.values()):,}", flush=True)

    if args.dry_run:
        print("Dry run complete; no dataset was written.", flush=True)
        return

    assert output_path is not None
    write_clean_dataset(
        dataset=dataset,
        input_path=input_path,
        output_path=output_path,
        bad_indices_by_split=bad_indices_by_split,
        image_mode=args.image_mode,
        audit_jsonl=audit_jsonl,
        check_empty_assistant=check_empty_assistant,
        check_image_refusal=check_image_refusal,
    )


if __name__ == "__main__":
    main()
