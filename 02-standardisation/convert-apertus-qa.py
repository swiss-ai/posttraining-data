#!/usr/bin/env python3
"""
Convert Apertus QA from its messages-style format to the standard SFT schema.

Input rows are expected to have:
  conversation_id, created_timestamp, dataset_source, original_metadata, messages

Output rows have:
  conversation_id, dataset_source, original_metadata, system_prompt,
  initial_prompt, available_functions, conversation_branches, created_timestamp
"""

import argparse
from collections import Counter
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import Dataset, DatasetDict, load_from_disk


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(as_text(item) for item in value if item is not None)
    if isinstance(value, dict):
        chunks: List[str] = []
        if value.get("text"):
            chunks.append(as_text(value["text"]))
        for part in value.get("parts") or []:
            if isinstance(part, dict):
                chunks.append(as_text(part.get("text") if part.get("text") is not None else part.get("content")))
        for block in value.get("blocks") or []:
            if isinstance(block, dict):
                chunks.append(as_text(block.get("text") if block.get("text") is not None else block.get("content")))
        return "\n".join(chunk for chunk in chunks if chunk)
    return str(value)


def make_part(part_type: str, content: str = "", name: str = "", args: str = "") -> Dict[str, Any]:
    return {
        "type": part_type,
        "content": content,
        "metadata": {},
        "name": name,
        "args": args,
    }


def tools_from_developer(content: Any) -> List[Dict[str, str]]:
    if not isinstance(content, dict):
        return []

    raw_tools = content.get("tools")
    if not raw_tools:
        return []
    if isinstance(raw_tools, str):
        try:
            raw_tools = json.loads(raw_tools)
        except json.JSONDecodeError:
            return []
    if not isinstance(raw_tools, list):
        return []

    tools: List[Dict[str, str]] = []
    for tool in raw_tools:
        if not isinstance(tool, dict):
            continue
        name = str(tool.get("name") or "")
        if not name:
            continue
        parameters = tool.get("parameters") or {}
        tools.append({
            "name": name,
            "description": str(tool.get("description") or ""),
            "parameters": json.dumps(parameters, ensure_ascii=False, sort_keys=True) if not isinstance(parameters, str) else parameters,
        })
    return tools


def assistant_parts_from_content(content: Any) -> List[Dict[str, Any]]:
    if isinstance(content, dict):
        blocks = content.get("blocks") or []
        if isinstance(blocks, list) and blocks:
            parts: List[Dict[str, Any]] = []
            for block in blocks:
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type") or "response"
                if block_type == "response":
                    text = as_text(block)
                    if text.strip():
                        parts.append(make_part("response", text))
                elif block_type == "reasoning":
                    text = as_text(block)
                    if text.strip():
                        parts.append(make_part("thought", text))
                elif block_type == "tool_calls":
                    for call in block.get("calls") or []:
                        if isinstance(call, dict):
                            parts.append(make_part("tool_calls", "", str(call.get("name") or ""), str(call.get("arguments") or "")))
                elif block_type == "tool_outputs":
                    for output in block.get("outputs") or []:
                        if isinstance(output, dict):
                            parts.append(make_part("tool_outputs", str(output.get("output") or ""), str(output.get("name") or "")))
                else:
                    text = as_text(block)
                    if text.strip():
                        parts.append(make_part(str(block_type), text))
            return parts

    text = as_text(content)
    if text.strip():
        return [make_part("response", text)]
    return []


def convert_row(row: Dict[str, Any], row_index: int) -> Optional[Dict[str, Any]]:
    messages = row.get("messages") or []
    if not isinstance(messages, list):
        return None

    system_chunks: List[str] = []
    initial_prompt: Optional[Dict[str, Any]] = None
    branch_messages: List[Dict[str, Any]] = []
    available_functions: List[Dict[str, str]] = []

    for message in messages:
        if not isinstance(message, dict):
            continue

        role = message.get("role")
        content = message.get("content")

        if role == "system":
            text = as_text(content).strip()
            if text:
                system_chunks.append(text)
        elif role == "developer":
            text = as_text(content).strip()
            if text:
                system_chunks.append(text)
            available_functions.extend(tools_from_developer(content))
        elif role == "user":
            text = as_text(content).strip()
            if initial_prompt is None:
                initial_prompt = {"role": "user", "content": text, "metadata": {}}
            else:
                branch_messages.append({
                    "role": "user",
                    "parts": [make_part("response", text)],
                })
        elif role == "assistant":
            parts = assistant_parts_from_content(content)
            if parts:
                branch_messages.append({"role": "assistant", "parts": parts})

    if initial_prompt is None:
        return None
    if not any(
        msg.get("role") == "assistant"
        and any(
            isinstance(part, dict)
            and part.get("type") == "response"
            and str(part.get("content") or "").strip()
            for part in msg.get("parts") or []
        )
        for msg in branch_messages
    ):
        return None

    original_metadata = row.get("original_metadata") if isinstance(row.get("original_metadata"), dict) else {}
    original_metadata = dict(original_metadata)
    original_metadata["source_row_index"] = row_index
    original_metadata["source_format"] = "messages"

    conversation_id = str(row.get("conversation_id") or "").strip() or f"apertus_qa_{row_index}"

    return {
        "conversation_id": conversation_id,
        "dataset_source": str(row.get("dataset_source") or "apertus_qa"),
        "original_metadata": original_metadata,
        "created_timestamp": str(row.get("created_timestamp") or now_iso()),
        "system_prompt": {"content": "\n\n".join(system_chunks), "metadata": {}},
        "initial_prompt": initial_prompt,
        "available_functions": available_functions,
        "conversation_branches": [{"messages": branch_messages}],
    }


def save_metadata(output_path: Path, args: argparse.Namespace, num_input: int, num_output: int, num_skipped: int) -> None:
    metadata = {
        "format": "chat_format_v1",
        "source_dataset": "apertus_qa",
        "processing_log": [
            {
                "operation": "convert_apertus_qa_messages_to_standard",
                "script": "convert-apertus-qa.py",
                "timestamp": now_iso(),
                "input_path": str(args.input),
                "output_path": str(args.output),
                "num_input_rows": num_input,
                "num_output_rows": num_output,
                "num_skipped": num_skipped,
            }
        ],
    }
    with open(output_path / "dataset_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Apertus QA messages format to standard SFT format")
    parser.add_argument("-i", "--input", required=True, help="Input HF dataset path")
    parser.add_argument("-o", "--output", required=True, help="Output HF dataset path")
    parser.add_argument("--overwrite", action="store_true", default=False, help="Overwrite output without prompting")
    return parser.parse_args()


def main() -> None:
    args = cli()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if output_path.exists() and not args.overwrite:
        response = input(f"{output_path} exists. Overwrite? [y/N]: ")
        if response.lower() != "y":
            sys.exit(0)

    loaded = load_from_disk(str(input_path))
    source = loaded["train"] if isinstance(loaded, DatasetDict) else loaded

    converted = []
    skipped = 0
    for idx, row in enumerate(source):
        out = convert_row(row, idx)
        if out is None:
            skipped += 1
        else:
            converted.append(out)

    id_counts = Counter(row["conversation_id"] for row in converted)
    seen_ids: Counter[str] = Counter()
    for row in converted:
        base_id = row["conversation_id"]
        if id_counts[base_id] > 1:
            source_row_index = row.get("original_metadata", {}).get("source_row_index", seen_ids[base_id])
            row["conversation_id"] = f"{base_id}_row{source_row_index}"
        seen_ids[base_id] += 1

    dataset_dict = DatasetDict({"train": Dataset.from_list(converted)})
    output_path.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(output_path))
    save_metadata(output_path, args, len(source), len(converted), skipped)

    print(f"Loaded {len(source):,} rows")
    print(f"Converted {len(converted):,} rows")
    print(f"Skipped {skipped:,} rows")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
