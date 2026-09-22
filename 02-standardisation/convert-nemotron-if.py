"""
convert-nemotron-if.py
───────────────────────
Convert the nvidia/Nemotron-Instruction-Following-Chat-v1 dataset to the unified chat-tool schema.

The Nemotron-IF dataset contains multi-turn conversations structured as:
    - uuid: Unique identifier for the conversation
    - messages: List of message objects with role, content, and reasoning_content
    - license: License information (e.g., odc-by-1.0)
    - used_in: Tags indicating usage context (e.g., ["nano_v3"])
    - tools: List of available tools (usually empty)
    - reasoning: "on" or "off" indicating if reasoning is enabled
    - capability_target: "chat" or "instruction_following"

This converter:
1. Filters to only include samples with capability_target == "instruction_following"
2. Extracts system prompt from the first message if present
3. Converts multi-turn conversations to the parts-based format
4. Preserves reasoning_content as thought parts when present
5. Stores metadata about the conversation
"""

import sys
import json
import argparse
from datetime import datetime, UTC
from typing import Dict, Any, Optional, List
from pathlib import Path
from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets

SRC = "nvidia/Nemotron-Instruction-Following-Chat-v1"


def convert_sample(sample: Dict[str, Any], include_reasoning: bool = False) -> Dict[str, Any]:
    """Convert a single sample to the new format."""

    converted: Dict[str, Any] = {
        "conversation_id": sample.get("uuid", ""),
        "dataset_source": SRC,
        "original_metadata": {
            "license": sample.get("license", ""),
            "used_in": sample.get("used_in", []),
            "tools": sample.get("tools", []),
            "reasoning": sample.get("reasoning", ""),
            "capability_target": sample.get("capability_target", ""),
        },
        "created_timestamp": datetime.now(UTC).isoformat(),
    }

    messages = sample.get("messages", [])
    if not messages:
        return None

    # Extract system prompt if first message is system
    system_content = ""
    msg_start_idx = 0
    if messages and messages[0].get("role") == "system":
        system_content = messages[0].get("content", "") or ""
        msg_start_idx = 1

    converted["system_prompt"] = {
        "content": system_content,
        "metadata": {},
    }

    # Find the first user message for initial_prompt
    initial_prompt_content = ""
    initial_prompt_idx = msg_start_idx
    for i in range(msg_start_idx, len(messages)):
        if messages[i].get("role") == "user":
            initial_prompt_content = messages[i].get("content", "") or ""
            initial_prompt_idx = i
            break

    if not initial_prompt_content:
        return None

    converted["initial_prompt"] = {
        "role": "user",
        "content": initial_prompt_content,
        "metadata": {},
    }

    converted["available_functions"] = []

    # Process remaining messages into conversation branches
    branch_messages: List[Dict] = []

    for i in range(initial_prompt_idx + 1, len(messages)):
        msg = messages[i]
        role = msg.get("role", "")
        content = msg.get("content", "") or ""
        reasoning_content = msg.get("reasoning_content")

        if role not in ["user", "assistant"]:
            continue

        parts: List[Dict] = []

        # Add reasoning/thought if present and requested (typically for assistant messages)
        if include_reasoning and reasoning_content:
            parts.append({
                "type": "thought",
                "content": reasoning_content,
            })

        # Add the main response
        parts.append({
            "type": "response",
            "content": content,
            "metadata": {},
        })

        branch_messages.append({
            "role": role,
            "parts": parts,
        })

    converted["conversation_branches"] = [
        {
            "messages": branch_messages,
        }
    ]

    return converted


def load_existing_metadata(output_path: Path) -> Optional[Dict[str, Any]]:
    """Load existing dataset metadata if it exists."""
    meta_file = output_path / "dataset_metadata.json"
    if meta_file.exists():
        try:
            with open(meta_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return None


def save_dataset_and_metadata(dataset_dict: DatasetDict, output_path: Path, args: argparse.Namespace):
    """Save converted dataset with processing metadata."""
    output_path.mkdir(parents=True, exist_ok=True)

    dataset_dict.save_to_disk(str(output_path))

    metadata = load_existing_metadata(output_path) or {}

    processing_entry = {
        "operation": "convert_nemotron_if",
        "script": "convert-nemotron-if.py",
        "timestamp": datetime.now(UTC).isoformat(),
        "input_path": args.input,
        "output_path": str(output_path),
        "num_processes": args.num_proc,
        "limit": args.limit,
        "split": args.split,
        "include_reasoning": args.include_reasoning,
        "description": f"Converted {SRC} dataset (instruction_following only) to unified chat format"
    }

    if "processing_log" not in metadata:
        metadata["processing_log"] = []
    metadata["processing_log"].append(processing_entry)

    if "format" not in metadata:
        metadata["format"] = "chat_format_v1"
    if "source_dataset" not in metadata:
        metadata["source_dataset"] = SRC
    if "conversion_details" not in metadata:
        metadata["conversion_details"] = {
            "conversation_type": "multi_turn_chat",
            "added_fields": ["system_prompt", "conversation_branches"],
            "format": "new_chat_format_with_parts"
        }

    metadata_file = output_path / "dataset_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Dataset saved to {output_path}")
    print(f"Metadata saved to {metadata_file}")


def cli():
    p = argparse.ArgumentParser(description=f"Convert {SRC} dataset to unified chat format")
    p.add_argument("-i", "--input", default=None, help="Input file path. If None, will be loaded from the Hub.")
    p.add_argument("-o", "--output", required=True, help="Output directory path")
    p.add_argument("--split", default="all", choices=["all", "chat_if", "structured_outputs"],
                   help="Dataset split to convert (default: all)")
    p.add_argument("--num-proc", type=int, default=8, help="Number of processes for dataset operations")
    p.add_argument("--limit", type=int, default=None, help="Limit number of samples to process")
    p.add_argument("--include-reasoning", action="store_true", default=False,
                   help="Include reasoning_content as thought parts (default: False)")
    return p.parse_args()


def main():
    args = cli()
    output_path = Path(args.output)

    if output_path.exists():
        response = input(f"{output_path} exists. Overwrite? [y/N]: ")
        if response.lower() != "y":
            sys.exit(0)

    if args.input is None:
        if args.split == "all":
            print(f"Loading {SRC} (all splits) from HuggingFace Hub...")
            ds_chat = load_dataset(SRC, split="chat_if")
            ds_struct = load_dataset(SRC, split="structured_outputs")
            dataset = concatenate_datasets([ds_chat, ds_struct])
            print(f"Concatenated chat_if ({len(ds_chat)}) + structured_outputs ({len(ds_struct)})")
        else:
            print(f"Loading {SRC} ({args.split}) from HuggingFace Hub...")
            dataset = load_dataset(SRC, split=args.split)
    else:
        print(f"Loading data from {args.input}")
        try:
            with open(args.input, 'r') as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading input file: {e}")
            sys.exit(1)
        dataset = Dataset.from_list(data)

    print(f"Loaded {len(dataset)} samples")

    # Filter to only include instruction_following samples
    print("Filtering for capability_target == 'instruction_following'...")
    dataset = dataset.filter(
        lambda x: x.get("capability_target") == "instruction_following",
        num_proc=args.num_proc,
        desc="Filtering",
    )
    print(f"Filtered to {len(dataset)} instruction_following samples")

    if args.limit and args.limit > 0:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
        print(f"Limited to {len(dataset)} samples")

    print("Converting samples to new format...")
    if args.include_reasoning:
        print("Including reasoning_content as thought parts")

    def map_convert_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
        result = convert_sample(sample, include_reasoning=args.include_reasoning)
        if result is None:
            return {
                "conversation_id": "",
                "dataset_source": "",
                "original_metadata": {},
                "created_timestamp": "",
                "system_prompt": {"content": "", "metadata": {}},
                "initial_prompt": {"role": "", "content": "", "metadata": {}},
                "available_functions": [],
                "conversation_branches": [],
                "_valid": False,
            }
        result["_valid"] = True
        return result

    converted_data = dataset.map(
        map_convert_sample,
        num_proc=args.num_proc,
        remove_columns=dataset.column_names,
        desc="Converting",
    )

    # Filter out invalid samples
    converted_data = converted_data.filter(lambda x: x.get("_valid", True), num_proc=args.num_proc)
    converted_data = converted_data.remove_columns(["_valid"])

    dataset_dict = DatasetDict({"train": converted_data})

    print(f"Converted {len(converted_data)} samples")

    save_dataset_and_metadata(dataset_dict, output_path, args)

    print("Conversion complete!")


if __name__ == "__main__":
    main()
