#!/usr/bin/env python3
"""
Convert the nemotron-aug-no-reasoning HF dataset to the unified chat format.

Source (already an HF Dataset on disk):
  /capstor/.../dedup_pref_data_hf/nemotron-aug-no-reasoning

Source structure (every row is exactly [system, user, assistant]):
  conversations:  list of {role, content} dicts
  system_prompt:  str  — always "detailed thinking off"
  category:       str  — always "safety"
  license:        str
  reasoning:      str  — "off"
  generator:      str  — original generator model
  regen_model:    str  — regeneration model
  judge_safe:     bool — all True in this dataset
  judge_reason:   str
  used_in_training, version: str

Target: unified chat format with
  conversation_id, dataset_source, original_metadata, system_prompt,
  initial_prompt, available_functions, conversation_branches, created_timestamp
"""

import sys
import json
import hashlib
import argparse
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Dict, Optional

from datasets import Dataset, DatasetDict, load_from_disk

SRC = "nemotron-aug-no-reasoning"


def convert_sample(sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert a single row to the unified chat format."""
    conversations = sample["conversations"]  # list of {role, content}

    # Parse roles: expect [system?, user, assistant]
    system_content = ""
    user_content = None
    assistant_content = None

    for msg in conversations:
        role = msg.get("role", "")
        content = msg.get("content", "") or ""
        if role == "system" and not system_content:
            system_content = content
        elif role == "user" and user_content is None:
            user_content = content
        elif role == "assistant" and assistant_content is None:
            assistant_content = content

    if user_content is None or assistant_content is None:
        return None

    conversation_id = hashlib.sha256(user_content.encode()).hexdigest()[:16]

    return {
        "conversation_id": conversation_id,
        "dataset_source": SRC,
        "original_metadata": {
            "category":          sample.get("category", ""),
            "license":           sample.get("license", ""),
            "reasoning":         sample.get("reasoning", ""),
            "generator":         sample.get("generator", ""),
            "regen_model":       sample.get("regen_model", ""),
            "used_in_training":  sample.get("used_in_training", ""),
            "version":           sample.get("version", ""),
            "judge_safe":        sample.get("judge_safe", None),
            "judge_reason":      sample.get("judge_reason", ""),
        },
        "created_timestamp": datetime.now(UTC).isoformat(),
        "system_prompt": {"content": system_content, "metadata": {}},
        "initial_prompt": {
            "role": "user",
            "content": user_content,
            "metadata": {},
        },
        "available_functions": [],
        "conversation_branches": [
            {
                "messages": [
                    {
                        "role": "assistant",
                        "parts": [
                            {
                                "type": "response",
                                "content": assistant_content,
                                "metadata": {},
                            }
                        ],
                    }
                ]
            }
        ],
    }


def load_existing_metadata(output_path: Path) -> Optional[Dict[str, Any]]:
    meta_file = output_path / "dataset_metadata.json"
    if meta_file.exists():
        try:
            with open(meta_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return None


def save_dataset_and_metadata(dataset_dict: DatasetDict, output_path: Path,
                               args: argparse.Namespace):
    output_path.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(output_path))

    metadata = load_existing_metadata(output_path) or {}

    processing_entry = {
        "operation": f"convert_{SRC}",
        "script": f"convert-{SRC}.py",
        "timestamp": datetime.now(UTC).isoformat(),
        "input_path": args.input,
        "output_path": str(output_path),
        "limit": args.limit,
        "description": f"Converted {SRC} dataset to unified chat format",
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
            "conversation_type": "safety_sft",
            "added_fields": ["conversation_id", "system_prompt", "conversation_branches"],
            "format": "new_chat_format_with_parts",
        }

    with open(output_path / "dataset_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Dataset saved to {output_path}")


def cli():
    p = argparse.ArgumentParser(description=f"Convert {SRC} dataset to unified chat format")
    p.add_argument("-i", "--input", required=True,
                   help="Path to the HF dataset on disk (the nemotron-aug-no-reasoning folder)")
    p.add_argument("-o", "--output", required=True, help="Output directory path")
    p.add_argument("--limit", type=int, default=None, help="Limit number of samples to process")
    return p.parse_args()


def main():
    args = cli()
    output_path = Path(args.output)

    if output_path.exists():
        response = input(f"{output_path} exists. Overwrite? [y/N]: ")
        if response.lower() != "y":
            sys.exit(0)

    print(f"Loading dataset from {args.input} ...")
    data = load_from_disk(args.input)
    # Handle both Dataset and DatasetDict
    if hasattr(data, "keys"):
        data = data[list(data.keys())[0]]

    print(f"Loaded {len(data):,} samples")

    if args.limit and args.limit > 0:
        data = data.select(range(min(args.limit, len(data))))
        print(f"Limited to {len(data):,} samples")

    print("Converting samples...")
    converted_samples = []
    skipped = 0
    for i, sample in enumerate(data):
        if i % 1000 == 0:
            print(f"  {i}/{len(data)}")
        result = convert_sample(sample)
        if result is None:
            skipped += 1
        else:
            converted_samples.append(result)

    print(f"Converted: {len(converted_samples):,}  |  Skipped (bad structure): {skipped}")

    print("Creating DatasetDict...")
    dataset_dict = DatasetDict({"train": Dataset.from_list(converted_samples)})

    save_dataset_and_metadata(dataset_dict, output_path, args)
    print("Done!")


if __name__ == "__main__":
    main()
