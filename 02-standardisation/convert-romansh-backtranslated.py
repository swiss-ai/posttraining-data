#!/usr/bin/env python3
"""
Convert jvamvas/apertus-pretrain-romansh-backtranslated-sft to unified chat format.

Source (HuggingFace Hub, train split only, 30,000 rows):
  prompt:     JSON string — [{role: "user",      content: "Übersetze..."}]
  completion: JSON string — [{role: "assistant", content: "...Romansh..."}]
  variety:    str — one of 6 Romansh varieties
              (rm-rumgr, rm-puter, rm-sutsilv, rm-vallader, rm-sursilv, rm-surmiran)
              5,000 rows each

Target: unified chat format.
"""

import sys
import json
import hashlib
import argparse
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Dict, Optional

from datasets import Dataset, DatasetDict, load_dataset

SRC = "jvamvas/apertus-pretrain-romansh-backtranslated-sft"


def convert_sample(sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert a single row to the unified chat format."""
    try:
        prompt_msgs = json.loads(sample["prompt"])
        completion_msgs = json.loads(sample["completion"])
    except (json.JSONDecodeError, TypeError):
        return None

    if not prompt_msgs or not completion_msgs:
        return None

    user_content = prompt_msgs[0].get("content", "")
    assistant_content = completion_msgs[0].get("content", "")

    if not user_content or not assistant_content:
        return None

    conversation_id = hashlib.sha256(user_content.encode()).hexdigest()[:16]

    return {
        "conversation_id": conversation_id,
        "dataset_source": SRC,
        "original_metadata": {
            "variety": sample.get("variety", ""),
        },
        "created_timestamp": datetime.now(UTC).isoformat(),
        "system_prompt": {"content": "", "metadata": {}},
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
        "script": "convert-romansh-backtranslated.py",
        "timestamp": datetime.now(UTC).isoformat(),
        "input_path": SRC,
        "output_path": str(output_path),
        "limit": args.limit,
        "description": f"Converted {SRC} to unified chat format",
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
            "conversation_type": "translation_sft",
            "varieties": ["rm-rumgr", "rm-puter", "rm-sutsilv",
                          "rm-vallader", "rm-sursilv", "rm-surmiran"],
            "format": "new_chat_format_with_parts",
        }

    with open(output_path / "dataset_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Dataset saved to {output_path}")


def cli():
    p = argparse.ArgumentParser(description=f"Convert {SRC} to unified chat format")
    p.add_argument("-o", "--output", required=True, help="Output directory path")
    p.add_argument("--num-proc", type=int, default=8,
                   help="Number of processes for dataset operations")
    p.add_argument("--limit", type=int, default=None,
                   help="Limit number of samples to process")
    return p.parse_args()


def main():
    args = cli()
    output_path = Path(args.output)

    if output_path.exists():
        response = input(f"{output_path} exists. Overwrite? [y/N]: ")
        if response.lower() != "y":
            sys.exit(0)

    print(f"Loading {SRC} from HuggingFace Hub ...")
    data = load_dataset(SRC, split="train")
    print(f"Loaded {len(data):,} samples")

    if args.limit and args.limit > 0:
        data = data.select(range(min(args.limit, len(data))))
        print(f"Limited to {len(data):,} samples")

    print("Converting samples...")
    converted_samples = []
    skipped = 0
    for i, sample in enumerate(data):
        if i % 5000 == 0:
            print(f"  {i}/{len(data)}")
        result = convert_sample(sample)
        if result is None:
            skipped += 1
        else:
            converted_samples.append(result)

    print(f"Converted: {len(converted_samples):,}  |  Skipped: {skipped}")

    print("Creating DatasetDict...")
    dataset_dict = DatasetDict({"train": Dataset.from_list(converted_samples)})

    save_dataset_and_metadata(dataset_dict, output_path, args)
    print("Done!")


if __name__ == "__main__":
    main()
