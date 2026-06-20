#!/usr/bin/env python3
"""
Convert allenai/SciRIFF to the unified chat format.

Default source:
  load_dataset("allenai/SciRIFF", "8192", split="train")

Source columns:
  - input: prompt text
  - output: target response
  - metadata: task/source metadata
  - _instance_id: stable source row id
"""

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from datasets import Dataset, DatasetDict, load_dataset


DATASET_ID = "allenai/SciRIFF"
DEFAULT_SUBSET = "8192"
DEFAULT_SPLIT = "train"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_id(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.:-]+", "_", value).strip("_")


def make_part(content: str) -> Dict[str, Any]:
    return {
        "type": "response",
        "content": content,
        "metadata": {},
        "name": "",
        "args": "",
    }


def dataset_source_for(subset: str) -> str:
    return "allenai_SciRIFF_{}".format(safe_id(subset))


def conversation_id_for(row: Dict[str, Any], input_text: str, row_index: int, subset: str) -> str:
    instance_id = row.get("_instance_id")
    if instance_id:
        return "allenai_SciRIFF_{}_{}".format(safe_id(subset), safe_id(str(instance_id)))
    return "allenai_SciRIFF_{}_row_{}_{}".format(safe_id(subset), row_index, stable_id(input_text))


def convert_sample(row: Dict[str, Any], row_index: int, subset: str) -> Optional[Dict[str, Any]]:
    input_text = str(row.get("input") or "").strip()
    output_text = str(row.get("output") or "").strip()

    if not input_text or not output_text:
        return None

    original_metadata = {}
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        original_metadata.update(metadata)
    elif metadata is not None:
        original_metadata["metadata"] = metadata

    for key, value in row.items():
        if key not in {"input", "output", "metadata"}:
            original_metadata[key] = value

    return {
        "conversation_id": conversation_id_for(row, input_text, row_index, subset),
        "dataset_source": dataset_source_for(subset),
        "original_metadata": original_metadata,
        "created_timestamp": now_utc(),
        "system_prompt": {"content": "", "metadata": {}},
        "initial_prompt": {
            "role": "user",
            "content": input_text,
            "metadata": {},
        },
        "available_functions": [],
        "conversation_branches": [
            {
                "messages": [
                    {
                        "role": "assistant",
                        "parts": [make_part(output_text)],
                    },
                ],
            },
        ],
    }


def load_existing_metadata(output_path: Path) -> Optional[Dict[str, Any]]:
    metadata_file = output_path / "dataset_metadata.json"
    if not metadata_file.exists():
        return None
    try:
        with open(metadata_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def save_dataset_and_metadata(
    dataset_dict: DatasetDict,
    output_path: Path,
    args: argparse.Namespace,
    num_input_rows: int,
    num_skipped: int,
) -> None:
    output_path.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(output_path))

    metadata = load_existing_metadata(output_path) or {}
    processing_entry = {
        "operation": "convert_sciriff",
        "script": "convert-sciriff.py",
        "timestamp": now_utc(),
        "dataset_id": DATASET_ID,
        "subset": args.subset,
        "split": args.split,
        "output_path": str(output_path),
        "num_input_rows": num_input_rows,
        "num_output_rows": len(dataset_dict["train"]),
        "num_skipped_empty": num_skipped,
        "limit": args.limit,
        "description": "Converted allenai/SciRIFF input/output rows to unified chat format.",
    }

    metadata.setdefault("processing_log", []).append(processing_entry)
    metadata.setdefault("format", "chat_format_v1")
    metadata.setdefault("source_dataset", DATASET_ID)
    metadata.setdefault("source_subset", args.subset)
    metadata.setdefault("conversion_details", {
        "conversation_type": "scientific_instruction_response",
        "input_field": "input",
        "output_field": "output",
        "added_fields": ["system_prompt", "available_functions", "conversation_branches"],
        "format": "new_chat_format_with_parts",
    })

    with open(output_path / "dataset_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Dataset saved to {}".format(output_path))
    print("Metadata saved to {}".format(output_path / "dataset_metadata.json"))


def cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert allenai/SciRIFF to unified chat format")
    parser.add_argument("-o", "--output", required=True, help="Output directory path")
    parser.add_argument("--subset", default=DEFAULT_SUBSET, help="HF config/subset name")
    parser.add_argument("--split", default=DEFAULT_SPLIT, help="HF split name")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples to process")
    parser.add_argument("--overwrite", action="store_true", default=False, help="Overwrite output without prompting")
    return parser.parse_args()


def main() -> None:
    args = cli()
    output_path = Path(args.output)

    if output_path.exists() and not args.overwrite:
        response = input("{} exists. Overwrite? [y/N]: ".format(output_path))
        if response.lower() != "y":
            sys.exit(0)

    print("Loading {} subset={} split={} ...".format(DATASET_ID, args.subset, args.split), flush=True)
    data = load_dataset(DATASET_ID, args.subset, split=args.split)
    print("Loaded {:,} samples".format(len(data)), flush=True)

    if args.limit and args.limit > 0:
        data = data.select(range(min(args.limit, len(data))))
        print("Limited to {:,} samples".format(len(data)), flush=True)

    converted = []
    skipped = 0
    for i, row in enumerate(data):
        if i % 5000 == 0:
            print("Processing sample {}/{}".format(i, len(data)), flush=True)
        sample = convert_sample(row, i, args.subset)
        if sample is None:
            skipped += 1
        else:
            converted.append(sample)

    print("Converted {:,} samples; skipped {:,} empty input/output rows".format(len(converted), skipped), flush=True)
    dataset_dict = DatasetDict({"train": Dataset.from_list(converted)})
    save_dataset_and_metadata(dataset_dict, output_path, args, len(data), skipped)
    print("Conversion complete!", flush=True)


if __name__ == "__main__":
    main()
