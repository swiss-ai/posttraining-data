"""
convert_gsm8k.py
───────────────────────
Convert the openai/gsm8k dataset from HuggingFace into the unified chat-tool schema.

The GSM8K dataset contains grade school math word problems with:
    - question: The math word problem
    - answer: Step-by-step solution ending with "#### {final_numerical_answer}"

This converter:
1. Extracts the question as the user prompt
2. Uses the full answer (with reasoning) as the assistant response
3. Extracts the final numerical answer for verification
4. Preserves the verifiable answer in standard format
"""

import sys
import json
import re
import argparse
from datetime import datetime, UTC
from typing import Dict, Any, Optional
from pathlib import Path
from datasets import Dataset, DatasetDict, load_dataset

SRC = "gsm8k"


ANSWERS_TOOL = {
    "name": "display_answers",
    "description": "Display the answers to the user",
    "parameters": {
        "type": "object",
        "properties": {
            "answers": {
                "type": "array",
                "items": {"type": "string"},
                "description": "The answers to the user",
            },
        },
        "required": ["answers"],
    },
}


def extract_answer_parts(answer: str) -> tuple[str, str]:
    """Extract reasoning and final answer from the answer string."""
    match = re.search(r"####\s*(.+?)$", answer, re.MULTILINE)
    if match:
        final_answer = match.group(1).strip().replace(",", "")
        reasoning = answer[:match.start()].strip()
        # Remove <<...>> content which contains internal calculations
        reasoning = re.sub(r"<<.*?>>", "", reasoning)
        return reasoning, final_answer
    
    # If no final answer marker, still strip calculations from the whole string
    reasoning = re.sub(r"<<.*?>>", "", answer).strip()
    return reasoning, ""


def convert_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a single sample to the new format."""

    converted: Dict[str, Any] = {
        "conversation_id": "",
        "dataset_source": SRC,
        "original_metadata": {},
        "created_timestamp": datetime.now(UTC).isoformat(),
    }

    converted["system_prompt"] = {
        "content": "",
        "metadata": {},
    }

    question = sample["question"]
    reasoning, final_answer = extract_answer_parts(sample["answer"])

    converted["initial_prompt"] = {
        "role": "user",
        "content": question,
        "metadata": {}
    }

    converted["available_functions"] = [ANSWERS_TOOL]

    parts: list[Dict] = [
        {
            "type": "response",
            "content": reasoning,
            "metadata": {}
        }
    ]

    if final_answer:
        parts.append({
            "type": "verifiable-responses",
            "answers": [final_answer],
        })

    converted["conversation_branches"] = [
        {
            "messages": [
                {
                    "role": "assistant",
                    "parts": parts,
                },
            ],
        },
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
        "operation": f"convert_{SRC}",
        "script": f"convert_{SRC}.py",
        "timestamp": datetime.now(UTC).isoformat(),
        "input_path": args.input,
        "output_path": str(output_path),
        "num_processes": args.num_proc,
        "limit": args.limit,
        "description": f"Converted {SRC} dataset from HuggingFace to unified chat format"
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
            "conversation_type": "gsm8k_math",
            "added_fields": ["system_prompt", "conversation_branches"],
            "edited_elements": ["Extracted final answer for verification"],
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
    p.add_argument("--num-proc", type=int, default=8, help="Number of processes for dataset operations")
    p.add_argument("--limit", type=int, default=None, help="Limit number of samples to process")
    return p.parse_args()


def main():
    args = cli()
    output_path = Path(args.output)

    if output_path.exists():
        response = input(f"{output_path} exists. Overwrite? [y/N]: ")
        if response.lower() != "y":
            sys.exit(0)

    if args.input is None:
        print("Loading GSM8K from HuggingFace Hub...")
        dataset = load_dataset("openai/gsm8k", "main", split="train")
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

    if args.limit and args.limit > 0:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
        print(f"Limited to {len(dataset)} samples")

    print("Converting samples to new format...")
    converted_samples = []
    for i, sample in enumerate(dataset):
        if i % 1000 == 0:
            print(f"Processing sample {i}/{len(dataset)}")
        converted_samples.append(convert_sample(sample))

    print("Creating DatasetDict...")
    converted_dataset = Dataset.from_list(converted_samples)
    dataset_dict = DatasetDict({"train": converted_dataset})

    print(f"Converted {len(converted_samples)} samples")

    save_dataset_and_metadata(dataset_dict, output_path, args)

    print("Conversion complete!")


if __name__ == "__main__":
    main()
