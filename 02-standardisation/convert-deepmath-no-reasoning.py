import sys
import json
import argparse
from datetime import datetime, UTC
from typing import Dict, Any, Optional
from pathlib import Path
from datasets import Dataset, DatasetDict, load_dataset

SRC = "DeepMath-103K"

def convert_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a single sample to the new format, keeping only the answer (no thought)."""

    # Solutions have format: {thinking}</think>{answer} — no opening <think> tag.
    # Pick the solution whose thinking block is shortest (proxy for most concise reasoning),
    # then take only the answer part after </think>.
    thinkings_and_indices = [
        (0, sample["r1_solution_1"].split("</think>")[0]),
        (1, sample["r1_solution_2"].split("</think>")[0]),
        (2, sample["r1_solution_3"].split("</think>")[0]),
    ]
    index = min(thinkings_and_indices, key=lambda x: len(x[1]))[0]
    answer = sample[f"r1_solution_{index + 1}"].split("</think>")[1].strip()

    parts: list[Dict] = [
        {
            "type": "response",
            "content": answer,
            "metadata": {},
        },
        {
            "type": "verifiable-responses",
            "answers": [sample["final_answer"]],
        },
    ]

    return {
        "conversation_id": "",
        "dataset_source": SRC,
        "original_metadata": {},
        "created_timestamp": datetime.now(UTC).isoformat(),
        "system_prompt": {"content": "", "metadata": {}},
        "initial_prompt": {
            "role": "user",
            "content": sample["question"],
            "metadata": {},
        },
        "available_functions": [],
        "conversation_branches": [
            {
                "messages": [
                    {
                        "role": "assistant",
                        "parts": parts,
                    },
                ],
            },
        ],
    }


def load_existing_metadata(output_path: Path) -> Optional[Dict[str, Any]]:
    meta_file = output_path / "dataset_metadata.json"
    if meta_file.exists():
        try:
            with open(meta_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return None

def save_dataset_and_metadata(dataset_dict: DatasetDict, output_path: Path, args: argparse.Namespace):
    output_path.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(output_path))

    metadata = load_existing_metadata(output_path) or {}

    processing_entry = {
        "operation": f"convert_{SRC}_no_reasoning",
        "script": f"convert-{SRC}-no-reasoning.py",
        "timestamp": datetime.now(UTC).isoformat(),
        "input_path": args.input,
        "output_path": str(output_path),
        "num_processes": args.num_proc,
        "limit": args.limit,
        "description": f"Converted {SRC} dataset to unified chat format (no reasoning traces)",
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
            "conversation_type": "math_reasoning",
            "added_fields": ["system_prompt", "conversation_branches"],
            "edited_fields": ["r1_solution (thinking stripped, only answer after </think> kept)"],
            "format": "new_chat_format_with_parts",
        }

    metadata_file = output_path / "dataset_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Dataset saved to {output_path}")
    print(f"Metadata saved to {metadata_file}")

def cli():
    p = argparse.ArgumentParser(description=f"Convert {SRC} dataset to unified chat format (no reasoning)")
    p.add_argument("-i", "--input", default=None, help="Unused; dataset is always loaded from HuggingFace Hub.")
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

    data = load_dataset("zwhe99/DeepMath-103K", split="train")
    print(f"Loaded {len(data)} samples")

    if args.limit and args.limit > 0:
        data = data.select(range(min(args.limit, len(data))))
        print(f"Limited to {len(data)} samples")

    print("Converting samples to new format...")
    converted_samples = []
    for i, sample in enumerate(data):
        if i % 1000 == 0:
            print(f"Processing sample {i}/{len(data)}")
        converted_samples.append(convert_sample(sample))

    print("Creating DatasetDict...")
    dataset_dict = DatasetDict({"train": Dataset.from_list(converted_samples)})

    print(f"Converted {len(converted_samples)} samples")
    save_dataset_and_metadata(dataset_dict, output_path, args)
    print("Conversion complete!")

if __name__ == "__main__":
    main()
