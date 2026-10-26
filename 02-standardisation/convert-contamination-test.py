"""
convert_contamination_test.py
───────────────────────
Convert the synthetic decontamination test set (04-decontamination/contamination_dataset.jsonl)
into the unified chat-tool schema, so it can be run through 04-decontamination like any other source.

Each line of the JSONL file contains:
    - id: Sample identifier (e.g. sample_0000)
    - question: The question, problem or text to complete
    - choices: List of answer choices (multiple-choice items) or null
    - answer: Solution text, or the correct choice as a 0-based index ("0") or letter ("D")
    - is_contaminated: Whether the sample is taken from a benchmark
    - benchmark_hf_dir, benchmark_config, benchmark_split, benchmark_row: Source benchmark item (or null)
    - contamination_type: exact_copy, surface_change, light_rewording, option_shuffle (or null)
    - item_type: Kind of item (math, mcq, qa, completion, or the benchmark it was taken from)

This converter:
1. Uses the question as the user prompt, with the choices appended as "A. ..." lines
   (only the question with --stem-only)
2. Uses the solution text, or the correct choice, as the assistant response
3. Keeps all labels in original_metadata, so the decontamination result can be scored
   (a removed clean sample is a false positive, a kept contaminated sample a false negative)

The input file is only read, never changed.
"""

import sys
import json
import argparse
from datetime import datetime, UTC
from typing import Dict, Any, Optional
from pathlib import Path
from datasets import Dataset, DatasetDict
from conversation_ids import add_conversation_ids_for_output

SRC = "contamination_test"
DEFAULT_INPUT = Path(__file__).resolve().parent.parent / "04-decontamination" / "contamination_dataset.jsonl"
LABELS = "ABCDEFGHIJ"


def answer_label(answer: str, num_choices: int) -> str:
    """Return the letter of the correct choice from a 0-based index or a letter."""
    answer = answer.strip()
    if answer.isdigit() and int(answer) < num_choices:
        return LABELS[int(answer)]
    if len(answer) == 1 and answer.upper() in LABELS[:num_choices]:
        return answer.upper()
    raise ValueError(f"Cannot map answer {answer!r} to one of {num_choices} choices")


def convert_sample(sample: Dict[str, Any], stem_only: bool = False) -> Dict[str, Any]:
    """Convert a single sample to the new format."""
    converted: Dict[str, Any] = {
        "conversation_id": "",
        "dataset_source": SRC,
        "original_metadata": {key: sample.get(key) for key in sample if key != "question"},
        "created_timestamp": datetime.now(UTC).isoformat(),
    }

    converted["system_prompt"] = {
        "content": "",
        "metadata": {},
    }

    question = sample["question"]
    choices = sample.get("choices")
    if choices:
        label = answer_label(sample["answer"], len(choices))
        options = "\n".join(f"{LABELS[i]}. {choice}" for i, choice in enumerate(choices))
        if not stem_only:
            question = f"{question.rstrip()}\n{options}"
        answer = f"{label}. {choices[LABELS.index(label)]}"
    else:
        answer = sample["answer"]

    converted["initial_prompt"] = {
        "role": "user",
        "content": question,
        "metadata": {},
    }

    converted["available_functions"] = []

    converted["conversation_branches"] = [
        {
            "messages": [
                {
                    "role": "assistant",
                    "parts": [
                        {
                            "type": "response",
                            "content": answer,
                            "metadata": {},
                        }
                    ],
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

    dataset_dict = add_conversation_ids_for_output(dataset_dict, output_path)
    dataset_dict.save_to_disk(str(output_path))

    metadata = load_existing_metadata(output_path) or {}

    processing_entry = {
        "operation": f"convert_{SRC}",
        "script": f"convert_{SRC}.py",
        "timestamp": datetime.now(UTC).isoformat(),
        "input_path": str(args.input),
        "output_path": str(output_path),
        "limit": args.limit,
        "stem_only": args.stem_only,
        "description": f"Converted {SRC} dataset from JSONL to unified chat format"
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
            "conversation_type": "decontamination_test_set",
            "added_fields": ["system_prompt", "conversation_branches"],
            "edited elements": ["Choices appended to the user prompt"],
            "format": "new_chat_format_with_parts"
        }

    metadata_file = output_path / "dataset_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Dataset saved to {output_path}")
    print(f"Metadata saved to {metadata_file}")


def cli():
    p = argparse.ArgumentParser(description=f"Convert {SRC} dataset to unified chat format")
    p.add_argument("-i", "--input", default=str(DEFAULT_INPUT), help="Input JSONL file path")
    p.add_argument("-o", "--output", required=True, help="Output directory path")
    p.add_argument("--limit", type=int, default=None, help="Limit number of samples to process")
    p.add_argument("--stem-only", action="store_true", help="Do not append the choices to the user prompt")
    return p.parse_args()


def main():
    args = cli()
    output_path = Path(args.output)

    if output_path.exists():
        response = input(f"{output_path} exists. Overwrite? [y/N]: ")
        if response.lower() != "y":
            sys.exit(0)

    print(f"Loading data from {args.input}")
    with open(args.input, 'r') as f:
        data = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(data)} samples")

    if args.limit and args.limit > 0:
        data = data[:args.limit]
        print(f"Limited to {len(data)} samples")

    print("Converting samples to new format...")
    converted_samples = [convert_sample(sample, args.stem_only) for sample in data]

    dataset_dict = DatasetDict({"train": Dataset.from_list(converted_samples)})
    print(f"Converted {len(converted_samples)} samples")

    save_dataset_and_metadata(dataset_dict, output_path, args)

    print("Conversion complete!")


if __name__ == "__main__":
    main()
