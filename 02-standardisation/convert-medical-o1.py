"""
Convert FreedomIntelligence/medical-o1 data into the unified chat format.

The converter joins:
  - FreedomIntelligence/medical-o1-reasoning-SFT: Question, Complex_CoT, Response
  - FreedomIntelligence/medical-o1-verifiable-problem: Open-ended Verifiable Question,
    Ground-True Answer

Rows are kept only when a ground-truth answer is available.
"""

import sys
import json
import argparse
import hashlib
from datetime import datetime, UTC
from typing import Dict, Any, Optional, List
from pathlib import Path
from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets

SRC = "medical-o1-reasoning-SFT"


def stable_conversation_id(question: str) -> str:
    digest = hashlib.sha256(question.encode("utf-8")).hexdigest()[:16]
    return f"medical_o1_reasoning_sft_{digest}"

def extract_meta(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Extract source fields that are not the prompt/answer payload."""
    extra = {}
    for key in sample:
        if key not in [
            "Question",
            "Complex_CoT",
            "Response",
            "Open-ended Verifiable Question",
            "Ground-True Answer",
        ]:
            extra[key] = sample[key]
    return extra


def make_part(part_type: str, content: str = "", answers: Optional[List[str]] = None) -> Dict[str, Any]:
    part = {
        "type": part_type,
        "content": content,
        "metadata": {},
    }
    if answers is not None:
        part["answers"] = answers
    return part

def convert_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a single sample to the new format."""
    
    # Start with existing fields
    converted: Dict[str, Any] = {
        "conversation_id": stable_conversation_id(str(sample.get("Question") or "")),
        "dataset_source": SRC,
        "original_metadata": {},
        "created_timestamp": datetime.now(UTC).isoformat(),
    }
    
    # System prompt is always the same, we don't want it
    converted["system_prompt"] = {
        "content": "",
        "metadata": {},
    }
    
    question = str(sample.get("Question") or "").strip()
    reasoning = str(sample.get("Complex_CoT") or "").strip()
    answer = str(sample.get("Response") or "").strip()
    ground_truth = str(sample.get("Ground-True Answer") or "").strip()
    extra_info = extract_meta(sample)
    
    # Process initial_prompt
    converted["initial_prompt"] = {
        "role": "user",
        "content": question,
        "metadata": extra_info
        }
    
    # No available functions in this dataset
    converted["available_functions"] = []

    parts: list[Dict] = []
    if reasoning:
        parts.append(make_part("thought", reasoning))
    parts.append(make_part("response", answer))
    if ground_truth:
        parts.append(make_part("verifiable-responses", answers=[ground_truth]))
    
    # Process conversation branches    
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
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save dataset
    dataset_dict.save_to_disk(str(output_path))
    
    # Load existing metadata or create new
    metadata = load_existing_metadata(output_path) or {}
    
    # Create processing entry
    processing_entry = {
        "operation": f"convert_{SRC}",
        "script": f"convert_{SRC}.py",
        "timestamp": datetime.now(UTC).isoformat(),
        "input_path": args.input,
        "output_path": str(output_path),
        "num_processes": args.num_proc,
        "limit": args.limit,
        "description": f"Converted {SRC} dataset from JSON to unified chat format"
    }
    
    # Add to processing log
    if "processing_log" not in metadata:
        metadata["processing_log"] = []
    metadata["processing_log"].append(processing_entry)
    
    # Add format metadata if not already present
    if "format" not in metadata:
        metadata["format"] = "chat_format_v1"
    if "source_dataset" not in metadata:
        metadata["source_dataset"] = SRC
    if "conversion_details" not in metadata:
        metadata["conversion_details"] = {
            "conversation_type": "medical_reasoning_verifiable",
            "added_fields": ["system_prompt", "conversation_branches"],
            "edited_fields": ["Complex_CoT preserved as assistant thought part"],
            "format": "new_chat_format_with_parts"
        }
    
    # Save metadata
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
    p.add_argument("--overwrite", action="store_true", default=False, help="Overwrite output without prompting")
    return p.parse_args()

def main():
    args = cli()
    output_path = Path(args.output)
    
    # Check if output exists
    if output_path.exists() and not args.overwrite:
        response = input(f"{output_path} exists. Overwrite? [y/N]: ")
        if response.lower() != "y":
            sys.exit(0)
    
    if args.input is None:
        dataset_reasoning = concatenate_datasets([load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", "en", split="train"), load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", "zh", split="train")])
        dataset_direct = load_dataset("FreedomIntelligence/medical-o1-verifiable-problem", split="train")
    else:
        raise ValueError("Input file is not supported for this dataset.")

    dataset_reasoning_pandas = dataset_reasoning.to_pandas()
    dataset_direct_pandas = dataset_direct.to_pandas()

    print(dataset_reasoning, dataset_direct)

    # Merge datasets based on matching questions
    merged_data = dataset_reasoning_pandas.merge(
        dataset_direct_pandas[["Open-ended Verifiable Question", "Ground-True Answer"]], 
        left_on="Question", 
        right_on="Open-ended Verifiable Question", 
        how="left"
    )
    
    data = Dataset.from_pandas(merged_data)

    def filter_fn(sample: Dict[str, Any]) -> bool:
        return sample["Ground-True Answer"] is not None
    
    data = data.filter(filter_fn)

    print(f"Loaded {len(data)} samples")
    
    # Apply limit if specified
    if args.limit and args.limit > 0:
        data = data.select(range(min(args.limit, len(data))))
        print(f"Limited to {len(data)} samples")
    
    # Convert samples
    print("Converting samples to new format...")
    converted_samples = []
    for i, sample in enumerate(data):
        if i % 1000 == 0:
            print(f"Processing sample {i}/{len(data)}")
        converted_samples.append(convert_sample(sample))
    
    # Create Dataset and DatasetDict
    print("Creating DatasetDict...")    
    dataset = Dataset.from_list(converted_samples)
    dataset_dict = DatasetDict({"train": dataset})
    
    print(f"Converted {len(converted_samples)} samples")
    
    # Save dataset and metadata
    save_dataset_and_metadata(dataset_dict, output_path, args)
    
    print("Conversion complete!")

if __name__ == "__main__":
    main()
