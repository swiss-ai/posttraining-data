"""
convert_LEXam_open.py
───────────────────────
Convert the LEXam-Benchmark/LEXam dataset (open_question subset) from HuggingFace into the unified chat-tool schema.

The LEXam Open Questions dataset contains multiple-choice questions structured as:
    - question: The multiple-choice question
    - answer: Reference answer provided by legal domain experts
    - course: Title of the law course from which the question was derived
    - language: Language of the question (en or de)
    - area: Legal area covered by the question (criminal, public, private, or interdisciplinary)
    - jurisdiction: Legal jurisdiction of the question (Swiss, international, or generic)
    - year: Year when the exam was administered (2016 to 2022)
    - id: Unique identifier for the question


This converter:
1. Converts the question and answer into a clear prompt format
2. Includes metadata about the question
"""



import enum
import re
import sys
import json
import argparse
import random
from datetime import datetime, UTC
from typing import Dict, Any, Optional
from pathlib import Path
from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets

SRC = "LEXam_open_question"

def extract_meta(sample: Dict[str, Any]) -> Dict[str, Any]:
    '''Extract extra information (e.g. language, course etc.) to provide as metadata'''
    extra = {}
    for key in sample:
        if key not in ['question', 'answer']:
            extra[key] = sample[key]
    return extra

def convert_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a single sample to the new format."""
    
    # Start with existing fields
    converted: Dict[str, Any] = {
        "conversation_id": "",
        "dataset_source": SRC,
        "original_metadata": {},
        "created_timestamp": datetime.now(UTC).isoformat(),
    }
    
    # System prompt is always the same, we don't want it
    converted["system_prompt"] = {
        "content": "",
        "metadata": {},
    }
    
    # Extracts parts.
    question = sample['question']
    answer = sample['answer']
    extra_info = extract_meta(sample)
    
    # Process initial_prompt
    converted["initial_prompt"] = {
        "role": "user",
        "content": question,
        "metadata": extra_info
        }
    
    # No available functions in this dataset
    converted["available_functions"] = []

    parts: list[Dict] = [
        {
            "type": "response",
            "content": answer,
            "metadata": 
                {}
        }
    ]

    # if answer is not None:
    #     parts.append({
    #         "type": "verifiable-responses",
    #         "answers": [],
    #     })
    
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
            "conversation_type": "LEXam_open",
            "added_fields": ["system_prompt", "conversation_branches"],
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
    return p.parse_args()

def main():
    args = cli()
    output_path = Path(args.output)
    
    # Check if output exists
    if output_path.exists():
        response = input(f"{output_path} exists. Overwrite? [y/N]: ")
        if response.lower() != "y":
            sys.exit(0)
    
    if args.input is None:
        dataset_test = load_dataset("LEXam-Benchmark/LEXam", "open_question", split="test")
        dataset_dev = load_dataset("LEXam-Benchmark/LEXam", "open_question", split="dev")
        dataset = DatasetDict({"test": concatenate_datasets([dataset_test, dataset_dev])})
    else:
        # Load JSON data
        print(f"Loading data from {args.input}")
        try:
            with open(args.input, 'r') as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading input file: {e}")
            sys.exit(1)
        dataset = DatasetDict({'test': Dataset.from_list(data)})
    
    if not isinstance(dataset, DatasetDict):
        print(f"Unexpected dataset type.")
        sys.exit(1)
        
    data = dataset["test"]
    print(f"Loaded {len(data)} samples")
    
    # Apply limit if specified
    if args.limit and args.limit > 0:
        data = data[:args.limit]
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