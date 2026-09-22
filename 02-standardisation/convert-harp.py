import os
import zipfile
import json
import urllib.request
import tempfile
from datasets import Dataset
import enum
import re
import sys
import json
import argparse
import random
from datetime import datetime, UTC
from typing import Dict, Any, Optional
from pathlib import Path
from llmlatex import Parser, Formatter
from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets

SRC = "HARP"

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
    
    # Process initial_prompt
    converted["initial_prompt"] = {
        "role": "user",
        "content": sample["problem"],
        "metadata": {},
    }
    
    converted["available_functions"] = []

    conversation_branches = []
    for solution in sample["solutions"]:
        parts: list[Dict] = [
            {
                "type": "response",
                "content": solution,
                "metadata": {}
            }
        ]

        if sample["answer"] is not None:
            parts.append({
                "type": "verifiable-responses",
                "answers": [sample["answer"]],
            })
    
        conversation_branches.append({"messages": [{"role": "assistant", "parts": parts}],})
    
    converted["conversation_branches"] = conversation_branches
    
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
        "input_path": None,
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
    
    # Download and extract HARP data if not already present
    if not os.path.exists("HARP_raw.jsonl"):
        print("Downloading HARP_raw.jsonl.zip...")
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp_zip:
            urllib.request.urlretrieve(
                "https://github.com/aadityasingh/HARP/raw/refs/heads/main/HARP_raw.jsonl.zip",
                tmp_zip.name,
            )
            tmp_zip_path = tmp_zip.name

        print("Extracting HARP_raw.jsonl...")
        with zipfile.ZipFile(tmp_zip_path, "r") as zip_ref:
            zip_ref.extractall(".")

        os.remove(tmp_zip_path)
        print("Download complete!")
    
    def harp_generator():
        with open("HARP_raw.jsonl", "r") as f:
            for line in f:
                data = json.loads(line)

                if "[asy]" in data["problem"] or all("[asy]" in data[f"solution_{i+1}"] for i in range(data["num_solutions"])):
                    continue

                yield {
                    "problem": data["problem"],
                    "solutions": [data[f"solution_{i+1}"] for i in range(data["num_solutions"])],
                    "answer": data["answer"],
                }


    data = Dataset.from_generator(harp_generator)
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
