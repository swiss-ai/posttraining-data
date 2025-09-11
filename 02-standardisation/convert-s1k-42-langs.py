#!/usr/bin/env python3
"""
convert_s1k_42_langs.py
───────────────────────
Convert the s1K_42_langs dataset from JSON format into the unified chat-tool schema.

The s1K dataset contains mathematical problems with thinking traces, structured as:
• conversation_id - Unique identifier
• dataset_source - Source identifier 
• original_metadata - Original metadata
• initial_prompt - User's mathematical problem
• conversation_branches - Assistant's thinking and response with verifiable answers
• created_timestamp - Creation timestamp

This converter:
1. Adds missing system_prompt and available_functions fields
2. Removes <think> and </think> tags from thought content
3. Ensures all parts have required fields (type, content, metadata, name, args)
4. Saves as DatasetDict for pipeline compatibility
"""

import re
import json
import sys
import argparse
import hashlib
from pathlib import Path
from datetime import datetime, UTC
from typing import List, Dict, Any, Optional
from datasets import Dataset, DatasetDict

# Hardcoded input path as specified
INPUT_PATH = "/iopsstor/scratch/cscs/smoalla/projects/swiss-alignment/artifacts/shared/datasets/s1K_42_langs/s1k_en_thk_42_langs_1k.json"
SRC = "s1k_42_langs"

# ───────────── helpers ────────────── #
def clean_think_tags(text: str) -> str:
    """Remove <think> and </think> tags from text."""
    if not text:
        return ""
    # Remove opening and closing think tags
    text = re.sub(r'^<think>\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*</think>$', '', text, flags=re.IGNORECASE)
    return text.strip()

def ensure_part_fields(part: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure part has all required fields with proper defaults."""
    # Required fields for all part types
    part_clean = {
        "type": part.get("type", "response"),
        "content": part.get("content", ""),
        "metadata": part.get("metadata", {}),
        "name": part.get("name", ""),
        "args": part.get("args", "")
    }
    
    # Clean think tags from thought content
    if part_clean["type"] == "thought":
        part_clean["content"] = clean_think_tags(part_clean["content"])
    
    # Special handling for verifiable-responses
    if part_clean["type"] == "verifiable-responses":
        if "answers" in part:
            part_clean["answers"] = part["answers"]
    
    return part_clean

def process_message(message: Dict[str, Any]) -> Dict[str, Any]:
    """Process a message to ensure proper format."""
    processed = {
        "role": message.get("role", "assistant")
    }
    
    # Handle parts if present
    if "parts" in message:
        processed["parts"] = [ensure_part_fields(part) for part in message["parts"]]
    elif "content" in message:
        # Convert old format to parts format
        processed["parts"] = [{
            "type": "response",
            "content": message["content"],
            "metadata": message.get("metadata", {}),
            "name": "",
            "args": ""
        }]
    else:
        processed["parts"] = []
    
    return processed

def convert_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a single sample to the new format."""
    
    # Start with existing fields
    converted = {
        "conversation_id": sample.get("conversation_id", ""),
        "dataset_source": sample.get("dataset_source", SRC),
        "original_metadata": sample.get("original_metadata", {}),
        "created_timestamp": sample.get("created_timestamp", datetime.now(UTC).isoformat())
    }
    
    # Add missing system_prompt field
    converted["system_prompt"] = sample.get("system_prompt", {
        "content": "",
        "metadata": {}
    })
    
    # Process initial_prompt
    initial = sample.get("initial_prompt", {})
    converted["initial_prompt"] = {
        "role": initial.get("role", "user"),
        "content": initial.get("content", ""),
        "metadata": initial.get("metadata", {})
    }
    
    # Add missing available_functions field
    converted["available_functions"] = sample.get("available_functions", [])
    
    # Process conversation branches
    branches = sample.get("conversation_branches", [])
    processed_branches = []
    
    for branch in branches:
        processed_messages = []
        for message in branch.get("messages", []):
            processed_messages.append(process_message(message))
        processed_branches.append({"messages": processed_messages})
    
    converted["conversation_branches"] = processed_branches
    
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
        "operation": "convert_s1k_42_langs",
        "script": "convert_s1k_42_langs.py",
        "timestamp": datetime.now(UTC).isoformat(),
        "input_path": INPUT_PATH,
        "output_path": str(output_path),
        "num_processes": args.num_proc,
        "limit": args.limit,
        "description": "Converted s1K_42_langs dataset from JSON to unified chat format with think tag removal"
    }
    
    # Add to processing log
    if "processing_log" not in metadata:
        metadata["processing_log"] = []
    metadata["processing_log"].append(processing_entry)
    
    # Add format metadata if not already present
    if "format" not in metadata:
        metadata["format"] = "chat_format_v1"
    if "source_dataset" not in metadata:
        metadata["source_dataset"] = "s1K_42_langs"
    if "conversion_details" not in metadata:
        metadata["conversion_details"] = {
            "conversation_type": "mathematical_problem_solving_with_thinking",
            "added_fields": ["system_prompt", "available_functions"],
            "cleaned_elements": ["<think> tags from thought content"],
            "field_normalization": "ensured all parts have required fields",
            "format": "new_chat_format_with_parts"
        }
    
    # Save metadata
    metadata_file = output_path / "dataset_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Dataset saved to {output_path}")
    print(f"Metadata saved to {metadata_file}")

# ───────────— CLI / main ───────────── #
def cli():
    p = argparse.ArgumentParser(description="Convert s1K_42_langs dataset to unified chat format")
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
    
    # Load JSON data
    print(f"Loading data from {INPUT_PATH}")
    try:
        with open(INPUT_PATH, 'r') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading input file: {e}")
        sys.exit(1)
    
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