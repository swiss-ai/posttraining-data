#!/usr/bin/env python3
"""
convert_charter_qa.py
─────────────────────
Convert the Charter QA dataset into the unified chat format with parts structure.

The Charter QA dataset contains questions and responses about the Swiss AI Charter:
• `question` - The user's question about the Swiss AI Charter
• `response` - The assistant's response explaining charter principles

Turn mapping
────────────
question → role="user"      (initial_prompt.content)
response → role="assistant"  part.type="response"
"""

import json
import sys
import argparse
import hashlib
from pathlib import Path
from datetime import datetime, UTC
from typing import Dict, Any, Optional
from datasets import Dataset, DatasetDict, load_from_disk

# Dataset source identifier
SRC = "charter-qa"

# No system prompt - empty content
SYSTEM_PROMPT = ""

# ───────────── helpers ────────────── #
def conv_id(seed: str, idx: int) -> str:
    """Generate a unique conversation ID."""
    combined = f"{seed}_{idx}"
    return f"{SRC}_{hashlib.sha256(combined.encode()).hexdigest()[:12]}"

def make_part(ptype: str,
              content: str = "",
              name: str = "",
              args: str = "",
              metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create a message part with all required fields for Arrow compatibility."""
    return {
        "type": ptype,
        "content": content,
        "metadata": metadata or {},
        "name": name,
        "args": args
    }

def clean_text(text: str) -> str:
    """Clean and normalize text content."""
    if not text:
        return ""
    # Remove excessive whitespace while preserving paragraph structure
    text = text.strip()
    # Normalize line breaks
    import re
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    return text

def convert_row(row: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """Convert a single row to the new chat format."""
    
    # Extract and clean fields
    question = clean_text(row.get("question", ""))
    response = clean_text(row.get("response", ""))
    
    # Create conversation ID
    conversation_id = conv_id(question, idx)
    
    # Create the converted sample
    converted = {
        "conversation_id": conversation_id,
        "dataset_source": SRC,
        "original_metadata": {
            "row_index": idx
        },
        "system_prompt": {
            "content": SYSTEM_PROMPT,
            "metadata": {}
        },
        "initial_prompt": {
            "role": "user",
            "content": question,
            "metadata": {}
        },
        "available_functions": [],
        "conversation_branches": [
            {
                "messages": [
                    {
                        "role": "assistant",
                        "parts": [
                            make_part(
                                "response",
                                response,
                                metadata={}
                            )
                        ]
                    }
                ]
            }
        ],
        "created_timestamp": datetime.now(UTC).isoformat()
    }
    
    return converted

def process_dataset(dataset: Dataset, num_proc: int) -> Dataset:
    """Process the dataset with the converter."""
    # Convert with progress tracking
    converted = dataset.map(
        convert_row,
        with_indices=True,
        num_proc=num_proc,
        desc="Converting Charter QA"
    )
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

def save_dataset_and_metadata(dataset_dict: DatasetDict, output_path: Path, 
                             args: argparse.Namespace, input_path: str):
    """Save converted dataset with processing metadata."""
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save dataset
    dataset_dict.save_to_disk(str(output_path))
    
    # Load existing metadata or create new
    metadata = load_existing_metadata(output_path) or {}
    
    # Gather statistics
    stats = {}
    for split_name, split_data in dataset_dict.items():
        stats[split_name] = {
            "num_samples": len(split_data)
        }
    
    # Create processing entry
    processing_entry = {
        "operation": "convert_charter_qa",
        "script": "convert_charter_qa.py",
        "timestamp": datetime.now(UTC).isoformat(),
        "input_path": input_path,
        "output_path": str(output_path),
        "num_processes": args.num_proc,
        "limit": args.limit,
        "statistics": stats,
        "description": "Converted Charter QA dataset to unified chat format with parts structure"
    }
    
    # Add to processing log
    if "processing_log" not in metadata:
        metadata["processing_log"] = []
    metadata["processing_log"].append(processing_entry)
    
    # Add format metadata if not already present
    if "format" not in metadata:
        metadata["format"] = "chat_format_v2"
    if "source_dataset" not in metadata:
        metadata["source_dataset"] = "charter-qa"
    if "conversion_details" not in metadata:
        metadata["conversion_details"] = {
            "conversation_type": "question_answering",
            "language": "English",
            "domain": "Swiss AI Charter",
            "field_mapping": {
                "question": "initial_prompt.content",
                "response": "conversation_branches[0].messages[0].parts[0].content"
            },
            "system_prompt": "Charter-specific expert assistant",
            "metadata_preservation": "minimal"
        }
    
    # Save metadata
    metadata_file = output_path / "dataset_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"Dataset saved to {output_path}")
    print(f"Metadata saved to {metadata_file}")
    
    # Print statistics
    print("\nDataset Statistics:")
    for split_name, split_stats in stats.items():
        print(f"  {split_name} split: {split_stats['num_samples']} samples")

# ───────────— CLI / main ───────────── #
def cli():
    p = argparse.ArgumentParser(description="Convert Charter QA dataset to unified chat format")
    p.add_argument("input_path", 
                   nargs='?',
                   default="/capstor/store/cscs/swissai/infra01/posttrain_data/01_raw_hf_data/charter_qa",
                   help="Path to the input Charter QA dataset (default: /capstor/store/cscs/swissai/infra01/posttrain_data/01_raw_hf_data/charter_qa)")
    p.add_argument("-o", "--output", required=True, help="Output directory path")
    p.add_argument("--num-proc", type=int, default=8, help="Number of processes for dataset operations")
    p.add_argument("--limit", type=int, default=None, help="Limit number of samples to process")
    return p.parse_args()

def main():
    args = cli()
    input_path = args.input_path
    output_path = Path(args.output)
    
    # Handle output path - if directory provided, append dataset name
    if args.output.endswith("/"):
        output_path = Path(args.output) / "charter-qa"
    
    # Check if output exists
    if output_path.exists():
        response = input(f"{output_path} exists. Overwrite? [y/N]: ")
        if response.lower() != "y":
            sys.exit(0)
    
    # Load dataset
    print(f"Loading dataset from {input_path}")
    try:
        dataset = load_from_disk(input_path)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)
    
    # Ensure it's a DatasetDict
    if not isinstance(dataset, DatasetDict):
        dataset = DatasetDict({"train": dataset})
    
    print(f"Loaded dataset with splits: {list(dataset.keys())}")
    for split_name, split_data in dataset.items():
        print(f"  {split_name}: {len(split_data)} samples")
    
    # Process each split
    output_dict = DatasetDict()
    for split_name, split_data in dataset.items():
        print(f"\nProcessing {split_name} split...")
        
        # Apply limit if specified
        if args.limit and args.limit > 0:
            split_data = split_data.select(range(min(args.limit, len(split_data))))
            print(f"  Limited to {len(split_data)} samples")
        
        # Convert the split
        converted_split = process_dataset(split_data, args.num_proc)
        output_dict[split_name] = converted_split
        print(f"  Converted {len(converted_split)} samples")
    
    # Save dataset and metadata
    save_dataset_and_metadata(output_dict, output_path, args, input_path)
    
    print("\nConversion complete!")

if __name__ == "__main__":
    main()