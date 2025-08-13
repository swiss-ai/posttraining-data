#!/usr/bin/env python3
"""
convert_swiss_german_dialects.py
────────────────────────────────
Convert the Swiss German dialect instruction dataset into the unified chat-tool schema.

The Swiss German dataset contains instruction-response pairs in various Swiss German dialects:
• `instruction` - The user's request/question in Swiss German
• `output` - The assistant's response in Swiss German
• `source` - Dataset origin (akoksal/muri-it, lightblue/tagengo-gpt4, CohereForAI/aya_dataset)
• `dialect` - Swiss German dialect code (ch_sg, ch_be, ch_gr, ch_zh, ch_vs, ch_bs, ch_ag, ch_lu)

Turn mapping
------------
instruction → role="user"      (initial_prompt)
output     → role="assistant"  part.type="response"
"""

import json
import sys
import argparse
import hashlib
from pathlib import Path
from datetime import datetime, UTC
from typing import Dict, Any, Optional
from datasets import Dataset, DatasetDict, load_from_disk

# Hardcoded input path as specified
INPUT_PATH = "/capstor/store/cscs/swissai/infra01/posttrain_data/01_raw_hf_data/swiss-german-instruct-data-dialects-with-translation-instructions-v2"
SRC = "swiss-german-dialects"

# Dialect mapping for better readability in metadata
DIALECT_NAMES = {
    "ch_sg": "St. Gallen",
    "ch_be": "Bern",
    "ch_gr": "Graubünden",
    "ch_zh": "Zürich",
    "ch_vs": "Valais",
    "ch_bs": "Basel-Stadt",
    "ch_ag": "Aargau",
    "ch_lu": "Luzern"
}

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
    """Create a message part with all required fields."""
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
    # Remove excessive whitespace
    text = text.strip()
    # Normalize line breaks
    import re
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    return text

def convert_row(row: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """Convert a single row to the new chat format."""
    
    # Extract and clean fields
    instruction = clean_text(row.get("instruction", ""))
    output = clean_text(row.get("output", ""))
    source = row.get("source", "")
    dialect = row.get("dialect", "")
    dialect_name = DIALECT_NAMES.get(dialect, dialect)
    
    # Create conversation ID
    conversation_id = conv_id(instruction, idx)
    
    # Create the converted sample
    converted = {
        "conversation_id": conversation_id,
        "dataset_source": SRC,
        "original_metadata": {
            "row_index": idx,
            "source": source,
            "dialect": dialect,
            "dialect_name": dialect_name
        },
        "system_prompt": {
            "content": "",
            "metadata": {}
        },
        "initial_prompt": {
            "role": "user",
            "content": instruction,
            "metadata": {
                "dialect": dialect,
                "dialect_name": dialect_name,
                "source": source
            }
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
                                output,
                                metadata={
                                    "dialect": dialect,
                                    "dialect_name": dialect_name
                                }
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
        desc="Converting Swiss German dialects"
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
                             args: argparse.Namespace, original_dataset: DatasetDict):
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
        # Get dialect distribution
        dialect_counts = {}
        source_counts = {}
        
        for sample in split_data:
            dialect = sample["original_metadata"]["dialect"]
            source = sample["original_metadata"]["source"]
            
            dialect_counts[dialect] = dialect_counts.get(dialect, 0) + 1
            source_counts[source] = source_counts.get(source, 0) + 1
        
        stats[split_name] = {
            "num_samples": len(split_data),
            "dialect_distribution": dialect_counts,
            "source_distribution": source_counts
        }
    
    # Create processing entry
    processing_entry = {
        "operation": "convert_swiss_german_dialects",
        "script": "convert_swiss_german_dialects.py",
        "timestamp": datetime.now(UTC).isoformat(),
        "input_path": INPUT_PATH,
        "output_path": str(output_path),
        "num_processes": args.num_proc,
        "limit": args.limit,
        "statistics": stats,
        "description": "Converted Swiss German dialect instruction dataset to unified chat format"
    }
    
    # Add to processing log
    if "processing_log" not in metadata:
        metadata["processing_log"] = []
    metadata["processing_log"].append(processing_entry)
    
    # Add format metadata if not already present
    if "format" not in metadata:
        metadata["format"] = "chat_format_v1"
    if "source_dataset" not in metadata:
        metadata["source_dataset"] = "swiss-german-instruct-data-dialects-with-translation-instructions-v2"
    if "conversion_details" not in metadata:
        metadata["conversion_details"] = {
            "conversation_type": "instruction_following",
            "language": "Swiss German dialects",
            "dialects": list(DIALECT_NAMES.keys()),
            "dialect_names": DIALECT_NAMES,
            "sources": ["akoksal/muri-it", "lightblue/tagengo-gpt4", "CohereForAI/aya_dataset"],
            "field_mapping": {
                "instruction": "initial_prompt.content",
                "output": "conversation_branches[0].messages[0].parts[0].content",
                "source": "original_metadata.source",
                "dialect": "original_metadata.dialect"
            },
            "metadata_preservation": "full_original_metadata"
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
        print(f"\n{split_name} split:")
        print(f"  Total samples: {split_stats['num_samples']}")
        print(f"  Dialect distribution:")
        for dialect, count in sorted(split_stats['dialect_distribution'].items()):
            dialect_name = DIALECT_NAMES.get(dialect, dialect)
            print(f"    {dialect} ({dialect_name}): {count}")
        print(f"  Source distribution:")
        for source, count in sorted(split_stats['source_distribution'].items()):
            print(f"    {source}: {count}")

# ───────────— CLI / main ───────────── #
def cli():
    p = argparse.ArgumentParser(description="Convert Swiss German dialect dataset to unified chat format")
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
    
    # Load dataset
    print(f"Loading dataset from {INPUT_PATH}")
    try:
        dataset = load_from_disk(INPUT_PATH)
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
    save_dataset_and_metadata(output_dict, output_path, args, dataset)
    
    print("\nConversion complete!")

if __name__ == "__main__":
    main()