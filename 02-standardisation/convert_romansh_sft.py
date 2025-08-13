#!/usr/bin/env python3
"""
convert_romansh_sft.py
──────────────────────
Convert the Romansh SFT dataset into the unified chat-tool schema.

The Romansh dataset contains German-to-Romansh translation instruction pairs:
• `input` - Translation instructions in German (typically asking to translate German to Sursilvan or vice versa)
• `output` - The translation in Romansh (Sursilvan dialect) or German

Turn mapping
------------
input  → role="user"      (initial_prompt)
output → role="assistant"  part.type="response"
"""

import json
import sys
import argparse
import hashlib
import re
from pathlib import Path
from datetime import datetime, UTC
from typing import Dict, Any, Optional
from datasets import Dataset, DatasetDict, load_from_disk

# Hardcoded input path as specified
INPUT_PATH = "/capstor/store/cscs/swissai/infra01/posttrain_data/01_raw_hf_data/SFT_Romansh"
SRC = "romansh-sft"

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
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    return text

def detect_translation_direction(input_text: str) -> Dict[str, str]:
    """Detect the translation direction from the input text."""
    input_lower = input_text.lower()
    
    # Common patterns for detecting translation direction
    if "ins sursilvan" in input_lower or "ins romansch" in input_lower or "ins rumantsch" in input_lower:
        return {"source_lang": "German", "target_lang": "Romansh (Sursilvan)"}
    elif "ins deutsche" in input_lower or "auf deutsch" in input_lower:
        return {"source_lang": "Romansh (Sursilvan)", "target_lang": "German"}
    elif "sursilvan-begriffe" in input_lower:
        return {"source_lang": "Romansh (Sursilvan)", "target_lang": "German"}
    elif "deutscher begriffe" in input_lower:
        return {"source_lang": "German", "target_lang": "Romansh (Sursilvan)"}
    else:
        # Default assumption based on common pattern
        return {"source_lang": "German", "target_lang": "Romansh (Sursilvan)"}

def convert_row(row: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """Convert a single row to the new chat format."""
    
    # Extract and clean fields
    input_text = clean_text(row.get("input", ""))
    output_text = clean_text(row.get("output", ""))
    
    # Detect translation direction
    translation_info = detect_translation_direction(input_text)
    
    # Create conversation ID
    conversation_id = conv_id(input_text, idx)
    
    # Create the converted sample
    converted = {
        "conversation_id": conversation_id,
        "dataset_source": SRC,
        "original_metadata": {
            "row_index": idx,
            "translation_direction": translation_info,
            "task_type": "translation"
        },
        "system_prompt": {
            "content": "",
            "metadata": {}
        },
        "initial_prompt": {
            "role": "user",
            "content": input_text,
            "metadata": {
                "source_lang": translation_info["source_lang"],
                "target_lang": translation_info["target_lang"],
                "task_type": "translation"
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
                                output_text,
                                metadata={
                                    "language": translation_info["target_lang"]
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
        desc="Converting Romansh SFT dataset"
    )
    return converted

def analyze_dataset_patterns(dataset: Dataset, sample_size: int = 100) -> Dict[str, Any]:
    """Analyze patterns in the dataset to provide statistics."""
    sample_size = min(sample_size, len(dataset))
    sample = dataset.select(range(sample_size))
    
    patterns = {
        "german_to_romansh": 0,
        "romansh_to_german": 0,
        "list_translations": 0,
        "single_translations": 0
    }
    
    for item in sample:
        input_text = item.get("input", "").lower()
        
        # Direction detection
        if "ins sursilvan" in input_text or "ins romansch" in input_text:
            patterns["german_to_romansh"] += 1
        elif "ins deutsche" in input_text:
            patterns["romansh_to_german"] += 1
        
        # Format detection
        if "*" in input_text or "-" in input_text or re.search(r'\d+\)', input_text) or re.search(r'[a-z]\)', input_text):
            patterns["list_translations"] += 1
        else:
            patterns["single_translations"] += 1
    
    # Extrapolate to full dataset
    scale_factor = len(dataset) / sample_size
    for key in patterns:
        patterns[key] = int(patterns[key] * scale_factor)
    
    return patterns

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
        # Analyze patterns in the original dataset
        patterns = analyze_dataset_patterns(original_dataset[split_name])
        
        stats[split_name] = {
            "num_samples": len(split_data),
            "estimated_patterns": patterns
        }
    
    # Create processing entry
    processing_entry = {
        "operation": "convert_romansh_sft",
        "script": "convert_romansh_sft.py",
        "timestamp": datetime.now(UTC).isoformat(),
        "input_path": INPUT_PATH,
        "output_path": str(output_path),
        "num_processes": args.num_proc,
        "limit": args.limit,
        "statistics": stats,
        "description": "Converted Romansh SFT translation dataset to unified chat format"
    }
    
    # Add to processing log
    if "processing_log" not in metadata:
        metadata["processing_log"] = []
    metadata["processing_log"].append(processing_entry)
    
    # Add format metadata if not already present
    if "format" not in metadata:
        metadata["format"] = "chat_format_v1"
    if "source_dataset" not in metadata:
        metadata["source_dataset"] = "SFT_Romansh"
    if "conversion_details" not in metadata:
        metadata["conversion_details"] = {
            "conversation_type": "translation_task",
            "languages": ["German", "Romansh (Sursilvan)"],
            "primary_task": "German-Romansh bidirectional translation",
            "field_mapping": {
                "input": "initial_prompt.content",
                "output": "conversation_branches[0].messages[0].parts[0].content"
            },
            "metadata_preservation": "translation_direction_detection",
            "notes": "Dataset focuses on Sursilvan dialect of Romansh, one of Switzerland's official languages"
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
        if "estimated_patterns" in split_stats:
            patterns = split_stats["estimated_patterns"]
            print(f"  Estimated translation patterns (based on sample):")
            print(f"    German → Romansh: ~{patterns['german_to_romansh']:,}")
            print(f"    Romansh → German: ~{patterns['romansh_to_german']:,}")
            print(f"    List translations: ~{patterns['list_translations']:,}")
            print(f"    Single translations: ~{patterns['single_translations']:,}")

# ───────────— CLI / main ───────────── #
def cli():
    p = argparse.ArgumentParser(description="Convert Romansh SFT dataset to unified chat format")
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