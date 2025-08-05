#!/usr/bin/env python3
"""
Convert datasets from old chat format to new chat format.

Old format: Assistant content is a string
New format: Assistant/user content becomes a list of parts with types
"""

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from datasets import Dataset, DatasetDict, load_from_disk
from tqdm import tqdm
from datetime import datetime


def load_existing_metadata(dataset_path: Path) -> Dict[str, Any]:
    """Load existing metadata from dataset directory."""
    metadata_file = dataset_path / "dataset_metadata.json"
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            return json.load(f)
    return {}


def save_dataset_with_processing_log(dataset_dict: DatasetDict, output_path: Path, 
                                   input_path: str, total_samples: int):
    """Save dataset and update processing log."""
    # Load existing metadata
    original_metadata = load_existing_metadata(Path(input_path))
    
    # Create processing log entry
    processing_entry = {
        "operation": "format_conversion",
        "script": "convert_old_to_new_format.py",
        "timestamp": datetime.now().isoformat(),
        "input_path": input_path,
        "output_path": str(output_path),
        "samples_processed": total_samples,
        "conversion_details": {
            "from_format": "string_content", 
            "to_format": "parts_structure",
            "description": "Convert assistant and user messages from string content to parts list structure"
        }
    }
    
    # Update metadata
    metadata = {
        **original_metadata,
        "processing_log": original_metadata.get("processing_log", []) + [processing_entry]
    }
    
    # Save dataset and metadata
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(output_path))
    
    with open(output_path / "dataset_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)


def convert_message_to_parts(message: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a message from old format to new format with parts.
    
    Old format: {"role": "assistant", "content": "text", "metadata": {...}}
    New format: {"role": "assistant", "parts": [{"type": "response", "content": "text", "metadata": {...}}]}
    """
    new_message = copy.deepcopy(message)
    
    # If content exists and is a string, convert to parts
    if "content" in new_message and isinstance(new_message["content"], str):
        content = new_message.pop("content")
        metadata = new_message.get("metadata", {})
        
        # Create parts list with single response part
        new_message["parts"] = [{
            "type": "response",
            "content": content,
            "metadata": metadata
        }]
    
    # If content is already a list (already new format), rename to parts
    elif "content" in new_message and isinstance(new_message["content"], list):
        new_message["parts"] = new_message.pop("content")
    
    return new_message


def convert_sample_to_new_format(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a single sample from old format to new format.
    
    Main changes:
    1. Assistant messages: content (string) -> parts (list)
    2. User messages in conversation: also get parts structure
    3. Preserve all other fields unchanged
    """
    # Deep copy to avoid modifying original
    new_sample = copy.deepcopy(sample)
    
    # Process conversation_branches
    if "conversation_branches" in new_sample:
        for branch in new_sample["conversation_branches"]:
            if "messages" in branch:
                converted_messages = []
                for message in branch["messages"]:
                    # Convert both assistant and user messages
                    if message.get("role") in ["assistant", "user"]:
                        converted_message = convert_message_to_parts(message)
                        converted_messages.append(converted_message)
                    else:
                        # Keep other message types unchanged
                        converted_messages.append(message)
                branch["messages"] = converted_messages
    
    # Update timestamp to reflect conversion
    if "created_timestamp" not in new_sample:
        new_sample["created_timestamp"] = datetime.now().isoformat()
    
    return new_sample


def validate_new_format(sample: Dict[str, Any]) -> bool:
    """
    Validate that a sample is in the correct new format.
    
    Returns True if valid, False otherwise.
    """
    # Check conversation_branches structure
    if "conversation_branches" not in sample:
        return False
    
    for branch in sample["conversation_branches"]:
        if "messages" not in branch:
            return False
        
        for message in branch["messages"]:
            # Check that assistant/user messages have parts, not content
            if message.get("role") in ["assistant", "user"]:
                if "content" in message and isinstance(message["content"], str):
                    # Still has old format
                    return False
                if "parts" not in message:
                    # Missing parts field
                    return False
                if not isinstance(message["parts"], list):
                    # Parts should be a list
                    return False
    
    return True


def process_dataset(dataset: Dataset, validate: bool = True) -> Dataset:
    """
    Process a single dataset split, converting all samples to new format.
    """
    def convert_batch(examples: Dict[str, List]) -> Dict[str, List]:
        """
        Batch processing function for dataset.map()
        Following critical guideline: never modify input in-place
        """
        # Deep copy to avoid corruption
        processed_examples = copy.deepcopy(examples)
        
        # Get the number of examples
        num_examples = len(next(iter(processed_examples.values())))
        
        # Process each field for each example
        for idx in range(num_examples):
            # Extract single example
            single_example = {
                key: values[idx] 
                for key, values in processed_examples.items()
            }
            
            # Convert to new format
            converted = convert_sample_to_new_format(single_example)
            
            # Validate if requested
            if validate and not validate_new_format(converted):
                print(f"Warning: Sample {idx} failed validation after conversion")
            
            # Put converted values back
            for key, value in converted.items():
                if key in processed_examples:
                    processed_examples[key][idx] = value
        
        return processed_examples
    
    # Process with batching for efficiency
    converted_dataset = dataset.map(
        convert_batch,
        batched=True,
        batch_size=1000,
        desc="Converting to new format"
    )
    
    return converted_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Convert datasets from old chat format to new chat format"
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to input dataset in old format"
    )
    parser.add_argument(
        "output_path",
        type=str,
        help="Path to save converted dataset in new format"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        default=True,
        help="Validate each sample after conversion (default: True)"
    )
    parser.add_argument(
        "--no-validate",
        dest="validate",
        action="store_false",
        help="Skip validation of converted samples"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Only convert first N samples (for testing)"
    )
    
    args = parser.parse_args()
    
    # Load input dataset
    print(f"Loading dataset from: {args.input_path}")
    dataset = load_from_disk(args.input_path)
    
    # Handle both Dataset and DatasetDict
    if isinstance(dataset, DatasetDict):
        print(f"Processing DatasetDict with splits: {list(dataset.keys())}")
        converted_datasets = {}
        
        for split_name, split_dataset in dataset.items():
            print(f"\nProcessing split: {split_name}")
            
            # Sample if requested
            if args.sample:
                split_dataset = split_dataset.select(range(min(args.sample, len(split_dataset))))
                print(f"  Sampling first {len(split_dataset)} examples")
            
            # Convert the split
            converted_split = process_dataset(split_dataset, validate=args.validate)
            converted_datasets[split_name] = converted_split
            
            print(f"  Converted {len(converted_split)} samples")
        
        # Create new DatasetDict
        converted_dataset = DatasetDict(converted_datasets)
    else:
        # Single Dataset
        print("Processing single Dataset")
        
        # Sample if requested
        if args.sample:
            dataset = dataset.select(range(min(args.sample, len(dataset))))
            print(f"  Sampling first {len(dataset)} examples")
        
        # Convert and wrap in DatasetDict per project requirements
        converted_split = process_dataset(dataset, validate=args.validate)
        converted_dataset = DatasetDict({"train": converted_split})
        
        print(f"Converted {len(converted_split)} samples")
    
    # Calculate total samples for processing log
    total_samples = sum(len(split_dataset) for split_dataset in converted_dataset.values())
    
    # Save converted dataset
    output_path = Path(args.output_path)
    
    # If output_path is a directory, use the same dataset name as input
    if output_path.is_dir() or args.output_path.endswith('/'):
        input_name = Path(args.input_path).name
        output_path = output_path / input_name
    
    print(f"\nSaving converted dataset to: {output_path}")
    save_dataset_with_processing_log(converted_dataset, output_path, args.input_path, total_samples)
    
    # Print summary
    print("\n" + "="*50)
    print("Conversion Summary:")
    print("="*50)
    for split_name, split_dataset in converted_dataset.items():
        print(f"  {split_name}: {len(split_dataset)} samples")
    
    # Show example of converted format
    if len(converted_dataset) > 0:
        first_split = list(converted_dataset.keys())[0]
        first_sample = converted_dataset[first_split][0]
        
        print("\nExample converted sample (first message only):")
        print("-"*50)
        
        if "conversation_branches" in first_sample and first_sample["conversation_branches"]:
            branch = first_sample["conversation_branches"][0]
            if "messages" in branch and branch["messages"]:
                first_message = branch["messages"][0]
                print(json.dumps(first_message, indent=2)[:500] + "...")
    
    print("\nConversion complete!")


if __name__ == "__main__":
    main()