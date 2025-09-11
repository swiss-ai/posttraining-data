#!/usr/bin/env python3
"""
Convert WikiQA dataset from existing chat format to new format with parts structure.

This converter handles WikiQA's specific structure and converts it to our 
dataset schema while preserving all original metadata.
"""

import sys
import json
import os
import copy
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import pyarrow as pa
import pandas as pd
from datasets import Dataset, DatasetDict
from tqdm import tqdm


def load_wikiqa_from_arrow(dataset_path: str) -> List[Dict[str, Any]]:
    """Load WikiQA dataset directly from Arrow files, bypassing corrupted metadata."""
    print(f"Loading WikiQA from Arrow files in: {dataset_path}")
    
    # Find all Arrow files
    arrow_files = []
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.arrow'):
                arrow_files.append(os.path.join(root, file))
    
    if not arrow_files:
        raise ValueError(f"No Arrow files found in {dataset_path}")
    
    print(f"Found {len(arrow_files)} Arrow files")
    
    # Read all Arrow files
    all_samples = []
    for arrow_file in tqdm(arrow_files, desc="Reading Arrow files"):
        try:
            # Use stream reader (worked in our testing)
            with open(arrow_file, 'rb') as f:
                reader = pa.ipc.open_stream(f)
                table = reader.read_all()
                df = table.to_pandas()
                
                # Convert to list of dicts
                samples = df.to_dict('records')
                all_samples.extend(samples)
                print(f"  {os.path.basename(arrow_file)}: {len(samples):,} samples")
                
        except Exception as e:
            print(f"  Failed to read {arrow_file}: {e}")
            continue
    
    if not all_samples:
        raise ValueError("Failed to read any samples from Arrow files")
    
    print(f"Total samples loaded: {len(all_samples):,}")
    return all_samples


def convert_wikiqa_to_newformat(sample: Dict[str, Any], dataset_source: str, row_index: Optional[int] = None) -> Dict[str, Any]:
    """
    Convert WikiQA sample from existing chat format to new format with parts structure.
    
    WikiQA existing format:
    - conversation_id: str
    - dataset_source: str  
    - language_code: str
    - language_name: str
    - initial_prompt: {role: str, content: str, metadata: dict}
    - conversation_branches: [{messages: [{role: str, content: str, metadata: dict}]}]
    - created_timestamp: str
    
    New format:
    - conversation_id: str (preserved)
    - dataset_source: str (preserved)
    - original_metadata: dict (contains all extra fields)
    - system_prompt: {content: str, metadata: dict}
    - initial_prompt: {role: str, content: str, metadata: dict} (preserved)
    - available_functions: [] (empty)
    - conversation_branches: [{messages: [{role: str, parts: [{type: str, content: str, metadata: dict, name: str, args: str}]}]}]
    - created_timestamp: str (preserved)
    """
    
    # Create new sample structure
    converted = {
        "conversation_id": sample.get("conversation_id", f"wikiqa_{row_index}"),
        "dataset_source": sample.get("dataset_source", dataset_source),
        "original_metadata": {},
        "system_prompt": {
            "content": "",
            "metadata": {}
        },
        "initial_prompt": {},
        "available_functions": [],
        "conversation_branches": [],
        "created_timestamp": sample.get("created_timestamp", datetime.now().isoformat())
    }
    
    # Store extra metadata in original_metadata
    extra_fields = {}
    for key, value in sample.items():
        if key not in ["conversation_id", "dataset_source", "initial_prompt", "conversation_branches", "created_timestamp"]:
            extra_fields[key] = value
    
    converted["original_metadata"] = extra_fields
    
    # Preserve initial_prompt structure
    if "initial_prompt" in sample and sample["initial_prompt"]:
        converted["initial_prompt"] = copy.deepcopy(sample["initial_prompt"])
    else:
        # Fallback if missing
        converted["initial_prompt"] = {
            "role": "user",
            "content": "",
            "metadata": {}
        }
    
    # Convert conversation_branches to new parts format
    if "conversation_branches" in sample and sample["conversation_branches"]:
        new_branches = []
        
        for branch_idx, branch in enumerate(sample["conversation_branches"]):
            if not isinstance(branch, dict) or "messages" not in branch:
                print(f"Warning: Invalid branch structure in sample {converted['conversation_id']}, branch {branch_idx}")
                continue
            
            new_messages = []
            
            for msg_idx, message in enumerate(branch["messages"]):
                if not isinstance(message, dict):
                    print(f"Warning: Invalid message structure in sample {converted['conversation_id']}, branch {branch_idx}, message {msg_idx}")
                    continue
                
                # Convert message content to parts structure
                message_content = message.get("content", "")
                message_role = message.get("role", "assistant")
                message_metadata = message.get("metadata", {})
                
                # Create new message with parts structure
                new_message = {
                    "role": message_role,
                    "parts": [
                        {
                            "type": "response",
                            "content": message_content,
                            "metadata": message_metadata,
                            "name": "",
                            "args": ""
                        }
                    ]
                }
                
                new_messages.append(new_message)
            
            # Create new branch
            new_branch = {"messages": new_messages}
            new_branches.append(new_branch)
        
        converted["conversation_branches"] = new_branches
    else:
        # Empty conversation branches
        converted["conversation_branches"] = []
    
    return converted


def validate_converted_sample(sample: Dict[str, Any]) -> bool:
    """Validate that the converted sample follows the new format specification."""
    
    # Check required top-level fields
    required_fields = [
        "conversation_id", "dataset_source", "original_metadata", 
        "system_prompt", "initial_prompt", "available_functions",
        "conversation_branches", "created_timestamp"
    ]
    
    for field in required_fields:
        if field not in sample:
            print(f"Missing required field: {field}")
            return False
    
    # Validate system_prompt structure
    if not isinstance(sample["system_prompt"], dict):
        print("system_prompt must be a dict")
        return False
    
    if "content" not in sample["system_prompt"] or "metadata" not in sample["system_prompt"]:
        print("system_prompt missing content or metadata")
        return False
    
    # Validate initial_prompt structure  
    if not isinstance(sample["initial_prompt"], dict):
        print("initial_prompt must be a dict")
        return False
    
    required_prompt_fields = ["role", "content", "metadata"]
    for field in required_prompt_fields:
        if field not in sample["initial_prompt"]:
            print(f"initial_prompt missing field: {field}")
            return False
    
    # Validate available_functions is a list
    if not isinstance(sample["available_functions"], list):
        print("available_functions must be a list")
        return False
    
    # Validate conversation_branches structure
    if not isinstance(sample["conversation_branches"], list):
        print("conversation_branches must be a list")
        return False
    
    for branch_idx, branch in enumerate(sample["conversation_branches"]):
        if not isinstance(branch, dict) or "messages" not in branch:
            print(f"Branch {branch_idx} invalid structure")
            return False
        
        if not isinstance(branch["messages"], list):
            print(f"Branch {branch_idx} messages must be a list")
            return False
        
        for msg_idx, message in enumerate(branch["messages"]):
            if not isinstance(message, dict):
                print(f"Message {msg_idx} in branch {branch_idx} must be a dict")
                return False
            
            if "role" not in message or "parts" not in message:
                print(f"Message {msg_idx} in branch {branch_idx} missing role or parts")
                return False
            
            if not isinstance(message["parts"], list):
                print(f"Message {msg_idx} in branch {branch_idx} parts must be a list")
                return False
            
            for part_idx, part in enumerate(message["parts"]):
                if not isinstance(part, dict):
                    print(f"Part {part_idx} in message {msg_idx}, branch {branch_idx} must be a dict")
                    return False
                
                required_part_fields = ["type", "content", "metadata", "name", "args"]
                for field in required_part_fields:
                    if field not in part:
                        print(f"Part {part_idx} missing field: {field}")
                        return False
    
    return True


def process_wikiqa_dataset(input_path: str, output_path: str, dataset_name: str = "WikiQA"):
    """Process the entire WikiQA dataset."""
    
    print(f"Converting WikiQA dataset: {dataset_name}")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print("="*60)
    
    # Load samples from Arrow files
    samples = load_wikiqa_from_arrow(input_path)
    
    # Convert samples
    print(f"\nConverting {len(samples):,} samples to new format...")
    converted_samples = []
    failed_count = 0
    
    for idx, sample in enumerate(tqdm(samples, desc="Converting")):
        try:
            converted = convert_wikiqa_to_newformat(sample, dataset_name, idx)
            
            # Validate conversion
            if validate_converted_sample(converted):
                converted_samples.append(converted)
            else:
                print(f"Sample {idx} failed validation")
                failed_count += 1
                
        except Exception as e:
            print(f"Error converting sample {idx}: {e}")
            failed_count += 1
    
    print(f"Successfully converted: {len(converted_samples):,}")
    print(f"Failed conversions: {failed_count}")
    
    if not converted_samples:
        raise ValueError("No samples were successfully converted")
    
    # Create Dataset
    print("Creating dataset...")
    dataset = Dataset.from_list(converted_samples)
    
    # Wrap in DatasetDict for consistency
    dataset_dict = DatasetDict({"train": dataset})
    
    # Save dataset
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving dataset to: {output_path}")
    dataset_dict.save_to_disk(str(output_path))
    
    # Analyze languages if available
    languages = set()
    for sample in converted_samples[:1000]:  # Sample first 1000 to check languages
        if "language_code" in sample.get("original_metadata", {}):
            languages.add(sample["original_metadata"]["language_code"])
    
    # Create metadata
    metadata = {
        "dataset_name": dataset_name,
        "source_format": "wikiqa_chat_format",
        "target_format": "new_chat_format_with_parts",
        "conversion_script": "convert_wikiqa_to_newformat.py",
        "total_samples": len(converted_samples),
        "failed_conversions": failed_count,
        "languages_detected": sorted(list(languages)),
        "processing_log": [
            {
                "operation": "wikiqa_format_conversion",
                "script": "02-standardisation/convert_wikiqa_to_newformat.py",
                "timestamp": datetime.now().isoformat(),
                "input_path": input_path,
                "output_path": str(output_path),
                "conversion_type": "wikiqa_to_new_chat_with_parts",
                "samples_input": len(samples),
                "samples_output": len(converted_samples),
                "samples_failed": failed_count,
                "preservation": "original_metadata_preserved"
            }
        ]
    }
    
    # Save metadata
    with open(output_path / "dataset_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"✓ WikiQA conversion completed successfully")
    print(f"Original samples: {len(samples):,}")
    print(f"Converted samples: {len(converted_samples):,}")
    print(f"Failed conversions: {failed_count}")
    if languages:
        print(f"Languages detected: {', '.join(sorted(languages))}")
    print(f"Output: {output_path}")
    print(f"Format: New chat format with parts structure")
    print(f"{'='*60}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Convert WikiQA dataset to new chat format with parts structure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert WikiQA dataset
  python convert_wikiqa_to_newformat.py \\
    /capstor/store/cscs/swissai/infra01/posttrain_data/01_raw_hf_data/WikiQA \\
    data/02-standardised/WikiQA
    
  # Convert with custom dataset name
  python convert_wikiqa_to_newformat.py \\
    /path/to/WikiQA \\
    data/02-standardised/WikiQA-converted \\
    --dataset-name "WikiQA-Multilingual"
        """
    )
    
    parser.add_argument(
        "input_path",
        help="Path to WikiQA dataset directory (with Arrow files)"
    )
    parser.add_argument(
        "output_path",
        help="Path for output dataset directory"
    )
    parser.add_argument(
        "--dataset-name",
        default="WikiQA",
        help="Name for the dataset (default: WikiQA)"
    )
    
    args = parser.parse_args()
    
    # Validate input
    input_path = Path(args.input_path)
    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}")
        sys.exit(1)
    
    # Process dataset
    try:
        process_wikiqa_dataset(str(input_path), args.output_path, args.dataset_name)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()