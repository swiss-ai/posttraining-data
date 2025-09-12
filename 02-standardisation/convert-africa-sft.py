#!/usr/bin/env python3
"""
Convert africa-sft dataset to new standardized chat format with parts structure.

This dataset has a specific structure where:
- initial_prompt is null
- conversation_branches contains messages with reversed roles:
  - "assistant" role contains the initial prompt/question
  - "user" role contains the actual assistant response
"""

import sys
import json
import ast
import argparse
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
from datasets import load_from_disk, Dataset, DatasetDict
from tqdm import tqdm


def create_schema_compliant_part(part_type: str, content: str = "", metadata: Optional[Dict] = None, 
                                 name: str = "", args: str = "") -> Dict[str, Any]:
    """Create a schema-compliant part with all required fields for Arrow compatibility."""
    return {
        "type": part_type,
        "content": content,
        "metadata": metadata or {},
        "name": name,
        "args": args
    }


def generate_conversation_id(dataset_source: str, content: str, sample_id: Optional[str] = None) -> str:
    """Generate a globally unique conversation ID."""
    dataset_prefix = dataset_source.replace('/', '_').replace('-', '_')
    
    if sample_id:
        # Use provided ID + short content hash for verification
        content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()[:8]
        return f"{dataset_prefix}_{sample_id}_{content_hash}"
    else:
        # Fallback to longer content hash for uniqueness
        content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]
        return f"{dataset_prefix}_{content_hash}"


def convert_africa_sft_format(row: Dict[str, Any], dataset_source: str, row_index: Optional[int] = None) -> Dict[str, Any]:
    """
    Convert africa-sft format to new standardized chat format.
    
    The africa-sft format has:
    - initial_prompt: null (needs to be extracted from first assistant message)
    - conversation_branches: string containing list with messages where:
      - First message with role "assistant" contains the initial user prompt/question
      - Second message with role "user" contains the assistant's response
      - Roles are swapped throughout the conversation
    - original_metadata: string with metadata
    - dataset_source: string
    """
    # Parse conversation_branches from string to list
    conversation_branches_str = row.get("conversation_branches", "[]")
    try:
        branches = ast.literal_eval(conversation_branches_str)
    except (ValueError, SyntaxError) as e:
        raise ValueError(f"Failed to parse conversation_branches: {e}")
    
    if not branches or not branches[0].get("messages"):
        raise ValueError("No valid messages found in conversation_branches")
    
    messages = branches[0]["messages"]
    
    # The first message (marked as "assistant") is actually the initial user prompt
    if len(messages) < 2:
        raise ValueError(f"Expected at least 2 messages, got {len(messages)}")
    
    if messages[0]["role"] != "assistant" or messages[1]["role"] != "user":
        raise ValueError(f"Unexpected role pattern: {[m['role'] for m in messages]}")
    
    # Extract the actual initial prompt (from first "assistant" message which is really the user's question)
    initial_prompt_content = messages[0]["content"]
    
    # Filter out samples containing "chatgpt" (case insensitive)
    if "chatgpt" in initial_prompt_content.lower():
        return None  # Skip this sample
    
    # Filter out samples starting with the style tag
    if initial_prompt_content.strip().startswith("<style>.atomic-structure-template-field-8x8"):
        return None  # Skip this sample
    
    # Extract the actual assistant response (from second "user" message which is really the assistant's answer)
    assistant_response_content = messages[1]["content"]
    
    # Generate conversation ID
    conversation_id = generate_conversation_id(dataset_source, initial_prompt_content, f"row_{row_index}")
    
    # Create initial user prompt
    initial_prompt = {
        "role": "user",
        "content": initial_prompt_content,
        "metadata": {}
    }
    
    # Create conversation with corrected roles and parts structure
    conversation_messages = [{
        "role": "assistant",
        "parts": [create_schema_compliant_part("response", assistant_response_content, {})]
    }]
    
    # Add any additional messages (if they exist)
    for i in range(2, len(messages), 2):
        if i + 1 < len(messages):
            # Next pair: "assistant" is user, "user" is assistant
            user_msg = messages[i]
            assistant_msg = messages[i + 1]
            
            # Add user message
            conversation_messages.append({
                "role": "user",
                "parts": [create_schema_compliant_part("response", user_msg["content"], {})]
            })
            
            # Add assistant message
            conversation_messages.append({
                "role": "assistant",
                "parts": [create_schema_compliant_part("response", assistant_msg["content"], {})]
            })
    
    # Parse original metadata if it's a string
    original_metadata_str = row.get("original_metadata", "{}")
    if isinstance(original_metadata_str, str):
        try:
            original_metadata = ast.literal_eval(original_metadata_str)
        except (ValueError, SyntaxError):
            original_metadata = {"raw_metadata": original_metadata_str}
    else:
        original_metadata = original_metadata_str
    
    # Add the original dataset_source to metadata
    original_metadata["original_dataset_source"] = row.get("dataset_source", "")
    
    # Create single conversation branch
    conversation_branches = [{
        "messages": conversation_messages
    }]
    
    result = {
        "conversation_id": conversation_id,
        "dataset_source": dataset_source,
        "original_metadata": original_metadata,
        "system_prompt": {"content": "", "metadata": {}},
        "initial_prompt": initial_prompt,
        "available_functions": [],
        "conversation_branches": conversation_branches,
        "created_timestamp": datetime.now().isoformat()
    }
    
    return result


def convert_dataset(dataset: Dataset, dataset_source: str) -> Dataset:
    """Convert dataset with validation."""
    def convert_with_validation(example, idx):
        try:
            converted = convert_africa_sft_format(example, dataset_source, idx)
            
            if converted is None:
                return None
            
            if not isinstance(converted, dict):
                print(f"Warning: Converter returned non-dict for sample {idx}: {type(converted)}")
                return None
            
            return converted
            
        except Exception as e:
            print(f"Warning: Failed to convert sample {idx}: {e}")
            return None
    
    print(f"Converting and validating {len(dataset):,} samples...")
    
    # Use dataset.map with enumeration for index
    converted = dataset.map(
        convert_with_validation,
        with_indices=True,
        desc="Converting",
        remove_columns=dataset.column_names
    )
    
    # Filter out None results (invalid/failed conversions)
    initial_count = len(converted)
    converted = converted.filter(lambda x: x is not None)
    final_count = len(converted)
    
    if initial_count > final_count:
        print(f"Filtered out {initial_count - final_count:,} invalid samples ({final_count:,} remain)")
    
    return converted


def load_existing_metadata(input_path: Path) -> Optional[Dict[str, Any]]:
    """Load existing dataset metadata if it exists."""
    metadata_file = Path(input_path) / "dataset_metadata.json"
    if metadata_file.exists():
        try:
            with open(metadata_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return None


def save_dataset_and_metadata(dataset, output_path: Path, dataset_name: str, input_path: Path):
    """Save converted dataset and update metadata."""
    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Always save as DatasetDict with 'train' split for pipeline compatibility
    if isinstance(dataset, DatasetDict):
        output_dataset = dataset
    else:
        output_dataset = DatasetDict({"train": dataset})
    
    # Save dataset
    print(f"Saving dataset to {output_path}...")
    output_dataset.save_to_disk(str(output_path))
    
    # Load or create metadata
    metadata = load_existing_metadata(input_path) or {}
    
    # Add processing log entry
    processing_entry = {
        "operation": "standardisation",
        "script": "convert_africa_sft.py",
        "timestamp": datetime.now().isoformat(),
        "input_path": str(input_path),
        "output_path": str(output_path),
        "samples_processed": len(dataset) if not isinstance(dataset, DatasetDict) else sum(len(split) for split in dataset.values()),
        "conversion_success": True,
        "target_schema": "chat_format_v1.0",
        "filters_applied": [
            "filtered_chatgpt_mentions",
            "filtered_style_tags"
        ]
    }
    
    if "processing_log" not in metadata:
        metadata["processing_log"] = []
    metadata["processing_log"].append(processing_entry)
    
    # Save updated metadata
    metadata_file = output_path / "dataset_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {metadata_file}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert africa-sft dataset to new chat format with parts structure",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "input_path",
        help="Path to input africa-sft dataset directory"
    )
    parser.add_argument(
        "output_path", 
        help="Path for output dataset directory"
    )
    parser.add_argument(
        "--dataset-name",
        default="africa-sft",
        help="Name for the dataset (default: africa-sft)"
    )
    
    return parser.parse_args()


def main():
    """Main conversion function."""
    args = parse_arguments()
    
    # Validate input path
    input_path = Path(args.input_path)
    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}")
        sys.exit(1)
    
    print(f"Converting dataset: {args.dataset_name}")
    print(f"Input:  {input_path}")
    print(f"Output: {args.output_path}")
    
    # Load input dataset
    print(f"Loading dataset from: {input_path}")
    try:
        dataset = load_from_disk(str(input_path))
        
        # Handle DatasetDict vs single Dataset
        if hasattr(dataset, 'keys'):
            # DatasetDict - use 'train' split or first available split
            available_splits = list(dataset.keys())
            print(f"Found DatasetDict with splits: {available_splits}")
            
            if 'train' in available_splits:
                input_dataset = dataset['train']
                print(f"Using 'train' split with {len(input_dataset):,} samples")
            else:
                split_name = available_splits[0]
                input_dataset = dataset[split_name]
                print(f"Using '{split_name}' split with {len(input_dataset):,} samples")
        else:
            # Single Dataset
            input_dataset = dataset
            print(f"Loaded single dataset with {len(input_dataset):,} samples")
        
        # Convert dataset
        print(f"Converting africa-sft format...")
        converted_dataset = convert_dataset(input_dataset, args.dataset_name)
        
        # Save output
        output_path = Path(args.output_path)
        save_dataset_and_metadata(converted_dataset, output_path, args.dataset_name, input_path)
        
        print(f"\n✅ Conversion complete!")
        print(f"Input:  {input_path}")
        print(f"Output: {output_path}")
        print(f"Samples: {len(converted_dataset):,}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()