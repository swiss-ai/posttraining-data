#!/usr/bin/env python3
"""
Browse and inspect samples which follow the data schema.

This script loads a dataset and displays sample conversations in the format
where assistant and user messages contain 'parts' lists with typed components.

Supports field-based filtering to browse only samples matching specific criteria.
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional
from datasets import load_from_disk
from datetime import datetime


def get_nested_value(obj: Dict[str, Any], field_path: str) -> Any:
    """
    Get value from nested dictionary using dot notation and array indexing.
    
    Args:
        obj: Dictionary to search in
        field_path: Dot-separated path like 'original_metadata.category' or 'messages[0].role'
    
    Returns:
        Value at the specified path, or None if not found
    """
    try:
        current = obj
        
        # Replace array notation with dots for parsing: messages[0].role -> messages.0.role
        normalized_path = field_path.replace('[', '.').replace(']', '')
        parts = normalized_path.split('.')
        
        for part in parts:
            if part.isdigit():
                # It's an array index
                index = int(part)
                if isinstance(current, list) and 0 <= index < len(current):
                    current = current[index]
                else:
                    return None
            else:
                # It's a dictionary key
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    return None
        
        return current
    except (KeyError, TypeError, AttributeError, IndexError):
        return None


def check_filter_match(sample: Dict[str, Any], field_path: Optional[str], 
                      include_values: Optional[List[str]], 
                      exclude_values: Optional[List[str]]) -> bool:
    """
    Check if a sample matches the filter criteria.
    
    Args:
        sample: Sample to check
        field_path: Field path to check
        include_values: Values to include (if specified, sample must have one of these)
        exclude_values: Values to exclude (if specified, sample must not have any of these)
    
    Returns:
        True if sample matches filter criteria, False otherwise
    """
    # If no filter specified, match everything
    if not field_path:
        return True
    
    # Get the field value
    field_value = get_nested_value(sample, field_path)
    field_value_str = str(field_value) if field_value is not None else "<NULL>"
    
    # Check exclude list first (takes precedence)
    if exclude_values and field_value_str in exclude_values:
        return False
    
    # Check include list
    if include_values and field_value_str not in include_values:
        return False
    
    return True


def pretty_print_sample(sample: dict, sample_idx: int = 0, total_samples: int = 0,
                       filtered_idx: Optional[int] = None, total_filtered: Optional[int] = None):
    """Pretty print a single chat format sample."""
    print(f"\n{'═'*80}")
    if filtered_idx is not None and total_filtered is not None:
        print(f"SAMPLE {sample_idx + 1} of {total_samples} (Filtered: {filtered_idx + 1} of {total_filtered})")
    else:
        print(f"SAMPLE {sample_idx + 1} of {total_samples}")
    print('═'*80)
    
    # Basic info
    print(f"\n┌─ Basic Information")
    print(f"├─ Conversation ID: {sample.get('conversation_id', 'N/A')}")
    print(f"├─ Dataset Source: {sample.get('dataset_source', 'N/A')}")
    print(f"└─ Created: {sample.get('created_timestamp', 'N/A')}")
    
    # System prompt
    if 'system_prompt' in sample and sample['system_prompt']:
        print(f"\n┌─ System Prompt")
        print("│")
        for line in sample['system_prompt']['content'].split('\n'):
            print(f"│  {line}")
        print("│")
        if sample['system_prompt'].get('metadata'):
            print(f"├─ Metadata ({len(sample['system_prompt']['metadata'])} fields):")
            def print_nested_dict(d, indent="│    ", is_last=False):
                items = list(d.items())
                for i, (key, value) in enumerate(items):
                    is_last_item = i == len(items) - 1
                    if isinstance(value, dict):
                        print(f"{indent}├─ {key}:")
                        print_nested_dict(value, indent + "│  ", is_last_item)
                    else:
                        connector = "└─" if is_last_item else "├─"
                        print(f"{indent}{connector} {key}: {value}")
            print_nested_dict(sample['system_prompt']['metadata'])
        print("└─")
    
    # Initial prompt
    if 'initial_prompt' in sample:
        print(f"\n┌─ Initial Prompt (Role: {sample['initial_prompt']['role']})")
        print("│")
        for line in sample['initial_prompt']['content'].split('\n'):
            print(f"│  {line}")
        print("│")
        if sample['initial_prompt'].get('metadata'):
            print(f"├─ Metadata ({len(sample['initial_prompt']['metadata'])} fields):")
            def print_nested_dict(d, indent="│    ", is_last=False):
                items = list(d.items())
                for i, (key, value) in enumerate(items):
                    is_last_item = i == len(items) - 1
                    if isinstance(value, dict):
                        print(f"{indent}├─ {key}:")
                        print_nested_dict(value, indent + "│  ", is_last_item)
                    else:
                        connector = "└─" if is_last_item else "├─"
                        print(f"{indent}{connector} {key}: {value}")
            print_nested_dict(sample['initial_prompt']['metadata'])
        print("└─")
    
    # Available functions (for augmented datasets)
    if 'available_functions' in sample and sample['available_functions']:
        print(f"\n┌─ Available Functions ({len(sample['available_functions'])} functions)")
        for i, func in enumerate(sample['available_functions']):
            is_last = i == len(sample['available_functions']) - 1
            connector = "└─" if is_last else "├─"
            print(f"{connector} {func.get('name', 'unnamed')}: {func.get('description', 'No description')[:80]}{'...' if len(func.get('description', '')) > 80 else ''}")
    
    # Conversation branches  
    branches = sample.get('conversation_branches', [])
    print(f"\n┌─ Conversation Branches ({len(branches)} branch{'es' if len(branches) != 1 else ''})")
    
    for branch_idx, branch in enumerate(branches):
        is_last_branch = branch_idx == len(branches) - 1
        branch_connector = "└─" if is_last_branch else "├─"
        branch_line = "   " if is_last_branch else "│  "
        
        print(f"│")
        print(f"{branch_connector} Branch {branch_idx + 1}:")
        messages = branch.get('messages', [])
        
        for msg_idx, msg in enumerate(messages):
            is_last_msg = msg_idx == len(messages) - 1
            msg_connector = "└─" if is_last_msg else "├─"
            msg_line = "   " if is_last_msg else "│  "
            
            print(f"{branch_line} │")
            print(f"{branch_line} {msg_connector} Message {msg_idx + 1} ({msg['role'].title()}):")
            print(f"{branch_line} {msg_line} │")
            
            # Handle 'parts' field
            parts = msg.get('parts', [])
            if not parts:
                print(f"{branch_line} {msg_line} │  [No parts found]")
                continue
                
            print(f"{branch_line} {msg_line} │  [{len(parts)} part{'s' if len(parts) != 1 else ''}]")
            
            for part_idx, part in enumerate(parts):
                is_last_part = part_idx == len(parts) - 1
                part_connector = "└─" if is_last_part else "├─"
                part_line = "   " if is_last_part else "│  "
                
                if isinstance(part, dict):
                    print(f"{branch_line} {msg_line} │  │")
                    
                    # Get the type of the part
                    part_type = part.get('type', 'unknown')
                    print(f"{branch_line} {msg_line} │  {part_connector} {part_type.upper()}:")
                    
                    # Handle different part types
                    if part_type == 'function-call':
                        print(f"{branch_line} {msg_line} │  {part_line} Name: {part.get('name', 'N/A')}")
                        if 'args' in part:
                            print(f"{branch_line} {msg_line} │  {part_line} Args: {json.dumps(part['args'], indent=0)}")
                    elif part_type == 'verifiable-responses':
                        # Handle verifiable responses with answers list
                        answers = part.get('answers', [])
                        print(f"{branch_line} {msg_line} │  {part_line} [{len(answers)} answer{'s' if len(answers) != 1 else ''}]")
                        for idx, answer in enumerate(answers):
                            print(f"{branch_line} {msg_line} │  {part_line}   {idx + 1}. {answer}")
                    else:
                        # For response, thought, function-output, etc.
                        part_content = part.get('content', '')
                        if isinstance(part_content, str):
                            for line in part_content.split('\n'):
                                print(f"{branch_line} {msg_line} │  {part_line} {line}")
                        else:
                            print(f"{branch_line} {msg_line} │  {part_line} {part_content}")
                    
                    # Print metadata if available
                    if part.get('metadata'):
                        print(f"{branch_line} {msg_line} │  {part_line}")
                        print(f"{branch_line} {msg_line} │  {part_line} ├─ Metadata ({len(part['metadata'])} fields):")
                        def print_part_nested_dict(d, indent=""):
                            items = list(d.items())
                            for i, (key, value) in enumerate(items):
                                is_last_item = i == len(items) - 1
                                if isinstance(value, dict):
                                    print(f"{branch_line} {msg_line} │  {part_line} │  {indent}├─ {key}:")
                                    print_part_nested_dict(value, indent + "│  ")
                                else:
                                    connector = "└─" if is_last_item else "├─"
                                    if isinstance(value, str) and len(value) > 100:
                                        # Truncate long strings
                                        print(f"{branch_line} {msg_line} │  {part_line} │  {indent}{connector} {key}: {value[:100]}...")
                                    else:
                                        print(f"{branch_line} {msg_line} │  {part_line} │  {indent}{connector} {key}: {value}")
                        print_part_nested_dict(part['metadata'])
                        print(f"{branch_line} {msg_line} │  {part_line} └─")
                else:
                    # Fallback for unexpected format
                    print(f"{branch_line} {msg_line} │  {part_connector} {part}")
            
            if msg.get('metadata'):
                print(f"{branch_line} {msg_line} │")
                print(f"{branch_line} {msg_line} ├─ Metadata ({len(msg['metadata'])} fields):")
                def print_nested_dict(d, indent="", is_last=False):
                    items = list(d.items())
                    for i, (key, value) in enumerate(items):
                        is_last_item = i == len(items) - 1
                        if isinstance(value, dict):
                            print(f"{branch_line} {msg_line} │  {indent}├─ {key}:")
                            print_nested_dict(value, indent + "│  ", is_last_item)
                        else:
                            connector = "└─" if is_last_item else "├─"
                            print(f"{branch_line} {msg_line} │  {indent}{connector} {key}: {value}")
                print_nested_dict(msg['metadata'])
            print(f"{branch_line} {msg_line} └─")
    
    # Original metadata
    if 'original_metadata' in sample and sample['original_metadata']:
        print(f"\n┌─ Original Metadata ({len(sample['original_metadata'])} fields)")
        items = list(sample['original_metadata'].items())
        for i, (key, value) in enumerate(items):
            is_last = i == len(items) - 1
            connector = "└─" if is_last else "├─"
            if isinstance(value, str) and len(value) > 50:
                print(f"{connector} {key}: {value[:50]}...")
            else:
                print(f"{connector} {key}: {value}")


def browse_dataset(dataset_path: str, num_samples: int = 1, start_idx: int = 0,
                  raw_json: bool = False, split: str = None,
                  field_path: Optional[str] = None,
                  include_values: Optional[List[str]] = None,
                  exclude_values: Optional[List[str]] = None):
    """Browse samples from a dataset with optional field filtering."""
    path = Path(dataset_path)
    
    if not path.exists():
        print(f"Error: Dataset path does not exist: {dataset_path}")
        return False
    
    try:
        print(f"Loading dataset from: {dataset_path}")
        dataset = load_from_disk(str(path))
        
        # Handle DatasetDict vs single Dataset
        if hasattr(dataset, 'keys'):
            available_splits = list(dataset.keys())
            print(f"\n┌─ Dataset Information")
            print(f"├─ Type: DatasetDict")
            print(f"├─ Available splits: {', '.join(available_splits)}")
            
            # Split selection logic
            if split:
                if split in available_splits:
                    dataset = dataset[split]
                    selected_split = split
                else:
                    print(f"│  ⚠️  Warning: Split '{split}' not found")
                    print(f"│  Available splits: {', '.join(available_splits)}")
                    selected_split = available_splits[0]
                    dataset = dataset[selected_split]
                    print(f"│  Using first available split: '{selected_split}'")
            elif 'train' in available_splits:
                dataset = dataset['train']
                selected_split = 'train'
            else:
                selected_split = available_splits[0]
                dataset = dataset[selected_split]
            
            print(f"├─ Selected split: '{selected_split}'")
            print(f"└─ Split size: {len(dataset):,} samples")
        else:
            print(f"\n┌─ Dataset Information")
            print(f"├─ Type: Single Dataset (no splits)")
            print(f"└─ Size: {len(dataset):,} samples")
        
        total_samples = len(dataset)
        
        # Display filter information if filtering is active
        if field_path:
            print(f"\n┌─ Filter Settings")
            print(f"├─ Field: {field_path}")
            if include_values:
                quoted_values = ['"' + v + '"' for v in include_values]
                print(f"├─ Include values: {', '.join(quoted_values)}")
            if exclude_values:
                quoted_values = ['"' + v + '"' for v in exclude_values]
                print(f"├─ Exclude values: {', '.join(quoted_values)}")
            
            # Count matching samples
            print(f"│  Counting matching samples...")
            matching_indices = []
            for i in range(total_samples):
                if check_filter_match(dataset[i], field_path, include_values, exclude_values):
                    matching_indices.append(i)
            
            total_filtered = len(matching_indices)
            print(f"└─ Matching samples: {total_filtered:,} of {total_samples:,} ({(total_filtered/total_samples)*100:.1f}%)")
            
            if total_filtered == 0:
                print(f"\n⚠️  No samples match the filter criteria")
                return True
        else:
            matching_indices = list(range(total_samples))
            total_filtered = total_samples
        
        # Validate indices
        if field_path and matching_indices:
            # Adjust start_idx for filtered results
            if start_idx >= len(matching_indices):
                print(f"Error: Start index {start_idx} is beyond filtered dataset size {len(matching_indices)}")
                return False
        elif start_idx >= total_samples:
            print(f"Error: Start index {start_idx} is beyond dataset size {total_samples}")
            return False
        
        if field_path and matching_indices:
            # Use filtered indices
            end_idx = min(start_idx + num_samples, len(matching_indices))
            actual_samples = end_idx - start_idx
        else:
            end_idx = min(start_idx + num_samples, total_samples)
            actual_samples = end_idx - start_idx
        
        if field_path and matching_indices:
            if num_samples == 1:
                print(f"Interactive browsing mode (filtered). Press Enter for next, 'q' to quit, or number to jump")
            else:
                print(f"Showing filtered samples {start_idx + 1} to {end_idx} ({actual_samples} samples)")
        else:
            if num_samples == 1:
                print(f"Interactive browsing mode. Press Enter for next, 'q' to quit, or number to jump")
            else:
                print(f"Showing samples {start_idx + 1} to {end_idx} ({actual_samples} samples)")
        
        # Display samples - interactive mode when num_samples is 1
        if num_samples == 1:
            if field_path and matching_indices:
                # Interactive mode with filtering
                filtered_idx = start_idx
                while filtered_idx < len(matching_indices):
                    actual_idx = matching_indices[filtered_idx]
                    sample = dataset[actual_idx]
                    
                    if raw_json:
                        print(f"\n{'='*80}")
                        print(f"RAW JSON SAMPLE {actual_idx + 1} of {total_samples} (Filtered: {filtered_idx + 1} of {total_filtered})")
                        print('='*80)
                        print(json.dumps(sample, indent=2, ensure_ascii=False, default=str))
                    else:
                        pretty_print_sample(sample, actual_idx, total_samples, filtered_idx, total_filtered)
                    
                    print(f"\nPress Enter for next matching sample, 'q' to quit, or number to jump to specific filtered sample:")
                    user_input = input().strip()
                    
                    if user_input.lower() == 'q':
                        break
                    elif user_input.isdigit():
                        new_idx = int(user_input) - 1  # Convert to 0-based index
                        if 0 <= new_idx < len(matching_indices):
                            filtered_idx = new_idx
                        else:
                            print(f"Invalid filtered sample number. Must be between 1 and {len(matching_indices)}")
                            continue
                    else:
                        filtered_idx += 1
                        
                    if filtered_idx >= len(matching_indices):
                        print(f"\nReached end of filtered dataset ({len(matching_indices)} matching samples)")
                        break
            else:
                # Original interactive mode without filtering
                current_idx = start_idx
                while current_idx < total_samples:
                    sample = dataset[current_idx]
                    
                    if raw_json:
                        print(f"\n{'='*80}")
                        print(f"RAW JSON SAMPLE {current_idx + 1} of {total_samples}")
                        print('='*80)
                        print(json.dumps(sample, indent=2, ensure_ascii=False, default=str))
                    else:
                        pretty_print_sample(sample, current_idx, total_samples)
                    
                    print(f"\nPress Enter for next sample, 'q' to quit, or number to jump to specific sample:")
                    user_input = input().strip()
                    
                    if user_input.lower() == 'q':
                        break
                    elif user_input.isdigit():
                        new_idx = int(user_input) - 1  # Convert to 0-based index
                        if 0 <= new_idx < total_samples:
                            current_idx = new_idx
                        else:
                            print(f"Invalid sample number. Must be between 1 and {total_samples}")
                            continue
                    else:
                        current_idx += 1
                        
                    if current_idx >= total_samples:
                        print(f"\nReached end of dataset ({total_samples} samples)")
                        break
        else:
            # Batch mode
            if field_path and matching_indices:
                # Display filtered samples
                for filtered_idx in range(start_idx, end_idx):
                    actual_idx = matching_indices[filtered_idx]
                    sample = dataset[actual_idx]
                    
                    if raw_json:
                        print(f"\n{'='*80}")
                        print(f"RAW JSON SAMPLE {actual_idx + 1} (Filtered: {filtered_idx + 1})")
                        print('='*80)
                        print(json.dumps(sample, indent=2, ensure_ascii=False, default=str))
                    else:
                        pretty_print_sample(sample, actual_idx, total_samples, filtered_idx, total_filtered)
            else:
                # Display all samples in range
                for i in range(start_idx, end_idx):
                    sample = dataset[i]
                    
                    if raw_json:
                        print(f"\n{'='*80}")
                        print(f"RAW JSON SAMPLE {i + 1}")
                        print('='*80)
                        print(json.dumps(sample, indent=2, ensure_ascii=False, default=str))
                    else:
                        pretty_print_sample(sample, i, total_samples)
        
        return True
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return False


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Browse and inspect samples from chat format datasets with parts structure and optional field filtering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic browsing
  python browse_sample.py ../data/converted/dataset
  python browse_sample.py ../data/converted/dataset --num-samples 3
  python browse_sample.py ../data/converted/dataset --start-idx 100 --raw-json
  
  # Field filtering - browse only refusal samples
  python browse_sample.py ../data/converted/dataset \\
    --field "conversation_branches[0].messages[0].metadata.refusal_classification.meta-llama/Llama-3.3-70B-Instruct.classification" \\
    --include "refusal" "soft_refusal"
  
  # Browse only low-quality samples
  python browse_sample.py ../data/converted/dataset \\
    --field "initial_prompt.metadata.quality_classification.meta-llama/Llama-3.3-70B-Instruct.well_formedness.score" \\
    --include "1"
  
  # Exclude specific categories
  python browse_sample.py ../data/converted/dataset \\
    --field "original_metadata.category" \\
    --exclude "math" "science"
        """
    )
    
    parser.add_argument(
        "dataset_path",
        help="Path to dataset directory"
    )
    parser.add_argument(
        "--num-samples", "-n",
        type=int,
        default=1,
        help="Number of samples to display (default: 1)"
    )
    parser.add_argument(
        "--start-idx", "-s", 
        type=int,
        default=0,
        help="Starting sample index (default: 0)"
    )
    parser.add_argument(
        "--raw-json", "-r",
        action="store_true",
        help="Display raw JSON instead of pretty-printed format"
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Dataset split to browse (default: first available split)"
    )
    
    # Field filtering arguments
    parser.add_argument(
        "--field",
        type=str,
        help="Field path for filtering (e.g., 'original_metadata.category' or 'messages[0].role')"
    )
    parser.add_argument(
        "--include",
        nargs="+",
        help="Values to include (samples must have one of these values)"
    )
    parser.add_argument(
        "--exclude",
        nargs="+",
        help="Values to exclude (samples must not have any of these values)"
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_arguments()
    
    # Validate filter arguments
    if (args.include or args.exclude) and not args.field:
        print("Error: --field must be specified when using --include or --exclude")
        sys.exit(1)
    
    success = browse_dataset(
        args.dataset_path,
        args.num_samples, 
        args.start_idx,
        args.raw_json,
        args.split,
        args.field,
        args.include,
        args.exclude
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()