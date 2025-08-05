#!/usr/bin/env python3
"""
Browse and inspect samples from standardized chat format datasets with parts structure.

This script loads a dataset and displays sample conversations in the format
where assistant and user messages contain 'parts' lists with typed components.
"""

import sys
import json
import argparse
from pathlib import Path
from datasets import load_from_disk
from datetime import datetime


def pretty_print_sample(sample: dict, sample_idx: int = 0, total_samples: int = 0):
    """Pretty print a single chat format sample."""
    print(f"\n{'═'*80}")
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
                        print(f"{branch_line} {msg_line} │  {part_line} Metadata: {len(part['metadata'])} fields")
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


def browse_dataset(dataset_path: str, num_samples: int = 1, start_idx: int = 0, raw_json: bool = False, split: str = None):
    """Browse samples from a dataset."""
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
        
        # Validate indices
        if start_idx >= total_samples:
            print(f"Error: Start index {start_idx} is beyond dataset size {total_samples}")
            return False
        
        end_idx = min(start_idx + num_samples, total_samples)
        actual_samples = end_idx - start_idx
        
        if num_samples == 1:
            print(f"Interactive browsing mode. Press Enter for next, 'q' to quit, or number to jump")
        else:
            print(f"Showing samples {start_idx + 1} to {end_idx} ({actual_samples} samples)")
        
        # Display samples - interactive mode when num_samples is 1
        if num_samples == 1:
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
        description="Browse and inspect samples from chat format datasets with parts structure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python browse_sample_new.py ../data/converted/dataset
  python browse_sample_new.py ../data/converted/dataset --num-samples 3
  python browse_sample_new.py ../data/converted/dataset --start-idx 100 --raw-json
  python browse_sample_new.py ../data/converted/dataset --split test
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
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_arguments()
    
    success = browse_dataset(
        args.dataset_path,
        args.num_samples, 
        args.start_idx,
        args.raw_json,
        args.split
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()