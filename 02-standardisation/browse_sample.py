#!/usr/bin/env python3
"""
Browse and inspect samples from standardized chat format datasets.

This script loads a dataset and displays sample conversations in a readable format,
showing the chat format structure with conversation branches and metadata.
"""

import sys
import json
import argparse
from pathlib import Path
from datasets import load_from_disk
from datetime import datetime


def pretty_print_sample(sample: dict, sample_idx: int = 0, total_samples: int = 0):
    """Pretty print a single chat format sample."""
    print(f"\n{'='*80}")
    print(f"SAMPLE {sample_idx + 1} of {total_samples}")
    print('='*80)
    
    # Basic info
    print(f"Conversation ID: {sample.get('conversation_id', 'N/A')}")
    print(f"Dataset Source: {sample.get('dataset_source', 'N/A')}")
    print(f"Created: {sample.get('created_timestamp', 'N/A')}")
    
    # System prompt
    if 'system_prompt' in sample and sample['system_prompt']:
        print(f"\nSystem Prompt:")
        print(f"   {sample['system_prompt']['content']}")
        if sample['system_prompt'].get('metadata'):
            print(f"   Metadata: {len(sample['system_prompt']['metadata'])} fields")
            def print_nested_dict(d, indent="     "):
                for key, value in d.items():
                    if isinstance(value, dict):
                        print(f"{indent}{key}:")
                        print_nested_dict(value, indent + "  ")
                    else:
                        print(f"{indent}{key}: {value}")
            print_nested_dict(sample['system_prompt']['metadata'])
    
    # Initial prompt
    if 'initial_prompt' in sample:
        print(f"\nInitial Prompt ({sample['initial_prompt']['role']}):")
        print(f"   {sample['initial_prompt']['content']}")
        if sample['initial_prompt'].get('metadata'):
            print(f"   Metadata: {len(sample['initial_prompt']['metadata'])} fields")
            def print_nested_dict(d, indent="     "):
                for key, value in d.items():
                    if isinstance(value, dict):
                        print(f"{indent}{key}:")
                        print_nested_dict(value, indent + "  ")
                    else:
                        print(f"{indent}{key}: {value}")
            print_nested_dict(sample['initial_prompt']['metadata'])
    
    # Conversation branches  
    branches = sample.get('conversation_branches', [])
    print(f"\nConversation Branches: {len(branches)}")
    
    for branch_idx, branch in enumerate(branches):
        print(f"\n  Branch {branch_idx + 1}:")
        messages = branch.get('messages', [])
        
        for msg_idx, msg in enumerate(messages):
            print(f"    {msg['role'].title()}: {msg['content']}")
            
            if msg.get('metadata'):
                print(f"       Metadata: {len(msg['metadata'])} fields")
                def print_nested_dict(d, indent="         "):
                    for key, value in d.items():
                        if isinstance(value, dict):
                            print(f"{indent}{key}:")
                            print_nested_dict(value, indent + "  ")
                        else:
                            print(f"{indent}{key}: {value}")
                print_nested_dict(msg['metadata'])
    
    # Original metadata
    if 'original_metadata' in sample and sample['original_metadata']:
        print(f"\nOriginal Metadata: {len(sample['original_metadata'])} fields")
        for key, value in sample['original_metadata'].items():
            if isinstance(value, str) and len(value) > 50:
                print(f"   {key}: {value[:50]}...")
            else:
                print(f"   {key}: {value}")


def browse_dataset(dataset_path: str, num_samples: int = 1, start_idx: int = 0, raw_json: bool = False):
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
            print(f"Found DatasetDict with splits: {available_splits}")
            
            if 'train' in available_splits:
                dataset = dataset['train']
                print(f"Using 'train' split")
            else:
                first_split = available_splits[0]
                dataset = dataset[first_split]
                print(f"Using '{first_split}' split")
        
        total_samples = len(dataset)
        print(f"Dataset size: {total_samples:,} samples")
        
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
        description="Browse and inspect samples from standardized chat format datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python browse_sample.py ../data/02-standardised/tulu-3-sft-mixture
  python browse_sample.py ../data/02-standardised/smoltalk --num-samples 3
  python browse_sample.py ../data/02-standardised/The-Tome --start-idx 100 --raw-json
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
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_arguments()
    
    success = browse_dataset(
        args.dataset_path,
        args.num_samples, 
        args.start_idx,
        args.raw_json
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()