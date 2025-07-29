#!/usr/bin/env python3
"""
Display dataset information including processing log and metadata.

This script reads a dataset directory and displays key information including
the processing log, dataset size, and other metadata.
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from datasets import load_from_disk


def format_timestamp(timestamp_str: str) -> str:
    """Format ISO timestamp for display."""
    try:
        dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    except:
        return timestamp_str


def display_dataset_info(dataset_path: str):
    """Display comprehensive dataset information."""
    path = Path(dataset_path)
    
    if not path.exists():
        print(f"Error: Dataset path does not exist: {dataset_path}")
        return False
    
    print(f"{'='*80}")
    print(f"DATASET INFORMATION")
    print(f"{'='*80}")
    print(f"Path: {path}")
    
    # Load dataset to get size info
    try:
        dataset = load_from_disk(str(path))
        
        # Handle DatasetDict vs single Dataset
        if hasattr(dataset, 'keys'):
            available_splits = list(dataset.keys())
            print(f"Type: DatasetDict with {len(available_splits)} splits")
            total_samples = 0
            for split_name in available_splits:
                split_size = len(dataset[split_name])
                total_samples += split_size
                print(f"  - {split_name}: {split_size:,} samples")
            print(f"Total samples: {total_samples:,}")
        else:
            print(f"Type: Single Dataset")
            print(f"Samples: {len(dataset):,}")
            
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return False
    
    # Load and display metadata
    metadata_file = path / "dataset_metadata.json"
    if not metadata_file.exists():
        print(f"\nNo metadata file found at {metadata_file}")
        return True
    
    try:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
    except Exception as e:
        print(f"Error reading metadata: {e}")
        return False
    
    # Display processing log
    processing_log = metadata.get("processing_log", [])
    if processing_log:
        print(f"\n{'='*80}")
        print(f"PROCESSING LOG ({len(processing_log)} entries)")
        print(f"{'='*80}")
        
        for i, entry in enumerate(processing_log, 1):
            print(f"\n{i}. {entry.get('operation', 'unknown').upper()}")
            print(f"   Script: {entry.get('script', 'N/A')}")
            print(f"   Timestamp: {format_timestamp(entry.get('timestamp', 'N/A'))}")
            print(f"   Input: {entry.get('input_path', 'N/A')}")
            print(f"   Output: {entry.get('output_path', 'N/A')}")
            
            # Operation-specific details
            if entry.get('operation') == 'standardisation':
                print(f"   Samples processed: {entry.get('samples_processed', 'N/A'):,}")
                print(f"   Target schema: {entry.get('target_schema', 'N/A')}")
                if entry.get('split_name'):
                    print(f"   Split: {entry.get('split_name')}")
                    
            elif entry.get('operation') == 'field_based_filtering':
                print(f"   Filter field: {entry.get('filter_field', 'N/A')}")
                print(f"   Filter action: {entry.get('filter_action', 'N/A')}")
                print(f"   Filter values: {entry.get('filter_values', 'N/A')}")
                print(f"   Original samples: {entry.get('original_samples', 'N/A'):,}")
                print(f"   Filtered samples: {entry.get('filtered_samples', 'N/A'):,}")
                
            # Show any other fields
            other_fields = {k: v for k, v in entry.items() 
                          if k not in ['operation', 'script', 'timestamp', 'input_path', 'output_path',
                                     'samples_processed', 'target_schema', 'split_name', 
                                     'filter_field', 'filter_action', 'filter_values', 
                                     'original_samples', 'filtered_samples']}
            if other_fields:
                for key, value in other_fields.items():
                    print(f"   {key}: {value}")
    else:
        print(f"\nNo processing log found in metadata")
    
    # Display other metadata (exclude processing_log)
    other_metadata = {k: v for k, v in metadata.items() if k != 'processing_log'}
    if other_metadata:
        print(f"\n{'='*80}")
        print(f"OTHER METADATA")
        print(f"{'='*80}")
        for key, value in other_metadata.items():
            if isinstance(value, (dict, list)):
                print(f"{key}: {type(value).__name__} with {len(value)} items")
            else:
                # Truncate long values
                value_str = str(value)
                if len(value_str) > 100:
                    value_str = value_str[:97] + "..."
                print(f"{key}: {value_str}")
    
    return True


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Display dataset information including processing log and metadata",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python inspect_dataset.py ../data/02-standardised/tulu-3-sft-mixture
  python inspect_dataset.py ../data/03-filtered/smoltalk-numina
  python inspect_dataset.py ../data/05-annotations/dataset-refusal
        """
    )
    
    parser.add_argument(
        "dataset_path",
        help="Path to dataset directory"
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_arguments()
    
    success = display_dataset_info(args.dataset_path)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()