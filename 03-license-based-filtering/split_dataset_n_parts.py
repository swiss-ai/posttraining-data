#!/usr/bin/env python3
"""
Split a DatasetDict split into N equal parts, creating separate datasets.

This script takes a DatasetDict and splits one or all of its splits into N equal parts,
creating N separate dataset directories with names like dataset-splitA, dataset-splitB, etc.
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
from datasets import load_from_disk, Dataset, DatasetDict
from tqdm import tqdm


def load_existing_metadata(input_path: Path) -> Dict[str, Any]:
    """Load existing dataset metadata if it exists."""
    metadata_file = input_path / "dataset_metadata.json"
    if metadata_file.exists():
        try:
            with open(metadata_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def get_split_suffix(index: int) -> str:
    """Convert index to letter suffix (0->A, 1->B, etc.)"""
    return chr(ord('A') + index)


def split_dataset_n_parts(input_path: Path, output_base_dir: Path, n_parts: int, split_name: str = None):
    """Split dataset into N equal parts, creating separate datasets."""
    
    # Validate input path
    if not input_path.exists():
        print(f"Error: Input dataset path does not exist: {input_path}")
        return False
    
    # Load dataset
    print(f"Loading dataset from: {input_path}")
    try:
        dataset = load_from_disk(str(input_path))
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return False
    
    # Get dataset name from input path
    dataset_name = input_path.name
    
    # Handle DatasetDict vs single Dataset
    if hasattr(dataset, 'keys'):
        # It's a DatasetDict
        available_splits = list(dataset.keys())
        print(f"Found DatasetDict with splits: {available_splits}")
        
        if split_name:
            if split_name not in available_splits:
                print(f"Error: Split '{split_name}' not found. Available: {available_splits}")
                return False
            splits_to_process = [split_name]
        else:
            splits_to_process = available_splits
    else:
        # Single Dataset - wrap in DatasetDict
        dataset = DatasetDict({"train": dataset})
        splits_to_process = ["train"]
    
    # Process splits
    total_samples = 0
    created_datasets = []
    
    for split in splits_to_process:
        split_data = dataset[split]
        split_size = len(split_data)
        total_samples += split_size
        
        print(f"\nProcessing split '{split}' with {split_size:,} samples")
        
        # Calculate part size
        base_part_size = split_size // n_parts
        remainder = split_size % n_parts
        
        # Create parts
        start_idx = 0
        for i in range(n_parts):
            # Add 1 extra sample to first 'remainder' parts to distribute evenly
            part_size = base_part_size + (1 if i < remainder else 0)
            end_idx = start_idx + part_size
            
            # Create output dataset name and path
            suffix = get_split_suffix(i)
            output_name = f"{dataset_name}-split{suffix}"
            output_path = output_base_dir / output_name
            
            # Select samples for this part
            part_data = split_data.select(range(start_idx, end_idx))
            
            # Create DatasetDict with train split
            part_dataset = DatasetDict({"train": part_data})
            
            # Save dataset
            print(f"  Saving {output_name}: {len(part_data):,} samples (indices {start_idx:,} - {end_idx-1:,})")
            output_path.mkdir(parents=True, exist_ok=True)
            part_dataset.save_to_disk(str(output_path))
            
            # Create processing log entry
            processing_entry = {
                "operation": "dataset_n_way_split",
                "script": "split_dataset_n_parts.py",
                "timestamp": datetime.now().isoformat(),
                "input_path": str(input_path),
                "output_path": str(output_path),
                "source_split": split,
                "part_index": i + 1,
                "total_parts": n_parts,
                "samples": len(part_data),
                "start_index": start_idx,
                "end_index": end_idx - 1
            }
            
            # Load original metadata and update
            original_metadata = load_existing_metadata(input_path)
            metadata = {
                **original_metadata,
                "processing_log": original_metadata.get("processing_log", []) + [processing_entry]
            }
            
            # Save metadata
            with open(output_path / "dataset_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            created_datasets.append((str(output_path), len(part_data)))
            start_idx = end_idx
    
    # Summary
    print(f"\n{'='*60}")
    print(f"DATASET SPLITTING COMPLETED")
    print(f"{'='*60}")
    print(f"Input: {input_path}")
    print(f"Output directory: {output_base_dir}")
    print(f"Total samples: {total_samples:,}")
    print(f"Datasets created: {len(created_datasets)}")
    print(f"\nCreated datasets:")
    for path, count in created_datasets:
        print(f"  {path}: {count:,} samples")
    
    return True


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Split a DatasetDict into N equal parts, creating separate datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Split all splits into 5 parts each
  python split_dataset_n_parts.py input_dataset output_directory 5
  
  # Split only the 'math' split into 5 parts
  python split_dataset_n_parts.py input_dataset output_directory 5 --split math
  
This will create separate datasets in the output directory:
  - dataset-splitA: First part with train split
  - dataset-splitB: Second part with train split
  - dataset-splitC: Third part with train split
  - etc.
        """
    )
    
    parser.add_argument(
        "input_dataset",
        help="Path to input dataset directory"
    )
    parser.add_argument(
        "output_directory",
        help="Directory where split datasets will be created"
    )
    parser.add_argument(
        "n_parts",
        type=int,
        help="Number of parts to split into"
    )
    parser.add_argument(
        "--split",
        help="Specific split to process (default: process all splits)"
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_arguments()
    
    if args.n_parts < 2:
        print("Error: n_parts must be at least 2")
        sys.exit(1)
    
    if args.n_parts > 26:
        print("Error: n_parts must be 26 or less (A-Z)")
        sys.exit(1)
    
    input_path = Path(args.input_dataset)
    output_dir = Path(args.output_directory)
    
    print(f"Dataset N-Way Splitting Tool")
    print(f"Input: {input_path}")
    print(f"Output directory: {output_dir}")
    print(f"Parts: {args.n_parts}")
    if args.split:
        print(f"Split: {args.split}")
    print(f"{'='*60}")
    
    success = split_dataset_n_parts(input_path, output_dir, args.n_parts, args.split)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()