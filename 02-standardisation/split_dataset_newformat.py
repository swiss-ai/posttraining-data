#!/usr/bin/env python3
"""
Universal dataset splitting tool for standardized chat format datasets.

Supports three modes:
1. splits: Extract DatasetDict splits into separate datasets
2. even: Divide dataset into N equal parts  
3. range: Extract samples from index A to B

Works with both old and new chat format datasets.
"""

import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
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


def save_dataset_with_metadata(dataset: DatasetDict, output_path: Path, 
                              processing_entry: Dict[str, Any], 
                              original_metadata: Dict[str, Any]):
    """Save dataset with updated metadata."""
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save dataset
    dataset.save_to_disk(str(output_path))
    
    # Update metadata
    metadata = {
        **original_metadata,
        "processing_log": original_metadata.get("processing_log", []) + [processing_entry]
    }
    
    # Save metadata
    with open(output_path / "dataset_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)


def get_split_suffix(index: int) -> str:
    """Convert index to letter suffix (0->A, 1->B, etc.)"""
    return chr(ord('A') + index)


def is_standard_split_name(split_name: str) -> bool:
    """Check if split name is a standard ML split name."""
    standard_names = {"train", "test", "validation", "valid", "eval", "dev"}
    return split_name.lower() in standard_names


def split_by_splits(dataset: DatasetDict, dataset_name: str, 
                   output_dir: Path, original_metadata: Dict[str, Any],
                   input_path: Path) -> bool:
    """
    Split DatasetDict by existing splits into separate datasets.
    
    Args:
        dataset: DatasetDict to split
        dataset_name: Name of the dataset
        output_dir: Directory to save split datasets
        original_metadata: Original dataset metadata
        input_path: Path to input dataset
        
    Returns:
        True if successful
    """
    splits = list(dataset.keys())
    print(f"Found DatasetDict with {len(splits)} splits: {splits}")
    
    # Show what will be created
    print(f"\nWill create the following datasets:")
    for split in splits:
        output_path = output_dir / f"{dataset_name}-{split}"
        sample_count = len(dataset[split])
        print(f"  {output_path} ({sample_count:,} samples)")
    
    # Check for standard split names and warn
    standard_splits = [s for s in splits if is_standard_split_name(s)]
    if standard_splits:
        print(f"\n⚠️  WARNING: Found standard split names that may cause confusion: {standard_splits}")
        print(f"These will be renamed to 'train' in the output datasets.")
        print(f"Waiting 3 seconds before continuing...")
        time.sleep(3)
    
    # Process each split
    print(f"\nSplitting dataset into {len(splits)} separate datasets...")
    success_count = 0
    
    for split_name in tqdm(splits, desc="Processing splits"):
        try:
            split_data = dataset[split_name]
            output_path = output_dir / f"{dataset_name}-{split_name}"
            
            # Create DatasetDict with "train" split
            split_dataset = DatasetDict({"train": split_data})
            
            # Create processing log entry
            processing_entry = {
                "operation": "dataset_split_by_splits",
                "script": "02-standardisation/split_dataset.py",
                "timestamp": datetime.now().isoformat(),
                "input_path": str(input_path),
                "output_path": str(output_path),
                "mode": "splits",
                "source_split": split_name,
                "target_split": "train",
                "samples": len(split_data)
            }
            
            # Save dataset with metadata
            save_dataset_with_metadata(split_dataset, output_path, 
                                      processing_entry, original_metadata)
            
            print(f"  ✓ {split_name} → {output_path} ({len(split_data):,} samples)")
            success_count += 1
            
        except Exception as e:
            print(f"  ✗ Error processing split '{split_name}': {e}")
    
    return success_count == len(splits)


def split_even_parts(dataset: Dataset, dataset_name: str, output_dir: Path,
                    n_parts: int, original_metadata: Dict[str, Any],
                    input_path: Path, split_name: str = "train") -> bool:
    """
    Split dataset into N equal parts.
    
    Args:
        dataset: Dataset to split (single Dataset, not DatasetDict)
        dataset_name: Name of the dataset
        output_dir: Directory to save split datasets
        n_parts: Number of parts to split into
        original_metadata: Original dataset metadata
        input_path: Path to input dataset
        split_name: Name of the split being processed
        
    Returns:
        True if successful
    """
    dataset_size = len(dataset)
    print(f"\nSplitting {split_name} split ({dataset_size:,} samples) into {n_parts} equal parts")
    
    # Calculate part sizes
    base_part_size = dataset_size // n_parts
    remainder = dataset_size % n_parts
    
    # Show what will be created
    print(f"\nWill create the following datasets:")
    for i in range(n_parts):
        suffix = get_split_suffix(i)
        output_name = f"{dataset_name}-split{suffix}"
        part_size = base_part_size + (1 if i < remainder else 0)
        print(f"  {output_dir / output_name} (~{part_size:,} samples)")
    
    # Create parts
    start_idx = 0
    created_datasets = []
    
    for i in tqdm(range(n_parts), desc="Creating parts"):
        try:
            # Calculate part size (distribute remainder across first parts)
            part_size = base_part_size + (1 if i < remainder else 0)
            end_idx = start_idx + part_size
            
            # Create output name and path
            suffix = get_split_suffix(i)
            output_name = f"{dataset_name}-split{suffix}"
            output_path = output_dir / output_name
            
            # Select samples for this part
            part_data = dataset.select(range(start_idx, end_idx))
            
            # Create DatasetDict with "train" split
            part_dataset = DatasetDict({"train": part_data})
            
            # Create processing log entry
            processing_entry = {
                "operation": "dataset_even_split",
                "script": "02-standardisation/split_dataset.py",
                "timestamp": datetime.now().isoformat(),
                "input_path": str(input_path),
                "output_path": str(output_path),
                "mode": "even",
                "source_split": split_name,
                "part_index": i + 1,
                "total_parts": n_parts,
                "samples": len(part_data),
                "start_index": start_idx,
                "end_index": end_idx - 1
            }
            
            # Save dataset with metadata
            save_dataset_with_metadata(part_dataset, output_path,
                                      processing_entry, original_metadata)
            
            created_datasets.append((str(output_path), len(part_data)))
            start_idx = end_idx
            
        except Exception as e:
            print(f"  ✗ Error creating part {i + 1}: {e}")
            return False
    
    # Summary
    print(f"\nCreated {len(created_datasets)} dataset parts:")
    for path, count in created_datasets:
        print(f"  {path}: {count:,} samples")
    
    return True


def split_by_range(dataset: Dataset, dataset_name: str, start_idx: int, 
                  end_idx: int, output_path: Optional[Path],
                  original_metadata: Dict[str, Any], input_path: Path,
                  split_name: str = "train") -> bool:
    """
    Extract samples from start to end index.
    
    Args:
        dataset: Dataset to extract from (single Dataset, not DatasetDict)
        dataset_name: Name of the dataset
        start_idx: Start index (inclusive)
        end_idx: End index (exclusive)
        output_path: Output path (if None, auto-generate)
        original_metadata: Original dataset metadata
        input_path: Path to input dataset
        split_name: Name of the split being processed
        
    Returns:
        True if successful
    """
    dataset_size = len(dataset)
    
    # Validate indices
    if start_idx < 0:
        print(f"Error: Start index {start_idx} cannot be negative")
        return False
    if end_idx > dataset_size:
        print(f"Error: End index {end_idx} exceeds dataset size {dataset_size}")
        return False
    if start_idx >= end_idx:
        print(f"Error: Start index {start_idx} must be less than end index {end_idx}")
        return False
    
    # Auto-generate output path if not specified
    if output_path is None:
        output_name = f"{dataset_name}-{start_idx}-{end_idx}"
        output_path = input_path.parent / output_name
        print(f"No output path specified, using: {output_path}")
    
    print(f"\nExtracting samples {start_idx:,} to {end_idx-1:,} from {split_name} split")
    print(f"Total samples to extract: {end_idx - start_idx:,}")
    
    try:
        # Select samples in range
        range_data = dataset.select(range(start_idx, end_idx))
        
        # Create DatasetDict with "train" split
        range_dataset = DatasetDict({"train": range_data})
        
        # Create processing log entry
        processing_entry = {
            "operation": "dataset_range_extraction",
            "script": "02-standardisation/split_dataset.py",
            "timestamp": datetime.now().isoformat(),
            "input_path": str(input_path),
            "output_path": str(output_path),
            "mode": "range",
            "source_split": split_name,
            "start_index": start_idx,
            "end_index": end_idx - 1,
            "samples": len(range_data)
        }
        
        # Save dataset with metadata
        save_dataset_with_metadata(range_dataset, output_path,
                                  processing_entry, original_metadata)
        
        print(f"✓ Extracted {len(range_data):,} samples to {output_path}")
        return True
        
    except Exception as e:
        print(f"✗ Error extracting range: {e}")
        return False


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Universal dataset splitting tool for standardized chat format datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Mode 1: Split by existing DatasetDict splits
  python split_dataset_newformat.py data/dataset output_dir --mode splits
  
  # Mode 2: Even N-way split (creates splitA, splitB, splitC, etc.)
  python split_dataset_newformat.py data/dataset output_dir --mode even --parts 5
  
  # Mode 3: Range extraction with custom output
  python split_dataset_newformat.py data/dataset --mode range --start 1000 --end 2000 --output data/custom_path
  
  # Mode 3: Range extraction with auto-generated output name
  python split_dataset_newformat.py data/dataset --mode range --start 1000 --end 2000
  
  # Process specific split only
  python split_dataset_newformat.py data/dataset output_dir --mode even --parts 3 --split validation

All outputs are DatasetDict with "train" split for pipeline compatibility.
        """
    )
    
    parser.add_argument(
        "input_dataset",
        help="Path to input dataset directory"
    )
    parser.add_argument(
        "output_directory",
        nargs='?',  # Optional for range mode
        help="Directory where split datasets will be created (required for splits/even modes)"
    )
    parser.add_argument(
        "--mode",
        choices=["splits", "even", "range"],
        required=True,
        help="Split mode: splits (by DatasetDict), even (N equal parts), range (index range)"
    )
    parser.add_argument(
        "--parts",
        type=int,
        help="Number of parts for even split mode (2-26)"
    )
    parser.add_argument(
        "--start",
        type=int,
        help="Start index for range mode (inclusive)"
    )
    parser.add_argument(
        "--end",
        type=int,
        help="End index for range mode (exclusive)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Custom output path for range mode (optional)"
    )
    parser.add_argument(
        "--split",
        help="Specific split to process (default: process all or use 'train')"
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_arguments()
    
    # Validate arguments based on mode
    if args.mode == "splits":
        if not args.output_directory:
            print("Error: output_directory is required for splits mode")
            sys.exit(1)
    elif args.mode == "even":
        if not args.output_directory:
            print("Error: output_directory is required for even mode")
            sys.exit(1)
        if not args.parts:
            print("Error: --parts is required for even mode")
            sys.exit(1)
        if args.parts < 2 or args.parts > 26:
            print("Error: --parts must be between 2 and 26")
            sys.exit(1)
    elif args.mode == "range":
        if args.start is None or args.end is None:
            print("Error: --start and --end are required for range mode")
            sys.exit(1)
        if args.output_directory and args.output:
            print("Error: Cannot specify both output_directory and --output for range mode")
            sys.exit(1)
    
    # Set up paths
    input_path = Path(args.input_dataset)
    if not input_path.exists():
        print(f"Error: Input dataset does not exist: {input_path}")
        sys.exit(1)
    
    dataset_name = input_path.name
    
    # Load dataset
    print(f"{'='*60}")
    print(f"Dataset Splitting Tool")
    print(f"Mode: {args.mode}")
    print(f"Input: {input_path}")
    if args.output_directory:
        print(f"Output directory: {args.output_directory}")
    print(f"{'='*60}")
    
    print(f"\nLoading dataset from: {input_path}")
    try:
        dataset = load_from_disk(str(input_path))
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)
    
    # Load original metadata
    original_metadata = load_existing_metadata(input_path)
    
    # Handle DatasetDict vs single Dataset
    is_dataset_dict = hasattr(dataset, 'keys')
    
    # Execute based on mode
    success = False
    
    if args.mode == "splits":
        # Mode 1: Split by DatasetDict splits
        if not is_dataset_dict:
            print("Error: splits mode requires a DatasetDict input")
            sys.exit(1)
        
        output_dir = Path(args.output_directory)
        success = split_by_splits(dataset, dataset_name, output_dir,
                                original_metadata, input_path)
    
    elif args.mode == "even":
        # Mode 2: Even N-way split
        output_dir = Path(args.output_directory)
        
        if is_dataset_dict:
            # Handle DatasetDict
            available_splits = list(dataset.keys())
            print(f"Found DatasetDict with splits: {available_splits}")
            
            if args.split:
                if args.split not in available_splits:
                    print(f"Error: Split '{args.split}' not found. Available: {available_splits}")
                    sys.exit(1)
                target_split = args.split
            elif 'train' in available_splits:
                target_split = 'train'
                print(f"Using 'train' split")
            else:
                target_split = available_splits[0]
                print(f"Using '{target_split}' split")
            
            dataset_to_split = dataset[target_split]
        else:
            # Single Dataset
            target_split = "train"
            dataset_to_split = dataset
        
        success = split_even_parts(dataset_to_split, dataset_name, output_dir,
                                 args.parts, original_metadata, input_path,
                                 target_split)
    
    elif args.mode == "range":
        # Mode 3: Range extraction
        if args.output:
            output_path = Path(args.output)
        elif args.output_directory:
            output_path = Path(args.output_directory) / f"{dataset_name}-{args.start}-{args.end}"
        else:
            output_path = None  # Will be auto-generated
        
        if is_dataset_dict:
            # Handle DatasetDict
            available_splits = list(dataset.keys())
            print(f"Found DatasetDict with splits: {available_splits}")
            
            if args.split:
                if args.split not in available_splits:
                    print(f"Error: Split '{args.split}' not found. Available: {available_splits}")
                    sys.exit(1)
                target_split = args.split
            elif 'train' in available_splits:
                target_split = 'train'
                print(f"Using 'train' split")
            else:
                target_split = available_splits[0]
                print(f"Using '{target_split}' split")
            
            dataset_to_extract = dataset[target_split]
        else:
            # Single Dataset
            target_split = "train"
            dataset_to_extract = dataset
        
        success = split_by_range(dataset_to_extract, dataset_name,
                               args.start, args.end, output_path,
                               original_metadata, input_path, target_split)
    
    # Final summary
    print(f"\n{'='*60}")
    if success:
        print(f"✓ Dataset splitting completed successfully")
    else:
        print(f"✗ Dataset splitting failed")
    print(f"{'='*60}")
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()