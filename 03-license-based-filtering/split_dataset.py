#!/usr/bin/env python3
"""
Split DatasetDict into separate datasets for individual processing.

Takes a DatasetDict with multiple splits and creates separate dataset directories
for each split, allowing them to be processed independently by decontamination
and other pipeline stages.
"""

import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
from datasets import load_from_disk, Dataset, DatasetDict
from tqdm import tqdm


def is_standard_split_name(split_name: str) -> bool:
    """Check if split name is a standard ML split name."""
    standard_names = {"train", "test", "validation", "valid", "eval", "dev"}
    return split_name.lower() in standard_names


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


def save_split_dataset(split_name: str, split_data: Dataset, dataset_name: str,
                      output_dir: Path, input_path: Path, original_metadata: Dict[str, Any]):
    """Save a single split as a separate DatasetDict."""
    
    # Create output path
    output_path = output_dir / f"{dataset_name}-{split_name}"
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"  Saving {split_name} split to {output_path}...")
    
    # Create DatasetDict with "train" split
    split_dataset = DatasetDict({"train": split_data})
    
    # Save dataset
    split_dataset.save_to_disk(str(output_path))
    
    # Create processing log entry
    processing_entry = {
        "operation": "dataset_splitting",
        "script": "split_dataset.py",
        "timestamp": datetime.now().isoformat(),
        "input_path": str(input_path),
        "output_path": str(output_path),
        "source_split": split_name,
        "target_split": "train",
        "samples": len(split_data)
    }
    
    # Update metadata
    metadata = {
        **original_metadata,
        "processing_log": original_metadata.get("processing_log", []) + [processing_entry]
    }
    
    # Save metadata
    with open(output_path / "dataset_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  ✓ {split_name} → {output_path} ({len(split_data):,} samples)")


def split_dataset(input_path: Path, output_dir: Path):
    """Split DatasetDict into separate datasets."""
    
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
    
    # Validate that it's a DatasetDict
    if not hasattr(dataset, 'keys'):
        print(f"Error: Input dataset is not a DatasetDict (found single Dataset)")
        print(f"This script is designed to split DatasetDict with multiple splits")
        return False
    
    # Get splits and dataset name
    splits = list(dataset.keys())
    dataset_name = input_path.name
    
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
        print(f"Waiting 4 seconds before continuing...")
        time.sleep(4)
    
    # Load original metadata
    original_metadata = load_existing_metadata(input_path)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each split
    print(f"\nSplitting dataset into {len(splits)} separate datasets...")
    
    success_count = 0
    for split_name in tqdm(splits, desc="Processing splits"):
        try:
            split_data = dataset[split_name]
            save_split_dataset(split_name, split_data, dataset_name, 
                             output_dir, input_path, original_metadata)
            success_count += 1
        except Exception as e:
            print(f"  ✗ Error processing split '{split_name}': {e}")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"DATASET SPLITTING COMPLETED")
    print(f"{'='*60}")
    print(f"Input: {input_path}")
    print(f"Output directory: {output_dir}")
    print(f"Original dataset: {dataset_name}")
    print(f"Splits processed: {success_count}/{len(splits)}")
    
    if success_count == len(splits):
        print(f"✓ All splits successfully extracted")
        
        # Show decontamination usage example
        print(f"\nTo run decontamination on individual splits:")
        for split in splits:
            split_path = output_dir / f"{dataset_name}-{split}"
            print(f"  ./submit_decontamination.sh {split_path} /path/to/output/{dataset_name}-{split}-decontam")
    else:
        print(f"⚠️  {len(splits) - success_count} splits failed to process")
    
    return success_count == len(splits)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Split DatasetDict into separate datasets for individual processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python split_dataset.py data/03-license-filtered/smoltalk2 data/04-decontamination
  python split_dataset.py data/02-standardised/dataset-name output/splits

This will create separate datasets like:
  data/04-decontamination/smoltalk2-split1/
  data/04-decontamination/smoltalk2-split2/
  
Each output dataset will be a DatasetDict with a single "train" split,
making them compatible with decontamination and other pipeline stages.
        """
    )
    
    parser.add_argument(
        "input_dataset",
        help="Path to input DatasetDict directory"
    )
    parser.add_argument(
        "output_directory", 
        help="Directory where split datasets will be created"
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_arguments()
    
    input_path = Path(args.input_dataset)
    output_dir = Path(args.output_directory)
    
    print(f"Dataset Splitting Tool")
    print(f"Input: {input_path}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")
    
    success = split_dataset(input_path, output_dir)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()