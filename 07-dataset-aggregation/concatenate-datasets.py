#!/usr/bin/env python3
"""
concatenate_datasets.py
───────────────────────
Fast concatenation of multiple harmonized datasets that are already in the same format.

This script is designed for merging datasets that have already been processed and
share the same schema (e.g., datasets from 06_sft_mixtures_newformat).

Usage:
    # Concatenate two datasets
    ./concatenate_datasets.py dataset1 dataset2 -o output_path
    
    # Concatenate multiple datasets
    ./concatenate_datasets.py dataset1 dataset2 dataset3 -o output_path
    
    # With custom settings
    ./concatenate_datasets.py dataset1 dataset2 -o output_path --num-proc 16
"""

import json
import sys
import argparse
from pathlib import Path
from datetime import datetime, UTC
from typing import List, Dict, Any, Optional
from datasets import Dataset, DatasetDict, load_from_disk, concatenate_datasets
from tqdm import tqdm

def load_dataset_safely(path: str) -> Dataset:
    """Load a dataset and ensure it's a single Dataset (not DatasetDict)."""
    data = load_from_disk(path)
    
    if isinstance(data, DatasetDict):
        # If it's a DatasetDict, concatenate all splits
        print(f"  Dataset is a DatasetDict with splits: {list(data.keys())}")
        all_splits = []
        for split_name, split_data in data.items():
            print(f"    {split_name}: {len(split_data)} samples")
            all_splits.append(split_data)
        return concatenate_datasets(all_splits)
    else:
        # Already a single Dataset
        return data

def load_existing_metadata(output_path: Path) -> Optional[Dict[str, Any]]:
    """Load existing dataset metadata if it exists."""
    meta_file = output_path / "dataset_metadata.json"
    if meta_file.exists():
        try:
            with open(meta_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return None

def gather_dataset_statistics(dataset: Dataset) -> Dict[str, Any]:
    """Gather statistics about the dataset."""
    stats = {
        "total_samples": len(dataset),
        "columns": dataset.column_names,
        "dataset_sources": {},
        "has_system_prompts": 0,
        "has_available_functions": 0,
        "conversation_branches_count": {}
    }
    
    # Sample analysis (on first 1000 samples for efficiency)
    sample_size = min(1000, len(dataset))
    for i in tqdm(range(sample_size), desc="  Analyzing samples", leave=False):
        sample = dataset[i]
        
        # Count dataset sources
        source = sample.get("dataset_source", "unknown")
        stats["dataset_sources"][source] = stats["dataset_sources"].get(source, 0) + 1
        
        # Check for system prompts
        if sample.get("system_prompt", {}).get("content"):
            stats["has_system_prompts"] += 1
        
        # Check for available functions
        if sample.get("available_functions"):
            stats["has_available_functions"] += 1
        
        # Count conversation branches
        branches = len(sample.get("conversation_branches", []))
        stats["conversation_branches_count"][branches] = stats["conversation_branches_count"].get(branches, 0) + 1
    
    # Scale up statistics from sample
    if sample_size < len(dataset):
        scale_factor = len(dataset) / sample_size
        stats["has_system_prompts"] = int(stats["has_system_prompts"] * scale_factor)
        stats["has_available_functions"] = int(stats["has_available_functions"] * scale_factor)
        for source in stats["dataset_sources"]:
            stats["dataset_sources"][source] = int(stats["dataset_sources"][source] * scale_factor)
        for branches in stats["conversation_branches_count"]:
            stats["conversation_branches_count"][branches] = int(stats["conversation_branches_count"][branches] * scale_factor)
        stats["note"] = f"Statistics based on sample of {sample_size} items, scaled to full dataset"
    
    return stats

def save_dataset_and_metadata(dataset: Dataset, output_path: Path, 
                             input_paths: List[str], args: argparse.Namespace,
                             input_stats: List[Dict[str, Any]]):
    """Save concatenated dataset with metadata."""
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save as DatasetDict for consistency
    if args.as_datasetdict:
        dataset_dict = DatasetDict({"train": dataset})
        dataset_dict.save_to_disk(str(output_path))
    else:
        # Save as single Dataset
        dataset.save_to_disk(str(output_path))
    
    # Load existing metadata or create new
    metadata = load_existing_metadata(output_path) or {}
    
    # Gather statistics on the concatenated dataset
    print("\nGathering statistics on concatenated dataset...")
    output_stats = gather_dataset_statistics(dataset)
    
    # Create processing entry
    processing_entry = {
        "operation": "concatenate_datasets",
        "script": "concatenate_datasets.py",
        "timestamp": datetime.now(UTC).isoformat(),
        "input_paths": input_paths,
        "output_path": str(output_path),
        "num_processes": args.num_proc,
        "saved_as": "DatasetDict" if args.as_datasetdict else "Dataset",
        "input_datasets": input_stats,
        "output_statistics": output_stats,
        "description": f"Concatenated {len(input_paths)} harmonized datasets"
    }
    
    # Add to processing log
    if "processing_log" not in metadata:
        metadata["processing_log"] = []
    metadata["processing_log"].append(processing_entry)
    
    # Add format metadata if not already present
    if "format" not in metadata:
        metadata["format"] = "chat_format_v1"
    if "concatenation_details" not in metadata:
        metadata["concatenation_details"] = {
            "num_input_datasets": len(input_paths),
            "total_samples": len(dataset),
            "concatenation_method": "datasets.concatenate_datasets",
            "schema_validation": "Assumes pre-harmonized datasets with identical schemas"
        }
    
    # Save metadata
    metadata_file = output_path / "dataset_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nDataset saved to {output_path}")
    print(f"Metadata saved to {metadata_file}")
    
    # Print summary statistics
    print("\nConcatenation Summary:")
    print(f"  Total samples: {output_stats['total_samples']:,}")
    print(f"  Dataset sources found:")
    for source, count in sorted(output_stats['dataset_sources'].items()):
        print(f"    {source}: ~{count:,}")
    if output_stats.get('has_system_prompts'):
        print(f"  Samples with system prompts: ~{output_stats['has_system_prompts']:,}")
    if output_stats.get('has_available_functions'):
        print(f"  Samples with available functions: ~{output_stats['has_available_functions']:,}")

# ───────────— CLI / main ───────────── #
def cli():
    p = argparse.ArgumentParser(
        description="Concatenate multiple harmonized datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Concatenate two datasets
  %(prog)s /path/to/dataset1 /path/to/dataset2 -o /path/to/output
  
  # Concatenate with more processes
  %(prog)s dataset1 dataset2 dataset3 -o output --num-proc 16
  
  # Save as DatasetDict with 'train' split
  %(prog)s dataset1 dataset2 -o output --as-datasetdict
        """
    )
    p.add_argument("datasets", nargs="+", help="Paths to datasets to concatenate")
    p.add_argument("-o", "--output", required=True, help="Output directory path")
    p.add_argument("--num-proc", type=int, default=8, help="Number of processes for dataset operations")
    p.add_argument("--as-datasetdict", action="store_true", 
                   help="Save as DatasetDict with 'train' split (default: save as Dataset)")
    return p.parse_args()

def main():
    args = cli()
    output_path = Path(args.output)
    
    # Validate input paths
    input_paths = []
    for dataset_path in args.datasets:
        path = Path(dataset_path)
        if not path.exists():
            print(f"Error: Dataset path does not exist: {dataset_path}")
            sys.exit(1)
        input_paths.append(str(path.absolute()))
    
    if len(input_paths) < 2:
        print("Error: Need at least 2 datasets to concatenate")
        sys.exit(1)
    
    print(f"Will concatenate {len(input_paths)} datasets:")
    for path in input_paths:
        print(f"  - {path}")
    
    # Check if output exists
    if output_path.exists():
        response = input(f"\n{output_path} exists. Overwrite? [y/N]: ")
        if response.lower() != "y":
            sys.exit(0)
    
    # Load all datasets
    print("\nLoading datasets...")
    datasets_to_concat = []
    input_stats = []
    
    for i, path in enumerate(input_paths, 1):
        print(f"\n[{i}/{len(input_paths)}] Loading {Path(path).name}")
        try:
            dataset = load_dataset_safely(path)
            print(f"  Loaded {len(dataset):,} samples")
            
            # Gather basic statistics
            stats = {
                "path": path,
                "name": Path(path).name,
                "num_samples": len(dataset),
                "columns": dataset.column_names
            }
            
            # Check first sample for dataset source
            if len(dataset) > 0:
                first_sample = dataset[0]
                stats["dataset_source"] = first_sample.get("dataset_source", "unknown")
            
            input_stats.append(stats)
            datasets_to_concat.append(dataset)
            
        except Exception as e:
            print(f"  Error loading dataset: {e}")
            sys.exit(1)
    
    # Verify schemas are compatible (basic check)
    print("\nVerifying schema compatibility...")
    reference_columns = set(datasets_to_concat[0].column_names)
    for i, dataset in enumerate(datasets_to_concat[1:], 2):
        current_columns = set(dataset.column_names)
        if current_columns != reference_columns:
            print(f"Warning: Dataset {i} has different columns!")
            print(f"  Reference: {reference_columns}")
            print(f"  Current:   {current_columns}")
            print(f"  Missing:   {reference_columns - current_columns}")
            print(f"  Extra:     {current_columns - reference_columns}")
            response = input("Continue anyway? [y/N]: ")
            if response.lower() != "y":
                sys.exit(1)
    
    # Concatenate datasets
    print("\nConcatenating datasets...")
    try:
        concatenated = concatenate_datasets(datasets_to_concat)
        print(f"Successfully concatenated: {len(concatenated):,} total samples")
    except Exception as e:
        print(f"Error during concatenation: {e}")
        print("\nThis usually happens when datasets have incompatible schemas.")
        print("Please ensure all datasets have been processed with the same converter.")
        sys.exit(1)
    
    # Save the result
    print("\nSaving concatenated dataset...")
    save_dataset_and_metadata(concatenated, output_path, input_paths, args, input_stats)
    
    print("\nConcatenation complete!")

if __name__ == "__main__":
    main()