#!/usr/bin/env python3
"""
License-based dataset filtering for specific supported datasets.

Removes samples from datasets based on known licensing issues or 
problematic sources. Each dataset has custom filtering rules.
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
from datasets import load_from_disk, Dataset, DatasetDict, concatenate_datasets
from tqdm import tqdm
import gc
import tempfile


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


# Dataset filtering configurations
DATASET_FILTERS = {
    "tulu-3-sft-mixture": {
        "filter_type": "exclude_sources",
        "field": "original_metadata.source",
        "values": [
            "ai2-adapt-dev/tulu_hard_coded_repeated_10",
            "ai2-adapt-dev/no_robots_converted",
            "ai2-adapt-dev/flan_v2_converted"
        ],
        "reason": "Licensing restrictions"
    },
    
    "tulu-3-sft-olmo-2-mixture-0225": {
        "filter_type": "exclude_sources",
        "field": "original_metadata.source",
        "values": [
            "olmo_hardcoded",
            "ai2-adapt-dev/no_robots_converted",
            "ai2-adapt-dev/flan_v2_converted"
        ],
        "reason": "Licensing restrictions"
    },
    
    "smoltalk": {
        "filter_type": "exclude_sources",
        "field": "original_metadata.source",
        "values": [
            "openhermes-100k",
            "longalign",
            "explore-instruct-rewriting"
        ],
        "reason": "Licensing restrictions"
    },
    
    "smoltalk2": {
        "filter_type": "remove_splits",
        "splits_to_remove": [
            "LongAlign_64k_Qwen3_32B_yarn_131k_think",
            "LongAlign_64k_context_lang_annotated_lang_6_no_think",
            "OpenHermes_2.5_no_think",
            "smoltalk_smollm3_explore_instruct_rewriting_no_think",
            "Mixture_of_Thoughts_science_no_think",
            "hermes_function_calling_v1_no_think",
            "xlam_traces_no_think",
            "smolagents_toolcalling_traces_think"
        ],
        "augment_source": True,
        "reason": "Schema incompatibility and licensing"
    },
    
    "The-Tome": {
        "filter_type": "exclude_sources",
        "field": "original_metadata.dataset",
        "values": [
            "infini-instruct-top-500k",
            "ultrainteract_trajectories_sharegpt",
            "qwen2-72b-magpie-en"
        ],
        "reason": "Licensing restrictions"
    },
    
    "AceReason-1.1-SFT": {
        "filter_type": "exclude_sources",
        "field": "original_metadata.source",
        "values": ["leetcode"],
        "reason": "Licensing restrictions"
    },
    
    "Llama-Nemotron-Post-Training-Dataset": {
        "filter_type": "include_licenses",
        "field": "original_metadata.license",
        "values": ["cc-by-4.0", "odc-by"],
        "reason": "Keep only open licenses"
    },
    
    "EuroBlocks-SFT-Synthetic-1124": {
        "filter_type": "no_filter",
        "reason": "No filtering needed"
    }
}


def apply_exclusion_filter(dataset: Dataset, field_path: str, excluded_values: list) -> Dataset:
    """Apply exclusion filter to dataset."""
    print(f"Excluding samples where {field_path} in {excluded_values}")
    
    def filter_func(sample):
        value = get_nested_value(sample, field_path)
        return value not in excluded_values
    
    return dataset.filter(filter_func)


def apply_exclusion_filter_chunked(dataset: Dataset, field_path: str, excluded_values: list, chunk_size: int = 100000) -> Dataset:
    """Apply exclusion filter to dataset using chunked processing."""
    print(f"Excluding samples where {field_path} in {excluded_values} (chunked: {chunk_size:,})")
    
    def filter_func(sample):
        value = get_nested_value(sample, field_path)
        return value not in excluded_values
    
    total_rows = len(dataset)
    if total_rows <= chunk_size:
        # Small dataset, use regular filter
        return dataset.filter(filter_func)
    
    filtered_chunks = []
    filtered_count = 0
    
    with tqdm(total=total_rows, desc="Filtering", unit="samples") as pbar:
        for chunk_start in range(0, total_rows, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_rows)
            
            # Create chunk indices
            chunk_indices = list(range(chunk_start, chunk_end))
            chunk = dataset.select(chunk_indices)
            
            # Apply filter to chunk
            filtered_chunk = chunk.filter(filter_func)
            
            if len(filtered_chunk) > 0:
                filtered_chunks.append(filtered_chunk)
                filtered_count += len(filtered_chunk)
            
            pbar.update(chunk_end - chunk_start)
            pbar.set_postfix(kept=filtered_count)
            
            # Force garbage collection
            del chunk, filtered_chunk
            gc.collect()
    
    if not filtered_chunks:
        # Return empty dataset with same structure
        return Dataset.from_list([])
    
    print(f"Concatenating {len(filtered_chunks)} filtered chunks...")
    result = concatenate_datasets(filtered_chunks)
    
    # Clean up
    del filtered_chunks
    gc.collect()
    
    return result


def apply_inclusion_filter(dataset: Dataset, field_path: str, allowed_values: list) -> Dataset:
    """Apply inclusion filter to dataset."""
    print(f"Including only samples where {field_path} in {allowed_values}")
    
    def filter_func(sample):
        value = get_nested_value(sample, field_path)
        return value in allowed_values
    
    return dataset.filter(filter_func)


def apply_inclusion_filter_chunked(dataset: Dataset, field_path: str, allowed_values: list, chunk_size: int = 100000) -> Dataset:
    """Apply inclusion filter to dataset using chunked processing."""
    print(f"Including only samples where {field_path} in {allowed_values} (chunked: {chunk_size:,})")
    
    def filter_func(sample):
        value = get_nested_value(sample, field_path)
        return value in allowed_values
    
    total_rows = len(dataset)
    if total_rows <= chunk_size:
        # Small dataset, use regular filter
        return dataset.filter(filter_func)
    
    filtered_chunks = []
    filtered_count = 0
    
    with tqdm(total=total_rows, desc="Filtering", unit="samples") as pbar:
        for chunk_start in range(0, total_rows, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_rows)
            
            # Create chunk indices
            chunk_indices = list(range(chunk_start, chunk_end))
            chunk = dataset.select(chunk_indices)
            
            # Apply filter to chunk
            filtered_chunk = chunk.filter(filter_func)
            
            if len(filtered_chunk) > 0:
                filtered_chunks.append(filtered_chunk)
                filtered_count += len(filtered_chunk)
            
            pbar.update(chunk_end - chunk_start)
            pbar.set_postfix(kept=filtered_count)
            
            # Force garbage collection
            del chunk, filtered_chunk
            gc.collect()
    
    if not filtered_chunks:
        # Return empty dataset with same structure
        return Dataset.from_list([])
    
    print(f"Concatenating {len(filtered_chunks)} filtered chunks...")
    result = concatenate_datasets(filtered_chunks)
    
    # Clean up
    del filtered_chunks
    gc.collect()
    
    return result


def handle_split_removal(dataset_dict: DatasetDict, filter_config: dict) -> Dataset:
    """Handle the complex smoltalk2 case with split removal and concatenation."""
    splits_to_remove = filter_config["splits_to_remove"]
    
    print(f"Removing splits: {splits_to_remove}")
    
    # Keep only splits not in removal list
    remaining_splits = {}
    for split_name in dataset_dict.keys():
        if split_name not in splits_to_remove:
            remaining_splits[split_name] = dataset_dict[split_name]
            print(f"Keeping split: {split_name} ({len(dataset_dict[split_name]):,} samples)")
    
    if filter_config.get("augment_source", False):
        print("Augmenting remaining splits with source metadata and concatenating...")
        
        # Add source metadata and concatenate
        augmented_datasets = []
        for split_name, split_data in remaining_splits.items():
            print(f"Augmenting split {split_name}...")
            
            def add_source_metadata(sample):
                original_metadata = sample.get("original_metadata", {})
                return {
                    **sample,
                    "original_metadata": {
                        **original_metadata,
                        "source": split_name
                    }
                }
            
            augmented = split_data.map(add_source_metadata)
            augmented_datasets.append(augmented)
        
        # Concatenate all splits
        print("Concatenating augmented splits...")
        return concatenate_datasets(augmented_datasets)
    else:
        # Return as DatasetDict
        return DatasetDict(remaining_splits)


def handle_split_removal_chunked(dataset_dict: DatasetDict, filter_config: dict, chunk_size: int = 100000) -> Dataset:
    """Handle split removal with chunked processing for large datasets."""
    splits_to_remove = filter_config["splits_to_remove"]
    
    print(f"Removing splits: {splits_to_remove}")
    
    # Keep only splits not in removal list
    remaining_splits = {}
    total_samples = 0
    for split_name in dataset_dict.keys():
        if split_name not in splits_to_remove:
            remaining_splits[split_name] = dataset_dict[split_name]
            split_size = len(dataset_dict[split_name])
            total_samples += split_size
            print(f"Keeping split: {split_name} ({split_size:,} samples)")
    
    if filter_config.get("augment_source", False):
        print("Augmenting remaining splits with source metadata and concatenating...")
        
        # Decide on chunked vs regular processing
        use_chunked = total_samples > chunk_size
        
        if use_chunked:
            print(f"Using chunked processing for {total_samples:,} total samples...")
            
        # Add source metadata and collect datasets
        augmented_datasets = []
        for split_name, split_data in remaining_splits.items():
            print(f"Augmenting split {split_name}...")
            
            def add_source_metadata(sample):
                original_metadata = sample.get("original_metadata", {})
                return {
                    **sample,
                    "original_metadata": {
                        **original_metadata,
                        "source": split_name
                    }
                }
            
            if use_chunked and len(split_data) > chunk_size:
                # Process split in chunks
                split_chunks = []
                with tqdm(total=len(split_data), desc=f"Augmenting {split_name}", unit="samples") as pbar:
                    for chunk_start in range(0, len(split_data), chunk_size):
                        chunk_end = min(chunk_start + chunk_size, len(split_data))
                        chunk_indices = list(range(chunk_start, chunk_end))
                        chunk = split_data.select(chunk_indices)
                        
                        augmented_chunk = chunk.map(add_source_metadata)
                        split_chunks.append(augmented_chunk)
                        
                        pbar.update(chunk_end - chunk_start)
                        
                        # Force garbage collection
                        del chunk, augmented_chunk
                        gc.collect()
                
                # Concatenate chunks for this split
                print(f"Concatenating {len(split_chunks)} chunks for split {split_name}...")
                augmented_split = concatenate_datasets(split_chunks)
                augmented_datasets.append(augmented_split)
                
                # Clean up
                del split_chunks
                gc.collect()
            else:
                # Regular processing for smaller splits
                augmented = split_data.map(add_source_metadata)
                augmented_datasets.append(augmented)
        
        # Concatenate all splits
        print("Concatenating all augmented splits...")
        result = concatenate_datasets(augmented_datasets)
        
        # Clean up
        del augmented_datasets
        gc.collect()
        
        return result
    else:
        # Return as DatasetDict
        return DatasetDict(remaining_splits)


def apply_dataset_filter(dataset, dataset_name: str, filter_config: dict, chunk_size: int = 100000):
    """Apply filtering based on configuration with optional chunking."""
    
    filter_type = filter_config["filter_type"]
    
    # Determine if we should use chunking based on dataset size
    total_samples = count_samples(dataset)
    use_chunking = total_samples > 500000  # Same threshold as convert script
    
    if use_chunking:
        print(f"Large dataset detected ({total_samples:,} samples), using chunked processing...")
    
    if filter_type == "exclude_sources":
        # Standard exclusion filter
        field_path = filter_config["field"]
        excluded_values = filter_config["values"]
        
        if hasattr(dataset, 'keys'):
            # DatasetDict - filter each split
            filtered_splits = {}
            for split_name in dataset.keys():
                split_data = dataset[split_name]
                if use_chunking and len(split_data) > chunk_size:
                    filtered_split = apply_exclusion_filter_chunked(split_data, field_path, excluded_values, chunk_size)
                else:
                    filtered_split = apply_exclusion_filter(split_data, field_path, excluded_values)
                
                if len(filtered_split) > 0:
                    filtered_splits[split_name] = filtered_split
            return DatasetDict(filtered_splits)
        else:
            # Single dataset
            if use_chunking:
                return apply_exclusion_filter_chunked(dataset, field_path, excluded_values, chunk_size)
            else:
                return apply_exclusion_filter(dataset, field_path, excluded_values)
            
    elif filter_type == "include_licenses":
        # Include only specific licenses
        field_path = filter_config["field"]
        allowed_values = filter_config["values"]
        
        if hasattr(dataset, 'keys'):
            # DatasetDict - filter each split
            filtered_splits = {}
            for split_name in dataset.keys():
                split_data = dataset[split_name]
                if use_chunking and len(split_data) > chunk_size:
                    filtered_split = apply_inclusion_filter_chunked(split_data, field_path, allowed_values, chunk_size)
                else:
                    filtered_split = apply_inclusion_filter(split_data, field_path, allowed_values)
                
                if len(filtered_split) > 0:
                    filtered_splits[split_name] = filtered_split
            return DatasetDict(filtered_splits)
        else:
            # Single dataset
            if use_chunking:
                return apply_inclusion_filter_chunked(dataset, field_path, allowed_values, chunk_size)
            else:
                return apply_inclusion_filter(dataset, field_path, allowed_values)
            
    elif filter_type == "remove_splits":
        # Special handling for DatasetDict split removal
        if not hasattr(dataset, 'keys'):
            raise ValueError("remove_splits filter requires DatasetDict input")
        
        if use_chunking:
            result = handle_split_removal_chunked(dataset, filter_config, chunk_size)
        else:
            result = handle_split_removal(dataset, filter_config)
        
        # Ensure we return DatasetDict
        if not hasattr(result, 'keys'):
            return DatasetDict({"train": result})
        return result
        
    elif filter_type == "no_filter":
        # No filtering needed
        print("No filtering applied")
        return dataset
        
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")


def count_samples(dataset) -> int:
    """Count total samples in dataset or DatasetDict."""
    if hasattr(dataset, 'keys'):
        return sum(len(dataset[split]) for split in dataset.keys())
    else:
        return len(dataset)


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


def save_with_metadata(dataset, output_path: Path, input_path: Path, 
                      dataset_name: str, filter_config: dict, 
                      original_count: int, filtered_count: int):
    """Save dataset with proper metadata tracking."""
    
    # Load existing metadata
    original_metadata = load_existing_metadata(input_path)
    
    samples_removed = original_count - filtered_count
    
    # Create processing log entry
    processing_entry = {
        "operation": "license_filtering",
        "script": "license_filter.py",
        "timestamp": datetime.now().isoformat(),
        "input_path": str(input_path),
        "output_path": str(output_path),
        "dataset_name": dataset_name,
        "filter_config": filter_config,
        "original_samples": original_count,
        "filtered_samples": filtered_count,
        "samples_removed": samples_removed,
        "removal_percentage": (samples_removed / original_count * 100) if original_count > 0 else 0
    }
    
    # Update metadata
    metadata = {
        **original_metadata,
        "processing_log": original_metadata.get("processing_log", []) + [processing_entry]
    }
    
    # Save dataset and metadata
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Saving filtered dataset to {output_path}...")
    dataset.save_to_disk(str(output_path))
    
    with open(output_path / "dataset_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Dataset saved with {filtered_count:,} samples ({samples_removed:,} removed)")
    print(f"Metadata saved to {output_path}/dataset_metadata.json")


def list_filters():
    """List all configured filters."""
    print("Available dataset filters:")
    print("=" * 80)
    
    for dataset_name, config in DATASET_FILTERS.items():
        print(f"\n{dataset_name}:")
        print(f"  Type: {config['filter_type']}")
        print(f"  Reason: {config['reason']}")
        
        if config["filter_type"] == "exclude_sources":
            print(f"  Field: {config['field']}")
            print(f"  Excluded values: {config['values']}")
        elif config["filter_type"] == "include_licenses":
            print(f"  Field: {config['field']}")
            print(f"  Allowed values: {config['values']}")
        elif config["filter_type"] == "remove_splits":
            print(f"  Removed splits: {config['splits_to_remove']}")
            print(f"  Augment source: {config.get('augment_source', False)}")


def process_dataset(dataset_path: Path, output_path: Path, force_format: str = None, chunk_size: int = 100000):
    """Process dataset with proper DatasetDict handling."""
    
    # Validate input
    if not dataset_path.exists():
        print(f"Error: Dataset path does not exist: {dataset_path}")
        return False
    
    dataset_name = dataset_path.name
    
    # Determine which filter config to use
    if force_format:
        if force_format not in DATASET_FILTERS:
            print(f"Error: Invalid force-format '{force_format}'")
            print(f"Supported formats: {list(DATASET_FILTERS.keys())}")
            return False
        filter_config = DATASET_FILTERS[force_format]
        print(f"Using forced format: {force_format}")
    else:
        # Check if filtering is defined for this dataset
        if dataset_name not in DATASET_FILTERS:
            print(f"Error: No filtering rules defined for dataset '{dataset_name}'")
            print(f"Supported datasets: {list(DATASET_FILTERS.keys())}")
            print(f"Tip: Use --force-format to specify a filter (e.g., --force-format smoltalk)")
            return False
        filter_config = DATASET_FILTERS[dataset_name]
    
    try:
        # Load dataset
        print(f"Loading dataset from: {dataset_path}")
        dataset = load_from_disk(str(dataset_path))
        
        # Show dataset info
        if hasattr(dataset, 'keys'):
            splits = list(dataset.keys())
            print(f"Found DatasetDict with splits: {splits}")
        else:
            print(f"Found single Dataset")
        
        original_count = count_samples(dataset)
        print(f"Original sample count: {original_count:,}")
        
        # Apply filtering
        print(f"\nApplying filter: {filter_config['reason']}")
        filtered_dataset = apply_dataset_filter(dataset, dataset_name, filter_config, chunk_size)
        
        # Ensure output is DatasetDict (pipeline convention)
        if not hasattr(filtered_dataset, 'keys'):
            print("Wrapping single Dataset in DatasetDict")
            filtered_dataset = DatasetDict({"train": filtered_dataset})
        
        filtered_count = count_samples(filtered_dataset)
        
        # Save with metadata
        save_with_metadata(filtered_dataset, output_path, dataset_path, 
                          dataset_name, filter_config, original_count, filtered_count)
        
        return True
        
    except Exception as e:
        print(f"Error processing dataset: {e}")
        return False


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Apply license-based filtering to specific datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Supported datasets:
  - tulu-3-sft-mixture
  - tulu-3-sft-olmo-2-mixture-0225
  - smoltalk
  - smoltalk2
  - The-Tome
  - AceReason-1.1-SFT
  - Llama-Nemotron-Post-Training-Dataset
  - EuroBlocks-SFT-Synthetic-1124

Examples:
  python license_filter.py data/02-standardised/tulu-3-sft-mixture
  python license_filter.py data/02-standardised/smoltalk --output data/03-license-filtered/smoltalk
  python license_filter.py data/02-standardised/Llama-Nemotron-Post-Training-Dataset --chunk-size 50000
  python license_filter.py --list-filters
        """
    )
    
    parser.add_argument("dataset_path", nargs='?', help="Path to input dataset")
    parser.add_argument("--output", "-o", help="Output path (default: data/03-license-filtered/DATASET_NAME)")
    parser.add_argument("--list-filters", action="store_true", 
                       help="List all configured filters and exit")
    parser.add_argument("--force-format", type=str, 
                       choices=list(DATASET_FILTERS.keys()),
                       help="Force using a specific dataset's filter configuration (e.g., --force-format smoltalk)")
    parser.add_argument("--chunk-size", type=int, default=100000,
                       help="Number of samples to process in each chunk for memory efficiency (default: 100000)")
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_arguments()
    
    if args.list_filters:
        list_filters()
        return
    
    if not args.dataset_path:
        print("Error: dataset_path is required (use --help for usage)")
        sys.exit(1)
    
    dataset_path = Path(args.dataset_path)
    dataset_name = dataset_path.name
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path("data/03-license-filtered") / dataset_name
    
    print(f"Processing dataset: {dataset_name}")
    print(f"Input path: {dataset_path}")
    print(f"Output path: {output_path}")
    
    success = process_dataset(dataset_path, output_path, args.force_format, args.chunk_size)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
