#!/usr/bin/env python3
"""
Memory-efficient field-based dataset filtering tool for new format datasets.

This script analyzes and filters new chat format datasets (with parts structure) 
based on field values. It can display field statistics and create filtered copies 
of datasets. Uses HuggingFace's native filtering for memory efficiency on large datasets.

Supports filtering on parts-specific fields like:
- conversation_branches[0].messages[0].parts[0].type
- conversation_branches[0].messages[0].parts[0].name
- available_functions[0].name
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Set, Union, Optional
from collections import Counter, defaultdict
from datasets import load_from_disk, Dataset, DatasetDict
from tqdm import tqdm




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


def analyze_field_structure(dataset, max_samples: Optional[int] = None) -> Dict[str, Any]:
    """
    Analyze the field structure of a dataset.
    
    Args:
        dataset: HuggingFace Dataset
        max_samples: Maximum number of samples to analyze (None = all samples)
    
    Returns:
        Dictionary with field analysis results
    """
    field_counts = defaultdict(Counter)
    field_types = defaultdict(set)
    
    if max_samples is None:
        sample_count = len(dataset)
        print(f"Analyzing field structure from all {sample_count:,} samples...")
    else:
        sample_count = min(len(dataset), max_samples)
        print(f"Analyzing field structure from {sample_count:,} samples (use --max-samples -1 to analyze all samples)...")
    
    with tqdm(total=sample_count, desc="Analyzing", unit="samples") as pbar:
        for i, sample in enumerate(dataset):
            if max_samples is not None and i >= max_samples:
                break
                
            # Analyze all fields recursively
            _analyze_dict_recursive(sample, "", field_counts, field_types)
            pbar.update(1)
    
    return {
        "field_counts": dict(field_counts),
        "field_types": {k: list(v) for k, v in field_types.items()},
        "total_samples_analyzed": sample_count,
        "total_dataset_size": len(dataset)
    }


def _analyze_dict_recursive(obj: Any, prefix: str, field_counts: defaultdict, field_types: defaultdict):
    """Recursively analyze dictionary structure."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            field_path = f"{prefix}.{key}" if prefix else key
            
            # Count non-null values
            if value is not None:
                field_counts[field_path][str(value)] += 1
                field_types[field_path].add(type(value).__name__)
            else:
                field_counts[field_path]["<NULL>"] += 1
                field_types[field_path].add("NoneType")
            
            # Recurse into nested structures
            if isinstance(value, dict):
                _analyze_dict_recursive(value, field_path, field_counts, field_types)
            elif isinstance(value, list) and value:
                # Analyze first few items in list
                for i, item in enumerate(value[:3]):  # Limit to first 3 items
                    list_path = f"{field_path}[{i}]"
                    _analyze_dict_recursive(item, list_path, field_counts, field_types)
    elif isinstance(obj, list):
        for i, item in enumerate(obj[:3]):  # Limit to first 3 items
            list_path = f"{prefix}[{i}]" if prefix else f"[{i}]"
            _analyze_dict_recursive(item, list_path, field_counts, field_types)


def display_schema(analysis: Dict[str, Any]):
    """Display dataset schema (field names and types only)."""
    print(f"\n{'='*80}")
    print(f"DATASET SCHEMA")
    print(f"{'='*80}")
    print(f"Dataset size: {analysis['total_dataset_size']:,} samples")
    print(f"Samples analyzed: {analysis['total_samples_analyzed']:,}")
    print(f"Total unique fields found: {len(analysis['field_counts'])}")
    
    # Sort fields by path for organized display
    sorted_fields = sorted(analysis['field_counts'].keys())
    
    print(f"\nField paths and types:")
    for field_path in sorted_fields:
        counts = analysis['field_counts'][field_path]
        types = analysis['field_types'][field_path]
        non_null_count = sum(1 for v in counts.keys() if v != '<NULL>')
        
        print(f"  {field_path:<50} {', '.join(types):<20} ({non_null_count:,} unique non-null values)")


def display_field_stats(analysis: Dict[str, Any], field_path: str):
    """Display statistics for a specific field."""
    if field_path not in analysis['field_counts']:
        print(f"Error: Field '{field_path}' not found in dataset")
        print(f"Available fields: {', '.join(sorted(analysis['field_counts'].keys()))}")
        return
    
    counts = analysis['field_counts'][field_path]
    types = analysis['field_types'][field_path]
    
    print(f"\n{'='*80}")
    print(f"FIELD STATISTICS: {field_path}")
    print(f"{'='*80}")
    print(f"Dataset size: {analysis['total_dataset_size']:,} samples")
    print(f"Samples analyzed: {analysis['total_samples_analyzed']:,}")
    print(f"Field types: {', '.join(types)}")
    print(f"Unique values: {len(counts):,}")
    non_null_occurrences = sum(count for v, count in counts.items() if v != '<NULL>')
    unique_non_null = sum(1 for v in counts.keys() if v != '<NULL>')
    print(f"Non-null occurrences: {non_null_occurrences:,} samples")
    print(f"Unique non-null values: {unique_non_null:,}")
    
    # Show all unique values
    total_unique = len(counts)
    most_common = counts.most_common()
    print(f"\nAll {total_unique} unique values:")
    
    for value, count in most_common:
        percentage = (count / analysis['total_samples_analyzed']) * 100
        display_value = str(value)
        # Add quotes around the value
        quoted_value = f'"{display_value}"'
        print(f"  {quoted_value} {count:>8,} ({percentage:5.1f}%)")


def display_field_analysis(analysis: Dict[str, Any]):
    """Display full field analysis results (legacy function for backward compatibility)."""
    print(f"\n{'='*80}")
    print(f"FIELD ANALYSIS RESULTS")
    print(f"{'='*80}")
    print(f"Dataset size: {analysis['total_dataset_size']:,} samples")
    print(f"Samples analyzed: {analysis['total_samples_analyzed']:,}")
    print(f"Total unique fields found: {len(analysis['field_counts'])}")
    
    # Sort fields by path for organized display
    sorted_fields = sorted(analysis['field_counts'].keys())
    
    for field_path in sorted_fields:
        counts = analysis['field_counts'][field_path]
        types = analysis['field_types'][field_path]
        
        print(f"\n{'-'*60}")
        print(f"Field: {field_path}")
        print(f"Types: {', '.join(types)}")
        print(f"Unique values: {len(counts)}")
        non_null_occurrences = sum(count for v, count in counts.items() if v != '<NULL>')
        unique_non_null = sum(1 for v in counts.keys() if v != '<NULL>')
        print(f"Non-null occurrences: {non_null_occurrences}")
        print(f"Unique non-null values: {unique_non_null}")
        
        # Show all values
        most_common = counts.most_common()
        print(f"All {len(counts)} unique values:")
        
        for value, count in most_common:
            percentage = (count / analysis['total_samples_analyzed']) * 100
            display_value = str(value)
            # Add quotes around the value
            quoted_value = f'"{display_value}"'
            print(f"  {quoted_value} {count:>8,} ({percentage:5.1f}%)")


def filter_dataset(dataset, field_path: str, target_values: List[str], keep_matches: bool = True) -> Dataset:
    """
    Filter dataset based on field values.
    
    Args:
        dataset: HuggingFace Dataset
        field_path: Dot-separated field path
        target_values: List of values to match
        keep_matches: If True, keep matching samples; if False, remove them
    
    Returns:
        Filtered Dataset
    """
    print(f"Filtering dataset on field '{field_path}'...")
    quoted_values = [f'"{v}"' for v in target_values]
    print(f"Target values: {', '.join(quoted_values)}")
    print(f"Action: {'KEEP' if keep_matches else 'REMOVE'} matching samples")
    
    filtered_samples = []
    matches_found = 0
    
    with tqdm(total=len(dataset), desc="Filtering") as pbar:
        for sample in dataset:
            field_value = get_nested_value(sample, field_path)
            field_value_str = str(field_value) if field_value is not None else "<NULL>"
            
            is_match = field_value_str in target_values
            
            if is_match:
                matches_found += 1
            
            # Keep sample based on filter logic
            if (keep_matches and is_match) or (not keep_matches and not is_match):
                filtered_samples.append(sample)
            
            pbar.update(1)
    
    print(f"Filtering complete:")
    print(f"  - Original samples: {len(dataset):,}")
    print(f"  - Matching samples: {matches_found:,}")
    print(f"  - Filtered samples: {len(filtered_samples):,}")
    
    return Dataset.from_list(filtered_samples)


def save_filtered_dataset(dataset, output_path: Path, metadata: Dict[str, Any]):
    """Save filtered dataset with metadata."""
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Check if dataset is empty
    if len(dataset) == 0:
        print(f"Warning: Filtered dataset is empty. No matches found for the specified criteria.")
        print(f"Creating empty dataset directory at {output_path}")
        
        # Still save metadata to track the filtering operation
        metadata_file = output_path / "dataset_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata saved to {metadata_file}")
        return
    
    print(f"Saving filtered dataset to {output_path}...")
    dataset.save_to_disk(str(output_path))
    
    # Save metadata
    metadata_file = output_path / "dataset_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Dataset saved: {len(dataset):,} samples")
    print(f"Metadata saved to {metadata_file}")


def split_dataset_by_field(dataset, field_path: str, output_base_path: Path, dataset_name: str, 
                          original_metadata: Dict[str, Any], input_path: Path, split_prefix: str = None) -> None:
    """
    Split dataset into multiple datasets based on field values using memory-efficient approach.
    
    Args:
        dataset: HuggingFace Dataset or DatasetDict
        field_path: Field to split on
        output_base_path: Base directory for output datasets
        dataset_name: Name of the input dataset
        original_metadata: Original dataset metadata
        input_path: Path to input dataset
        split_prefix: Custom prefix for output dataset names (default: dataset_name-)
    """
    # First, analyze the field to get all unique values
    print(f"Analyzing field '{field_path}' to find unique values...")
    
    # Collect unique values
    unique_values = set()
    is_dataset_dict = hasattr(dataset, 'keys')
    
    if is_dataset_dict:
        # Analyze all splits
        for split_name in dataset.keys():
            split_data = dataset[split_name]
            for sample in tqdm(split_data, desc=f"Analyzing {split_name}"):
                value = get_nested_value(sample, field_path)
                value_str = str(value) if value is not None else "<NULL>"
                unique_values.add(value_str)
    else:
        # Single dataset
        for sample in tqdm(dataset, desc="Analyzing dataset"):
            value = get_nested_value(sample, field_path)
            value_str = str(value) if value is not None else "<NULL>"
            unique_values.add(value_str)
    
    print(f"\nFound {len(unique_values)} unique values for field '{field_path}'")
    
    # Create filename mappings with collision detection
    value_to_filename = {}
    filename_counts = {}
    
    for unique_value in sorted(unique_values):
        base_filename = sanitize_filename(unique_value)
        
        # Check for collision
        if base_filename in filename_counts:
            # Add suffix to make unique
            count = filename_counts[base_filename]
            filename = f"{base_filename}_{count}"
            filename_counts[base_filename] += 1
        else:
            filename = base_filename
            filename_counts[base_filename] = 1
        
        value_to_filename[unique_value] = filename
    
    # Show the user what will be created
    print(f"\nWill create the following dataset splits:")
    for value, filename in sorted(value_to_filename.items()):
        if split_prefix:
            output_name = f"{dataset_name}-{split_prefix}{filename}"
        else:
            output_name = f"{dataset_name}-{filename}"
        output_path = output_base_path / output_name
        display_value = value if len(value) <= 30 else value[:27] + "..."
        print(f'  "{display_value}" → {output_path}')
    
    print(f"\nTotal: {len(unique_values)} datasets will be created")
    
    # Warn if many splits
    if len(unique_values) > 20:
        print(f"\n⚠️  WARNING: This will create {len(unique_values)} separate datasets!")
        response = input("Continue? [y/N] ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    
    # Create output datasets using memory-efficient approach
    print(f"\nSplitting dataset using memory-efficient approach...")
    
    # Process each unique value separately
    for value, filename in sorted(value_to_filename.items()):
        print(f"\n{'='*60}")
        print(f"Creating split for value: '{value}'")
        print(f"{'='*60}")
        
        if split_prefix:
            output_name = f"{dataset_name}-{split_prefix}{filename}"
        else:
            output_name = f"{dataset_name}-{filename}"
        output_path = output_base_path / output_name
        
        # Define filter function for this value
        def make_filter_fn(target_value):
            def filter_fn(sample):
                field_value = get_nested_value(sample, field_path)
                field_value_str = str(field_value) if field_value is not None else "<NULL>"
                return field_value_str == target_value
            return filter_fn
        
        filter_fn = make_filter_fn(value)
        
        if is_dataset_dict:
            # For DatasetDict, filter each split
            filtered_splits = {}
            total_split_samples = 0
            
            for split_name in dataset.keys():
                print(f"  Filtering split '{split_name}'...")
                
                # Use HuggingFace's native filter with batching
                filtered_split = dataset[split_name].filter(
                    filter_fn,
                    batch_size=10000,  # Process in 10k sample batches
                    desc=f"Filtering {split_name}"
                )
                
                if len(filtered_split) > 0:
                    filtered_splits[split_name] = filtered_split
                    total_split_samples += len(filtered_split)
                    print(f"    Found {len(filtered_split):,} matching samples")
                else:
                    print(f"    No matching samples found")
            
            if filtered_splits:
                # Create DatasetDict and save immediately
                output_dataset = DatasetDict(filtered_splits)
                
                # Create metadata
                split_metadata = {
                    **original_metadata,
                    "processing_log": original_metadata.get("processing_log", []) + [{
                        "operation": "field_based_split",
                        "script": "field_filter_newformat.py",
                        "timestamp": datetime.now().isoformat(),
                        "input_path": str(input_path),
                        "output_path": str(output_path),
                        "split_field": field_path,
                        "split_value": value,
                        "samples": total_split_samples,
                        "source_dataset": dataset_name
                    }]
                }
                
                # Save dataset and metadata
                print(f"  Saving to {output_path}...")
                output_path.mkdir(parents=True, exist_ok=True)
                output_dataset.save_to_disk(str(output_path))
                
                metadata_file = output_path / "dataset_metadata.json"
                with open(metadata_file, 'w') as f:
                    json.dump(split_metadata, f, indent=2)
                
                print(f"  ✓ Saved {output_path.name} ({total_split_samples:,} samples)")
            else:
                print(f"  ⚠️  No samples found for value '{value}', skipping...")
                
        else:
            # For single Dataset
            print(f"  Filtering dataset...")
            
            # Use HuggingFace's native filter
            filtered_dataset = dataset.filter(
                filter_fn,
                batch_size=10000,
                desc="Filtering"
            )
            
            if len(filtered_dataset) > 0:
                # Create metadata
                split_metadata = {
                    **original_metadata,
                    "processing_log": original_metadata.get("processing_log", []) + [{
                        "operation": "field_based_split",
                        "script": "field_filter_newformat.py",
                        "timestamp": datetime.now().isoformat(),
                        "input_path": str(input_path),
                        "output_path": str(output_path),
                        "split_field": field_path,
                        "split_value": value,
                        "samples": len(filtered_dataset),
                        "source_dataset": dataset_name
                    }]
                }
                
                # Save immediately
                print(f"  Saving to {output_path}...")
                output_path.mkdir(parents=True, exist_ok=True)
                filtered_dataset.save_to_disk(str(output_path))
                
                metadata_file = output_path / "dataset_metadata.json"
                with open(metadata_file, 'w') as f:
                    json.dump(split_metadata, f, indent=2)
                
                print(f"  ✓ Saved {output_path.name} ({len(filtered_dataset):,} samples)")
            else:
                print(f"  ⚠️  No samples found for value '{value}', skipping...")
    
    print(f"\n{'='*60}")
    print(f"SPLIT COMPLETE")
    print(f"{'='*60}")
    print(f"Created datasets in: {output_base_path}")


def sanitize_filename(value: str, max_length: int = 50) -> str:
    """
    Sanitize a field value to be a valid filename component.
    
    Args:
        value: The field value to sanitize
        max_length: Maximum length for the filename
        
    Returns:
        Sanitized filename string
    """
    # Handle special cases first
    if value is None or value == "None":
        return "NULL"
    if value == "":
        return "EMPTY"
    if value == "<NULL>":  # Our internal NULL representation
        return "NULL"
    
    # Convert to string if not already
    value = str(value)
    
    # Replace problematic characters
    replacements = {
        '/': '_',      # Path separator
        '\\': '_',     # Windows path separator
        ':': '-',      # Drive separator (Windows)
        '*': 'star',  # Wildcard
        '?': 'q',      # Wildcard
        '"': '',       # Quote
        '<': 'lt',     # Less than
        '>': 'gt',     # Greater than
        '|': '_',      # Pipe
        '\n': '_',     # Newline
        '\r': '_',     # Carriage return
        '\t': '_',     # Tab
        ' ': '_',      # Space
        '.': '_',      # Dots can be problematic at start/end
    }
    
    for old, new in replacements.items():
        value = value.replace(old, new)
    
    # Remove any remaining non-ASCII or control characters
    value = ''.join(c if c.isalnum() or c in '-_' else '_' for c in value)
    
    # Clean up multiple underscores
    while '__' in value:
        value = value.replace('__', '_')
    
    # Trim underscores from ends
    value = value.strip('_')
    
    # Truncate if too long
    if len(value) > max_length:
        value = value[:max_length] + '_trunc'
    
    # Ensure non-empty
    if not value:
        return "UNNAMED"
    
    return value


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze and filter new format chat datasets (with parts structure) by field values",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show dataset schema (all fields)
  python field_filter_newformat.py data/02-standardised/xlam-function-calling-60k

  # Analyze specific field
  python field_filter_newformat.py data/02-standardised/xlam-function-calling-60k --field original_metadata.category

  # Filter to keep only samples with thought parts
  python field_filter_newformat.py data/02-standardised/dataset \\
    --field conversation_branches[0].messages[0].parts[0].type \\
    --keep-values thought \\
    --output data/03-filtered/dataset-thoughts-only
  
  # Filter by function name in parts
  python field_filter_newformat.py data/02-standardised/dataset \\
    --field conversation_branches[0].messages[0].parts[0].name \\
    --keep-values get_weather_data \\
    --output data/03-filtered/dataset-weather-only
    
  # Split dataset by available function names
  python field_filter_newformat.py data/02-standardised/dataset \\
    --field available_functions[0].name \\
    --split \\
    --output data/03-splits/
        """
    )
    
    parser.add_argument(
        "dataset_path",
        help="Path to input dataset directory"
    )
    
    # Analysis options
    parser.add_argument(
        "--field",
        help="Specific field to analyze (shows schema if not provided)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=50000,
        help="Maximum samples to analyze for schema/field stats (default: 50000, use -1 for all samples)"
    )
    
    # Filtering/splitting options
    parser.add_argument(
        "--keep-values",
        nargs="+",
        help="Values to KEEP (samples with these values will be retained)"
    )
    parser.add_argument(
        "--remove-values", 
        nargs="+",
        help="Values to REMOVE (samples with these values will be excluded)"
    )
    parser.add_argument(
        "--split",
        nargs="?",
        const="",
        help="Split dataset into multiple datasets based on field values. Optional: provide prefix (e.g., --split abc- creates dataset-abc-value)"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output directory for filtered/split dataset(s)"
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_arguments()
    
    # Validate input path
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        print(f"Error: Dataset path does not exist: {dataset_path}")
        sys.exit(1)
    
    # Load dataset
    print(f"Loading dataset from: {dataset_path}")
    try:
        dataset = load_from_disk(str(dataset_path))
        
        # Handle DatasetDict vs single Dataset
        is_dataset_dict = hasattr(dataset, 'keys')
        if is_dataset_dict:
            available_splits = list(dataset.keys())
            print(f"Found DatasetDict with splits: {available_splits}")
        else:
            print(f"Found single Dataset with {len(dataset):,} samples")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)
    
    # Handle -1 for max_samples (means analyze all samples)
    max_samples = None if args.max_samples == -1 else args.max_samples
    
    # Check if filtering or splitting is requested
    is_filtering = args.keep_values or args.remove_values
    is_splitting = args.split is not None
    
    if not is_filtering and not is_splitting:
        # Analysis mode - analyze all splits if DatasetDict
        if is_dataset_dict:
            for split_name in available_splits:
                print(f"\n{'='*80}")
                print(f"ANALYZING SPLIT: {split_name}")
                print(f"{'='*80}")
                split_dataset = dataset[split_name]
                print(f"Split size: {len(split_dataset):,} samples")
                
                analysis = analyze_field_structure(split_dataset, max_samples)
                
                if args.field:
                    # Show specific field statistics
                    display_field_stats(analysis, args.field)
                else:
                    # Show dataset schema
                    display_schema(analysis)
        else:
            # Single dataset
            analysis = analyze_field_structure(dataset, max_samples)
            
            if args.field:
                # Show specific field statistics
                display_field_stats(analysis, args.field)
            else:
                # Show dataset schema
                display_schema(analysis)
        return
    
    # Validate arguments for filtering/splitting
    if not args.field:
        print("Error: Must specify --field for filtering or splitting")
        sys.exit(1)
    
    if args.split is not None and (args.keep_values or args.remove_values):
        print("Error: Cannot use --split with --keep-values or --remove-values")
        sys.exit(1)
    
    if args.keep_values and args.remove_values:
        print("Error: Cannot specify both --keep-values and --remove-values")
        sys.exit(1)
    
    if not args.output:
        print("Error: Must specify --output for filtering or splitting")
        sys.exit(1)
    
    # Load metadata
    original_metadata = {}
    metadata_file = dataset_path / "dataset_metadata.json"
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            original_metadata = json.load(f)
    
    # Handle split mode
    if args.split is not None:
        output_base_path = Path(args.output)
        dataset_name = dataset_path.name
        split_prefix = args.split if args.split else None
        split_dataset_by_field(dataset, args.field, output_base_path, dataset_name, 
                              original_metadata, dataset_path, split_prefix)
        return
    
    # Determine filter parameters
    if args.keep_values:
        target_values = args.keep_values
        keep_matches = True
        action = "keep"
    else:
        target_values = args.remove_values  
        keep_matches = False
        action = "remove"
    
    # Apply filter to all splits if DatasetDict
    if is_dataset_dict:
        filtered_splits = {}
        total_original = 0
        total_filtered = 0
        
        for split_name in available_splits:
            print(f"\nFiltering split: {split_name}")
            split_dataset = dataset[split_name]
            filtered_split = filter_dataset(split_dataset, args.field, target_values, keep_matches)
            
            if len(filtered_split) > 0:
                filtered_splits[split_name] = filtered_split
            
            total_original += len(split_dataset)
            total_filtered += len(filtered_split)
        
        if not filtered_splits:
            print("Warning: All splits resulted in empty datasets after filtering")
        
        # Create filtered DatasetDict
        from datasets import DatasetDict
        filtered_dataset = DatasetDict(filtered_splits)
        
        # Prepare metadata (already loaded above)
        
        filter_metadata = {
            **original_metadata,
            "processing_log": original_metadata.get("processing_log", []) + [{
                "operation": "field_based_filtering",
                "script": "field_filter_newformat.py",
                "timestamp": datetime.now().isoformat(),
                "input_path": str(dataset_path),
                "output_path": str(args.output),
                "filter_field": args.field,
                "filter_values": target_values,
                "filter_action": action,
                "original_samples": total_original,
                "filtered_samples": total_filtered,
                "splits_filtered": list(filtered_splits.keys())
            }]
        }
        
        # Save filtered DatasetDict
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving filtered DatasetDict to {output_path}...")
        filtered_dataset.save_to_disk(str(output_path))
        
        # Save metadata
        metadata_file = output_path / "dataset_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(filter_metadata, f, indent=2)
        
        print(f"DatasetDict saved with {len(filtered_splits)} splits:")
        for split_name, split_data in filtered_splits.items():
            print(f"  - {split_name}: {len(split_data):,} samples")
        print(f"Metadata saved to {metadata_file}")
        
    else:
        # Single dataset filtering
        filtered_dataset = filter_dataset(dataset, args.field, target_values, keep_matches)
        
        # Prepare metadata (already loaded above)
        
        filter_metadata = {
            **original_metadata,
            "processing_log": original_metadata.get("processing_log", []) + [{
                "operation": "field_based_filtering",
                "script": "field_filter_newformat.py",
                "timestamp": datetime.now().isoformat(),
                "input_path": str(dataset_path),
                "output_path": str(args.output),
                "filter_field": args.field,
                "filter_values": target_values,
                "filter_action": action,
                "original_samples": len(dataset),
                "filtered_samples": len(filtered_dataset)
            }]
        }
        
        # Save filtered dataset
        save_filtered_dataset(filtered_dataset, Path(args.output), filter_metadata)


if __name__ == "__main__":
    main()