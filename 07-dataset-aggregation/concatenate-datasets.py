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


from datasets import Features as ds_Features, Value as ds_Value, List as ds_List

FULL_FEATURES = ds_Features({
    "conversation_id": ds_Value("string"),
    "dataset_source": ds_Value("string"),
    "original_metadata": ds_Value("string"),
    "created_timestamp": ds_Value("string"),
    "system_prompt": {
        "content": ds_Value("string"), 
        "metadata": ds_Value("string")
    },
    "initial_prompt": {
        "content": ds_Value("string"), 
        "metadata": ds_Value("string"), 
        "role": ds_Value("string")
    },
    # The fix for the first error: Define the struct for functions
    "available_functions": ds_List({
        "description": ds_Value("string"),
        "name": ds_Value("string"),
        "parameters": ds_Value("string")
    }),
    # The fix for the nested branch errors:
    "conversation_branches": ds_List({
        "messages": ds_List({
            "parts": ds_List({
                "answers": ds_List(ds_Value("string")), # Force string list, not null
                "args": ds_Value("string"),
                "content": ds_Value("string"),
                "metadata": ds_Value("string"),
                "name": ds_Value("string"),
                "type": ds_Value("string")
            }),
            "role": ds_Value("string")
        })
    })
})

# Official schema columns from the standardization format
OFFICIAL_COLUMNS = {
    "conversation_id",
    "dataset_source", 
    "original_metadata",
    "system_prompt",
    "initial_prompt",
    "available_functions",
    "conversation_branches",
    "created_timestamp",
}

# Nested schema definitions - keys allowed at each level
SYSTEM_PROMPT_KEYS = {"content", "metadata"}
INITIAL_PROMPT_KEYS = {"role", "content", "metadata"}
MESSAGE_KEYS = {"role", "parts"}
PART_KEYS = {"type", "content", "metadata", "name", "args", "answers"}
FUNCTION_KEYS = {"name", "description", "parameters"}


def serialize_metadata(value: Any) -> str:
    """Serialize any metadata value to a JSON string."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def clean_dict(d: Dict[str, Any], allowed_keys: set) -> Dict[str, Any]:
    """Keep only allowed keys from a dictionary."""
    if d is None:
        return {}
    if not isinstance(d, dict):
        return {}
    return {k: v for k, v in d.items() if k in allowed_keys}


def normalize_part(part: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a message part to the official schema."""
    if not isinstance(part, dict):
        return {"type": "response", "content": str(part) if part else "", "metadata": "", "name": "", "args": ""}
    
    cleaned = clean_dict(part, PART_KEYS)
    
    # Ensure all standard keys exist with proper types
    result = {
        "type": cleaned.get("type", "response"),
        "content": cleaned.get("content", "") if cleaned.get("content") is not None else "",
        "metadata": serialize_metadata(cleaned.get("metadata")),
        "name": cleaned.get("name", "") if cleaned.get("name") is not None else "",
        "args": serialize_metadata(cleaned.get("args")),  # args can be dict, serialize it
    }
    
    # Handle answers for verifiable-responses (keep as-is if it"s a list)
    if "answers" in cleaned:
        result["answers"] = cleaned["answers"]
    # else:
    #     result["answers"] = [""]
    
    return result


def normalize_message(message: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a conversation message to the official schema."""
    if not isinstance(message, dict):
        return {"role": "user", "parts": []}
    
    cleaned = clean_dict(message, MESSAGE_KEYS)
    
    # Normalize parts
    parts = cleaned.get("parts", [])
    if not isinstance(parts, list):
        parts = []
    
    return {
        "role": cleaned.get("role", "user"),
        "parts": [normalize_part(p) for p in parts]
    }


def normalize_branch(branch: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a conversation branch to the official schema."""
    if not isinstance(branch, dict):
        return {"messages": []}
    
    messages = branch.get("messages", [])
    if not isinstance(messages, list):
        messages = []
    
    return {
        "messages": [normalize_message(m) for m in messages]
    }


def normalize_function(func: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a function definition to the official schema."""
    if not isinstance(func, dict):
        return {"name": "", "description": "", "parameters": ""}
    
    cleaned = clean_dict(func, FUNCTION_KEYS)
    
    return {
        "name": cleaned.get("name", ""),
        "description": cleaned.get("description", ""),
        "parameters": serialize_metadata(cleaned.get("parameters", {}))
    }


def normalize_system_prompt(sp: Any) -> Dict[str, Any]:
    """Normalize system_prompt to the official schema."""
    if sp is None:
        return {"content": "", "metadata": ""}
    if not isinstance(sp, dict):
        return {"content": str(sp), "metadata": ""}
    
    cleaned = clean_dict(sp, SYSTEM_PROMPT_KEYS)
    
    return {
        "content": cleaned.get("content", "") if cleaned.get("content") is not None else "",
        "metadata": serialize_metadata(cleaned.get("metadata"))
    }


def normalize_initial_prompt(ip: Any) -> Dict[str, Any]:
    """Normalize initial_prompt to the official schema."""
    if ip is None:
        return {"role": "user", "content": "", "metadata": ""}
    if not isinstance(ip, dict):
        return {"role": "user", "content": str(ip), "metadata": ""}
    
    cleaned = clean_dict(ip, INITIAL_PROMPT_KEYS)
    
    return {
        "role": cleaned.get("role", "user"),
        "content": cleaned.get("content", "") if cleaned.get("content") is not None else "",
        "metadata": serialize_metadata(cleaned.get("metadata"))
    }


def normalize_example(example: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a single example to the official schema, removing all extra keys."""
    result = {}
    
    # Top-level string fields
    result["conversation_id"] = example.get("conversation_id", "")
    result["dataset_source"] = example.get("dataset_source", "")
    result["created_timestamp"] = example.get("created_timestamp", "")
    
    # Serialize original_metadata to string
    result["original_metadata"] = serialize_metadata(example.get("original_metadata"))
    
    # Normalize nested structures
    result["system_prompt"] = normalize_system_prompt(example.get("system_prompt"))
    result["initial_prompt"] = normalize_initial_prompt(example.get("initial_prompt"))
    
    # Normalize available_functions
    funcs = example.get("available_functions", [])
    if not isinstance(funcs, list):
        funcs = []
    result["available_functions"] = [normalize_function(f) for f in funcs]
    
    # Normalize conversation_branches
    branches = example.get("conversation_branches", [])
    if not isinstance(branches, list):
        branches = []
    result["conversation_branches"] = [normalize_branch(b) for b in branches]
    
    return result


def normalize_dataset_schema(dataset: Dataset, dataset_name: str = "unknown") -> Dataset:
    """
    Normalize a dataset to the official schema.
    
    - Removes all extra columns at the top level
    - Removes extra keys from nested structures (system_prompt, initial_prompt, messages, parts)
    - Serializes all metadata dicts to JSON strings
    
    Args:
        dataset: The dataset to normalize
        dataset_name: Name of the dataset (for logging)
        
    Returns:
        Dataset with strict official schema
    """
    current_columns = set(dataset.column_names)
    extra_columns = current_columns - OFFICIAL_COLUMNS
    missing_columns = OFFICIAL_COLUMNS - current_columns
    
    print(f"  Current columns: {sorted(current_columns)}")
    
    if extra_columns:
        print(f"  Extra columns to remove: {sorted(extra_columns)}")
    if missing_columns:
        print(f"  Missing columns (will be added with defaults): {sorted(missing_columns)}")
    
    # Apply deep normalization
    print(f"  Normalizing schema (deep clean + metadata serialization)...")
    dataset = dataset.map(normalize_example, desc=f"  Normalizing", num_proc=1)
    
    # Select only official columns (in case map added extras somehow)
    columns_to_keep = [col for col in dataset.column_names if col in OFFICIAL_COLUMNS]
    dataset = dataset.select_columns(columns_to_keep)
    
    print(f"  Normalized to {len(columns_to_keep)} columns: {sorted(columns_to_keep)}")
    
    return dataset


def load_dataset_safely(path: str, normalize: bool = True) -> Dataset:
    """Load a dataset and ensure it"s a single Dataset (not DatasetDict).
    
    Args:
        path: Path to the dataset
        normalize: If True, normalize schema by moving extra columns to original_metadata
        
    Returns:
        Loaded and optionally normalized Dataset
    """
    data = load_from_disk(path)
    dataset_name = Path(path).name
    
    if isinstance(data, DatasetDict):
        # If it"s a DatasetDict, concatenate all splits
        print(f"  Dataset is a DatasetDict with splits: {list(data.keys())}")
        all_splits = []
        for split_name, split_data in data.items():
            print(f"    {split_name}: {len(split_data)} samples")
            all_splits.append(split_data)
        dataset = concatenate_datasets(all_splits)
    else:
        # Already a single Dataset
        dataset = data
    
    # Normalize schema if requested
    if normalize:
        dataset = normalize_dataset_schema(dataset, dataset_name)
    
    return dataset

def load_existing_metadata(output_path: Path) -> Optional[Dict[str, Any]]:
    """Load existing dataset metadata if it exists."""
    meta_file = output_path / "dataset_metadata.json"
    if meta_file.exists():
        try:
            with open(meta_file, "r") as f:
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
        "schema_normalized": not args.no_normalize,
        "strict_mode": args.strict,
        "shuffle_flags": args.shuffle,
        "official_columns": sorted(OFFICIAL_COLUMNS),
        "input_datasets": input_stats,
        "output_statistics": output_stats,
        "description": f"Concatenated {len(input_paths)} datasets" + (" with schema normalization" if not args.no_normalize else "")
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
            "schema_normalized": not args.no_normalize,
            "schema_validation": "Extra columns moved to original_metadata._extra_columns" if not args.no_normalize else "Assumes identical schemas"
        }
    
    # Save metadata
    metadata_file = output_path / "dataset_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nDataset saved to {output_path}")
    print(f"Metadata saved to {metadata_file}")
    
    # Print summary statistics
    print("\nConcatenation Summary:")
    print(f"  Total samples: {output_stats["total_samples"]:,}")
    print(f"  Dataset sources found:")
    for source, count in sorted(output_stats["dataset_sources"].items()):
        print(f"    {source}: ~{count:,}")
    if output_stats.get("has_system_prompts"):
        print(f"  Samples with system prompts: ~{output_stats["has_system_prompts"]:,}")
    if output_stats.get("has_available_functions"):
        print(f"  Samples with available functions: ~{output_stats["has_available_functions"]:,}")

# ───────────— CLI / main ───────────── #
def cli():
    p = argparse.ArgumentParser(
        description="Concatenate multiple harmonized datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Concatenate two datasets (default: normalizes schemas)
  %(prog)s /path/to/dataset1 /path/to/dataset2 -o /path/to/output

  # Concatenate with more processes
  %(prog)s dataset1 dataset2 dataset3 -o output --num-proc 16

  # Save as DatasetDict with "train" split
  %(prog)s dataset1 dataset2 -o output --as-datasetdict

  # Skip schema normalization (requires identical schemas)
  %(prog)s dataset1 dataset2 -o output --no-normalize

  # Strict mode: fail if schemas don"t match exactly
  %(prog)s dataset1 dataset2 -o output --strict

  # Shuffle specific datasets before sampling
  %(prog)s dataset1 dataset2 dataset3 dataset4 -o output --shuffle "true,false,false,true"

Schema Normalization:
  By default, extra columns beyond the official schema are moved into
  the "original_metadata" field under "_extra_columns". This allows
  concatenating datasets with slightly different schemas.
  
  Official columns: conversation_id, dataset_source, original_metadata,
                   system_prompt, initial_prompt, available_functions,
                   conversation_branches, created_timestamp
        """
    )
    p.add_argument("datasets", nargs="+", help="Paths to datasets to concatenate")
    p.add_argument("-o", "--output", required=True, help="Output directory path")
    p.add_argument("--num-proc", type=int, default=8, help="Number of processes for dataset operations")
    p.add_argument("--sample-range", type=str, default="all",
                   help="Range of samples for datasets. Can be single (all) or comma-separated list matching inputs (e.g. 'all,0:100,200:300'). Use 'all' instead of '-1' to avoid CLI parsing issues.")
    p.add_argument("--shuffle", type=str, default=None,
                   help="Comma-separated true/false values for each dataset (e.g. 'true,false,false,true'). If true, shuffle the dataset before sampling.")
    p.add_argument("--as-datasetdict", action="store_true",
                   help="Save as DatasetDict with 'train' split (default: save as Dataset)")
    p.add_argument("--no-normalize", action="store_true",
                   help="Skip schema normalization (requires identical schemas)")
    p.add_argument("--strict", action="store_true",
                   help="Fail if schemas don't match exactly (after normalization)")
    return p.parse_args()


def apply_range(dataset: Dataset, range_str: str) -> Dataset:
    """Apply a sample range to a dataset."""
    if range_str in ["-1", "all"]:
        return dataset
    
    total = len(dataset)
    try:
        if ":" in range_str:
            parts = range_str.split(":")
            start = int(parts[0]) if parts[0] else 0
            end = int(parts[1]) if parts[1] else total
            
            # Clamp and handle negative indices if necessary (Python-like)
            if start < 0: start = max(0, total + start)
            if end < 0: end = max(0, total + end)
            
            start = min(total, start)
            end = min(total, end)
            
            if start >= end:
                return dataset.select([])
            return dataset.select(range(start, end))
        else:
            # Single number: take first N samples
            val = int(range_str)
            if val == -1:
                return dataset
            if val >= 0:
                return dataset.select(range(0, min(total, val)))
            else:
                return dataset
    except (ValueError, IndexError):
        print(f"  Warning: Invalid range format '{range_str}'. Using all samples.")
        return dataset


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
    
    # Parse sample ranges
    sample_ranges = args.sample_range.split(',')
    if len(sample_ranges) == 1:
        sample_ranges = sample_ranges * len(input_paths)
    
    if len(sample_ranges) != len(input_paths):
        print(f"Error: Number of sample ranges ({len(sample_ranges)}) provided via --sample-range does not match number of input datasets ({len(input_paths)}).")
        print("Please provide either a single range (applies to all) or one range per dataset (comma-separated).")
        sys.exit(1)

    # Parse shuffle flags
    if args.shuffle:
        shuffle_flags = [s.strip().lower() == 'true' for s in args.shuffle.split(',')]
        if len(shuffle_flags) == 1:
            shuffle_flags = shuffle_flags * len(input_paths)
        if len(shuffle_flags) != len(input_paths):
            print(f"Error: Number of shuffle flags ({len(shuffle_flags)}) provided via --shuffle does not match number of input datasets ({len(input_paths)}).")
            print("Please provide either a single value (applies to all) or one value per dataset (comma-separated).")
            sys.exit(1)
    else:
        shuffle_flags = [False] * len(input_paths)

    print(f"Will concatenate {len(input_paths)} datasets:")
    for i, path in enumerate(input_paths):
        range_str = sample_ranges[i]
        shuffle_str = shuffle_flags[i]
        range_info = f" (range: {range_str})" if range_str not in ["-1", "all"] else ""
        shuffle_info = " (shuffle)" if shuffle_str else ""
        print(f"  - {path}{range_info}{shuffle_info}")
    
    # Check if output exists
    if output_path.exists():
        response = input(f"\n{output_path} exists. Overwrite? [y/N]: ")
        if response.lower() != "y":
            sys.exit(0)
    
    # Load all datasets
    normalize = not args.no_normalize
    if normalize:
        print(f"\nSchema normalization enabled (extra columns → original_metadata)")
        print(f"Official columns: {sorted(OFFICIAL_COLUMNS)}")
    
    print("\nLoading datasets...")
    datasets_to_concat = []
    input_stats = []
    
    for i, path in enumerate(input_paths, 1):
        print(f"\n[{i}/{len(input_paths)}] Loading {Path(path).name}")
        try:
            dataset = load_dataset_safely(path, normalize=normalize)
            original_len = len(dataset)

            # Apply shuffle if specified (before range selection)
            should_shuffle = shuffle_flags[i-1]
            if should_shuffle:
                dataset = dataset.shuffle()
                print(f"  Shuffled dataset")

            # Apply range if specified
            current_range = sample_ranges[i-1]
            if current_range not in ["-1", "all"]:
                dataset = apply_range(dataset, current_range)
                print(f"  Selected {len(dataset):,} samples from {original_len:,} (range: {current_range})")
            else:
                print(f"  Loaded {len(dataset):,} samples")
            
            # Gather basic statistics
            stats = {
                "path": path,
                "name": Path(path).name,
                "num_samples": len(dataset),
                "original_num_samples": original_len,
                "range_applied": current_range if current_range not in ["-1", "all"] else None,
                "shuffled": should_shuffle,
                "columns": dataset.column_names,
                "normalized": normalize
            }
            
            # Check first sample for dataset source
            if len(dataset) > 0:
                first_sample = dataset[0]
                stats["dataset_source"] = first_sample.get("dataset_source", "unknown")

            input_stats.append(stats)
            datasets_to_concat.append(dataset.cast(FULL_FEATURES))
            
        except Exception as e:
            print(f"  Error loading dataset: {e}")
            sys.exit(1)
    
    # Verify schemas are compatible
    print("\nVerifying schema compatibility...")
    reference_columns = set(datasets_to_concat[0].column_names)
    schema_mismatch = False
    for i, dataset in enumerate(datasets_to_concat[1:], 2):
        current_columns = set(dataset.column_names)
        if current_columns != reference_columns:
            schema_mismatch = True
            print(f"Warning: Dataset {i} has different columns!")
            print(f"  Reference: {sorted(reference_columns)}")
            print(f"  Current:   {sorted(current_columns)}")
            print(f"  Missing:   {reference_columns - current_columns}")
            print(f"  Extra:     {current_columns - reference_columns}")
    
    if schema_mismatch:
        if args.strict:
            print("\nError: Schema mismatch in --strict mode. Exiting.")
            sys.exit(1)
        response = input("\nSchema mismatch detected. Continue anyway? [y/N]: ")
        if response.lower() != "y":
            sys.exit(1)
    else:
        print("  All datasets have compatible schemas")
    
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
