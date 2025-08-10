#!/usr/bin/env python3
"""
List all benchmark datasets for parallel decontamination processing.

Extracts the benchmark list from gather_decontamination_prompts.py to support
splitting benchmarks across parallel jobs.
"""

import argparse
from datasets import load_from_disk

# Import the benchmark list from gather_decontamination_prompts
from gather_decontamination_prompts import BENCHMARK_DATASETS


def get_benchmark_names():
    """Extract all benchmark names from the BENCHMARK_DATASETS configuration."""
    benchmark_names = []
    
    for dataset_args in BENCHMARK_DATASETS:
        dataset_name = dataset_args["name_or_path"]
        dataset_name_save = dataset_name.replace("/", "__")
        
        # Handle special cases that create multiple benchmark entries
        if dataset_name == "include-base_v2":
            # This creates multiple benchmarks per language
            # We'll add them dynamically by checking the actual saved dataset
            continue
        elif dataset_name == "DAMO-NLP-SG/MultiJail":
            # Creates one benchmark per language column
            for prompt_col_name in dataset_args["prompt_col_name"]:
                benchmark_names.append(dataset_name_save + f"__{prompt_col_name}")
        elif (dataset_args["config_name"] != "iterate") and (
            isinstance(dataset_args["config_name"], str) or dataset_args["config_name"] is None
        ):
            # Single benchmark
            benchmark_names.append(dataset_name_save)
        else:
            # Multiple configs - these create one benchmark per config
            # We'll handle this by checking actual saved benchmarks
            benchmark_names.append(f"{dataset_name_save}__*")  # Placeholder for expansion
    
    return benchmark_names


def get_actual_benchmark_names(decontamination_prompts_path):
    """Get the actual benchmark names from the saved decontamination prompts dataset."""
    try:
        eval_data = load_from_disk(decontamination_prompts_path)
        return list(eval_data.keys())
    except Exception as e:
        print(f"Warning: Could not load decontamination prompts from {decontamination_prompts_path}: {e}")
        print("Falling back to estimated benchmark list")
        return get_benchmark_names()


def main():
    parser = argparse.ArgumentParser(
        description="List all benchmark datasets for parallel decontamination",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all benchmarks (estimated from config)
  python list_benchmarks.py
  
  # List actual benchmarks from saved decontamination prompts
  python list_benchmarks.py --prompts-path /path/to/decontamination_prompts
  
  # Save to file for job splitting
  python list_benchmarks.py --prompts-path /path/to/decontamination_prompts > benchmark_list.txt
  
  # Show count only
  python list_benchmarks.py --prompts-path /path/to/decontamination_prompts --count-only
        """
    )
    
    parser.add_argument(
        "--prompts-path",
        type=str,
        help="Path to saved decontamination prompts dataset (recommended for accuracy)"
    )
    parser.add_argument(
        "--count-only",
        action="store_true",
        help="Only show the count of benchmarks"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=20,
        help="Show how many chunks would be created with given chunk size (default: 20)"
    )
    
    args = parser.parse_args()
    
    if args.prompts_path:
        benchmark_names = get_actual_benchmark_names(args.prompts_path)
        source = f"from {args.prompts_path}"
    else:
        benchmark_names = get_benchmark_names()
        source = "estimated from configuration"
    
    # Filter out placeholders and sort
    actual_benchmarks = [name for name in benchmark_names if not name.endswith("__*")]
    actual_benchmarks.sort()
    
    if args.count_only:
        print(f"{len(actual_benchmarks)} benchmarks {source}")
        if args.chunk_size:
            num_chunks = (len(actual_benchmarks) + args.chunk_size - 1) // args.chunk_size
            print(f"Would create {num_chunks} parallel jobs with chunk size {args.chunk_size}")
    else:
        print(f"# {len(actual_benchmarks)} benchmarks {source}")
        print(f"# Chunk size {args.chunk_size} would create {(len(actual_benchmarks) + args.chunk_size - 1) // args.chunk_size} parallel jobs")
        print()
        for benchmark in actual_benchmarks:
            print(benchmark)


if __name__ == "__main__":
    main()