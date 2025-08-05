#!/usr/bin/env python3
"""
Fix System Prompts Script

Moves system prompts from original_metadata.chat_template_kwargs.custom_instructions
to the top-level system_prompt field as required by the standardized chat format schema.

Designed for high-performance processing of large datasets using HuggingFace Dataset.map()
with multiprocessing.
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from multiprocessing import cpu_count

from datasets import load_from_disk, Dataset, DatasetDict
from tqdm import tqdm


def process_batch(examples: Dict[str, List]) -> Dict[str, List]:
    """
    Process a batch of samples to fix system prompts.
    This is the main function called by Dataset.map() in batched mode.
    
    Args:
        examples: Batch of examples with list of values for each field
        
    Returns:
        Processed batch with same structure but new data (no in-place modification)
    """
    import copy
    
    batch_size = len(examples['conversation_id'])
    
    # Create a deep copy to avoid in-place modification
    processed_examples = copy.deepcopy(examples)
    
    # Process each sample in the batch
    for idx in range(batch_size):
        # Check if system_prompt already exists
        existing_system_prompt = None
        if 'system_prompt' in processed_examples and processed_examples['system_prompt'][idx]:
            existing_system_prompt = processed_examples['system_prompt'][idx]
        
        # Try to extract custom_instructions from metadata
        custom_instructions = None
        if ('original_metadata' in processed_examples and 
            processed_examples['original_metadata'][idx] and
            isinstance(processed_examples['original_metadata'][idx], dict)):
            
            metadata = processed_examples['original_metadata'][idx]
            if ('chat_template_kwargs' in metadata and 
                isinstance(metadata['chat_template_kwargs'], dict) and
                'custom_instructions' in metadata['chat_template_kwargs']):
                
                custom_instructions = metadata['chat_template_kwargs']['custom_instructions']
        
        # Handle the system prompt
        if custom_instructions:
            # Create or update system_prompt
            new_system_prompt = {
                'content': custom_instructions,
                'metadata': {}
            }
            
            # If there was an existing system prompt, preserve its metadata
            if existing_system_prompt and isinstance(existing_system_prompt, dict):
                if 'metadata' in existing_system_prompt:
                    new_system_prompt['metadata'] = existing_system_prompt['metadata']
            
            processed_examples['system_prompt'][idx] = new_system_prompt
        else:
            # No custom_instructions found - keep existing system_prompt or None
            if not existing_system_prompt:
                processed_examples['system_prompt'][idx] = None
    
    return processed_examples


def analyze_dataset_for_system_prompts(dataset: Dataset) -> Dict[str, int]:
    """
    Analyze dataset to count system prompt issues.
    
    Args:
        dataset: Input dataset
        
    Returns:
        Dictionary with counts
    """
    samples_with_custom_instructions = 0
    samples_with_existing_system_prompt = 0
    samples_missing_custom_instructions = 0
    samples_with_both = 0
    
    # Sample first 1000 for quick estimate if dataset is large
    sample_size = min(1000, len(dataset))
    
    for i in range(sample_size):
        sample = dataset[i]
        
        has_system_prompt = 'system_prompt' in sample and sample['system_prompt']
        has_custom_instructions = False
        
        if 'original_metadata' in sample and isinstance(sample['original_metadata'], dict):
            metadata = sample['original_metadata']
            if ('chat_template_kwargs' in metadata and 
                isinstance(metadata['chat_template_kwargs'], dict) and
                'custom_instructions' in metadata['chat_template_kwargs']):
                has_custom_instructions = True
        
        if has_custom_instructions:
            samples_with_custom_instructions += 1
        else:
            samples_missing_custom_instructions += 1
            
        if has_system_prompt:
            samples_with_existing_system_prompt += 1
            
        if has_custom_instructions and has_system_prompt:
            samples_with_both += 1
    
    # Extrapolate if sampled
    if sample_size < len(dataset):
        factor = len(dataset) / sample_size
        samples_with_custom_instructions = int(samples_with_custom_instructions * factor)
        samples_with_existing_system_prompt = int(samples_with_existing_system_prompt * factor)
        samples_missing_custom_instructions = int(samples_missing_custom_instructions * factor)
        samples_with_both = int(samples_with_both * factor)
        print(f"(Estimated from {sample_size:,} sample)")
    
    return {
        'samples_with_custom_instructions': samples_with_custom_instructions,
        'samples_with_existing_system_prompt': samples_with_existing_system_prompt,
        'samples_missing_custom_instructions': samples_missing_custom_instructions,
        'samples_with_both': samples_with_both
    }


def process_dataset(dataset: Dataset, num_proc: int = None, 
                   batch_size: int = 10000) -> Tuple[Dataset, Dict[str, Any]]:
    """
    Process dataset to fix system prompts using map().
    
    Args:
        dataset: Input HuggingFace Dataset
        num_proc: Number of processes for parallel processing
        batch_size: Size of batches for processing
        
    Returns:
        Tuple of (processed_dataset, statistics)
    """
    if num_proc is None:
        num_proc = 16
    
    print(f"Processing with {num_proc} processes, batch size {batch_size:,}")
    
    # Count initial statistics
    print("Analyzing dataset for system prompt issues...")
    initial_stats = analyze_dataset_for_system_prompts(dataset)
    
    # Add system_prompt column if it doesn't exist
    if 'system_prompt' not in dataset.column_names:
        print("Adding system_prompt column...")
        dataset = dataset.add_column('system_prompt', [None] * len(dataset))
    
    # Process with map
    processed_dataset = dataset.map(
        process_batch,
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
        load_from_cache_file=False,
        desc="Fixing system prompts"
    )
    
    # Count final statistics
    print("Computing fix statistics...")
    final_stats = analyze_dataset_for_system_prompts(processed_dataset)
    
    statistics = {
        'initial_samples_with_custom_instructions': initial_stats['samples_with_custom_instructions'],
        'initial_samples_with_existing_system_prompt': initial_stats['samples_with_existing_system_prompt'],
        'initial_samples_missing_custom_instructions': initial_stats['samples_missing_custom_instructions'],
        'samples_processed': len(processed_dataset),
        'samples_fixed': initial_stats['samples_with_custom_instructions'],
        'samples_with_warnings': initial_stats['samples_missing_custom_instructions'],
        'final_samples_with_system_prompt': final_stats['samples_with_existing_system_prompt']
    }
    
    # Print warnings summary
    if initial_stats['samples_missing_custom_instructions'] > 0:
        print(f"\n⚠️  WARNING: {initial_stats['samples_missing_custom_instructions']:,} samples missing custom_instructions in metadata")
    
    return processed_dataset, statistics


def load_existing_metadata(input_path: Path) -> Optional[Dict[str, Any]]:
    """Load existing dataset metadata if it exists."""
    metadata_file = Path(input_path) / "dataset_metadata.json"
    if metadata_file.exists():
        try:
            with open(metadata_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return None


def save_dataset_and_metadata(dataset_dict: DatasetDict, output_path: Path, 
                             input_path: Path, statistics: Dict[str, Any], 
                             args: argparse.Namespace):
    """Save processed dataset with metadata."""
    # Ensure output directory exists
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save dataset
    print(f"Saving processed dataset to {output_path}...")
    dataset_dict.save_to_disk(str(output_path))
    
    # Load or create metadata
    metadata = load_existing_metadata(input_path) or {}
    
    # Add processing log entry
    processing_entry = {
        "operation": "system_prompt_fix",
        "script": "fix_system_prompts.py",
        "timestamp": datetime.now().isoformat(),
        "input_path": str(input_path),
        "output_path": str(output_path),
        "batch_size": args.batch_size,
        "num_processes": args.num_proc or "auto",
        **statistics
    }
    
    if "processing_log" not in metadata:
        metadata["processing_log"] = []
    metadata["processing_log"].append(processing_entry)
    
    # Save updated metadata
    metadata_file = output_path / "dataset_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {metadata_file}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fix system prompts by moving custom_instructions to proper schema location",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-generate output name with -systemPromptFix suffix
  venv/bin/python 05-annotations/fix_system_prompts.py \\
    /capstor/store/cscs/swissai/infra01/posttrain_data/04_decontaminated/smoltalk2-smoltalk_smollm3_systemchats_30k_no_think \\
    --output data/05-annotations/
  # Creates: data/05-annotations/smoltalk2-smoltalk_smollm3_systemchats_30k_no_think-systemPromptFix

  # Specify custom output name
  venv/bin/python 05-annotations/fix_system_prompts.py \\
    data/02-standardised/dataset-with-bad-prompts \\
    --output data/05-annotations/my-custom-name \\
    --batch-size 50000 \\
    --num-proc 16

  # Single process (for debugging)
  venv/bin/python 05-annotations/fix_system_prompts.py \\
    data/02-standardised/test-dataset \\
    --output data/05-annotations/test-fixed \\
    --num-proc 1
        """
    )
    
    parser.add_argument(
        "input_path",
        help="Path to input dataset directory"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Path to output directory (if ends with /, auto-generates name with -systemPromptFix suffix)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10000,
        help="Number of samples per batch for map() (default: 10000)"
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=None,
        help="Number of processes for parallel processing (default: auto)"
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_arguments()
    
    # Validate input path
    input_path = Path(args.input_path)
    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}")
        sys.exit(1)
    
    # Determine output path
    output_arg = args.output
    if output_arg.endswith('/'):
        # Auto-generate name with -systemPromptFix suffix
        dataset_name = input_path.name
        output_path = Path(output_arg) / f"{dataset_name}-systemPromptFix"
        print(f"Auto-generated output path: {output_path}")
    else:
        # Use provided name
        output_path = Path(output_arg)
    
    # Check if output exists
    if output_path.exists():
        print(f"Warning: Output directory already exists: {output_path}")
        response = input("Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Aborted")
            sys.exit(0)
    
    # Load dataset
    print(f"Loading dataset from: {input_path}")
    try:
        dataset = load_from_disk(str(input_path))
        
        # Handle DatasetDict vs single Dataset
        if isinstance(dataset, DatasetDict):
            available_splits = list(dataset.keys())
            print(f"Found DatasetDict with splits: {available_splits}")
            
            # Process each split
            processed_splits = {}
            all_statistics = {}
            
            for split_name in available_splits:
                print(f"\n=== Processing split: {split_name} ===")
                split_dataset = dataset[split_name]
                print(f"Split size: {len(split_dataset):,} samples")
                
                # Process this split
                processed_split, split_stats = process_dataset(
                    split_dataset,
                    num_proc=args.num_proc,
                    batch_size=args.batch_size
                )
                
                processed_splits[split_name] = processed_split
                all_statistics[f"{split_name}_stats"] = split_stats
            
            # Create output DatasetDict
            output_dataset = DatasetDict(processed_splits)
            
            # Combine statistics
            total_samples = sum(len(dataset[split]) for split in available_splits)
            total_fixed = sum(stats['samples_fixed'] 
                             for stats in all_statistics.values())
            total_warnings = sum(stats['samples_with_warnings']
                               for stats in all_statistics.values())
            
            combined_stats = {
                'total_samples_all_splits': total_samples,
                'total_samples_fixed_all_splits': total_fixed,
                'total_warnings_all_splits': total_warnings,
                'splits_processed': available_splits,
                **all_statistics
            }
        else:
            # Single Dataset
            print(f"Dataset size: {len(dataset):,} samples")
            
            # Process dataset
            processed_dataset, statistics = process_dataset(
                dataset,
                num_proc=args.num_proc,
                batch_size=args.batch_size
            )
            
            # Wrap in DatasetDict for consistency
            output_dataset = DatasetDict({"train": processed_dataset})
            combined_stats = statistics
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)
    
    # Save final dataset and metadata
    save_dataset_and_metadata(output_dataset, output_path, input_path, 
                             combined_stats, args)
    
    # Display results
    print(f"\n=== System Prompt Fix Complete ===")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    
    if isinstance(dataset, DatasetDict):
        print(f"Splits processed: {', '.join(available_splits)}")
        for split_name in available_splits:
            stats = all_statistics[f"{split_name}_stats"]
            print(f"\n{split_name} split:")
            print(f"  Samples: {stats['samples_processed']:,}")
            print(f"  System prompts fixed: {stats['samples_fixed']:,}")
            print(f"  Warnings (missing custom_instructions): {stats['samples_with_warnings']:,}")
    else:
        print(f"Samples processed: {combined_stats['samples_processed']:,}")
        print(f"System prompts fixed: {combined_stats['samples_fixed']:,}")
        print(f"Warnings (missing custom_instructions): {combined_stats['samples_with_warnings']:,}")


if __name__ == "__main__":
    main()