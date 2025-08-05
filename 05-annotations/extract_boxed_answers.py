#!/usr/bin/env python3
"""
Extract Boxed Answers Script

Extracts \\boxed{answer} patterns from assistant messages in chat format datasets,
replacing them with just the answer text and storing the extracted answers in metadata.

Designed for high-performance processing of large datasets (up to 2M samples) using
HuggingFace Dataset.map() with multiprocessing.
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


def extract_boxed_from_text(text: str) -> Tuple[str, List[str]]:
    """
    Extract all \\boxed{...} patterns from text using proper brace matching.
    
    Args:
        text: Text potentially containing \\boxed{answer} patterns
        
    Returns:
        Tuple of (cleaned_text, list_of_answers)
    """
    if '\\boxed{' not in text:
        return text, []
    
    result = []
    answers = []
    i = 0
    
    while i < len(text):
        # Look for \\boxed{
        if text[i:i+7] == '\\boxed{':
            # Find the matching closing brace
            brace_count = 1
            j = i + 7  # Start after '\\boxed{'
            
            while j < len(text) and brace_count > 0:
                if text[j] == '{':
                    brace_count += 1
                elif text[j] == '}':
                    brace_count -= 1
                j += 1
            
            if brace_count == 0:
                # Found matching closing brace
                # Extract content between braces
                content = text[i+7:j-1]  # j-1 because j is one past the closing }
                answers.append(content)
                result.append(content)
                i = j  # Continue after the closing }
            else:
                # Malformed \\boxed{ without closing brace
                # Keep as is
                result.append(text[i])
                i += 1
        else:
            result.append(text[i])
            i += 1
    
    return ''.join(result), answers


def process_batch(examples: Dict[str, List]) -> Dict[str, List]:
    """
    Process a batch of samples to extract boxed answers.
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
        # Process conversation branches if they exist
        if 'conversation_branches' in processed_examples and processed_examples['conversation_branches'][idx]:
            branches = processed_examples['conversation_branches'][idx]
            
            for branch in branches:
                if 'messages' in branch and branch['messages']:
                    for msg_idx, message in enumerate(branch['messages']):
                        # Only process assistant messages
                        if message.get('role') == 'assistant':
                            content = message.get('content', '')
                            
                            # Extract boxed answers
                            cleaned_content, answers = extract_boxed_from_text(content)
                            
                            # Update message content
                            message['content'] = cleaned_content
                            
                            # Add metadata
                            if 'metadata' not in message:
                                message['metadata'] = {}
                            
                            # Always store as list (even if empty)
                            message['metadata']['verifiable_answer'] = answers
    
    return processed_examples


def process_dataset(dataset: Dataset, num_proc: int = None, 
                   batch_size: int = 10000) -> Tuple[Dataset, Dict[str, Any]]:
    """
    Process dataset to extract boxed answers using map().
    
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
    print("Analyzing dataset for boxed patterns...")
    initial_stats = analyze_dataset_for_boxed(dataset)
    
    # Process with map
    processed_dataset = dataset.map(
        process_batch,
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
        load_from_cache_file=False,
        desc="Extracting boxed answers"
    )
    
    # Count final statistics
    print("Computing extraction statistics...")
    final_stats = compute_extraction_stats(processed_dataset)
    
    statistics = {
        'initial_messages_with_boxed': initial_stats['messages_with_boxed'],
        'initial_total_boxed': initial_stats['total_boxed'],
        'samples_processed': len(processed_dataset),
        'samples_with_answers': final_stats['samples_with_answers'],
        'messages_with_answers': final_stats['messages_with_answers'],
        'total_answers_extracted': final_stats['total_answers'],
        'messages_with_multiple': final_stats['messages_with_multiple']
    }
    
    return processed_dataset, statistics


def analyze_dataset_for_boxed(dataset: Dataset) -> Dict[str, int]:
    """
    Quick analysis to count \\boxed{} patterns in dataset.
    
    Args:
        dataset: Input dataset
        
    Returns:
        Dictionary with counts
    """
    messages_with_boxed = 0
    total_boxed = 0
    
    # Sample first 1000 for quick estimate if dataset is large
    sample_size = min(1000, len(dataset))
    
    for i in range(sample_size):
        sample = dataset[i]
        if 'conversation_branches' in sample:
            for branch in sample['conversation_branches']:
                if 'messages' in branch:
                    for message in branch['messages']:
                        if message.get('role') == 'assistant':
                            content = message.get('content', '')
                            count = content.count('\\boxed{')
                            if count > 0:
                                messages_with_boxed += 1
                                total_boxed += count
    
    # Extrapolate if sampled
    if sample_size < len(dataset):
        factor = len(dataset) / sample_size
        messages_with_boxed = int(messages_with_boxed * factor)
        total_boxed = int(total_boxed * factor)
        print(f"(Estimated from {sample_size:,} sample)")
    
    return {
        'messages_with_boxed': messages_with_boxed,
        'total_boxed': total_boxed
    }


def compute_extraction_stats(dataset: Dataset) -> Dict[str, int]:
    """
    Compute statistics on extracted answers.
    
    Args:
        dataset: Processed dataset
        
    Returns:
        Dictionary with statistics
    """
    samples_with_answers = 0
    messages_with_answers = 0
    total_answers = 0
    messages_with_multiple = 0
    
    # Sample for statistics if dataset is large
    sample_size = min(1000, len(dataset))
    
    for i in range(sample_size):
        sample = dataset[i]
        sample_has_answer = False
        
        if 'conversation_branches' in sample:
            for branch in sample['conversation_branches']:
                if 'messages' in branch:
                    for message in branch['messages']:
                        if message.get('role') == 'assistant':
                            answers = message.get('metadata', {}).get('verifiable_answer', [])
                            if answers:
                                messages_with_answers += 1
                                total_answers += len(answers)
                                sample_has_answer = True
                                if len(answers) > 1:
                                    messages_with_multiple += 1
        
        if sample_has_answer:
            samples_with_answers += 1
    
    # Extrapolate if sampled
    if sample_size < len(dataset):
        factor = len(dataset) / sample_size
        samples_with_answers = int(samples_with_answers * factor)
        messages_with_answers = int(messages_with_answers * factor)
        total_answers = int(total_answers * factor)
        messages_with_multiple = int(messages_with_multiple * factor)
    
    return {
        'samples_with_answers': samples_with_answers,
        'messages_with_answers': messages_with_answers,
        'total_answers': total_answers,
        'messages_with_multiple': messages_with_multiple
    }


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
        "operation": "boxed_answer_extraction",
        "script": "extract_boxed_answers.py",
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
        description="Extract \\boxed{answer} patterns from assistant messages",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-generate output name with -boxedRemoved suffix
  venv/bin/python 05-annotations/extract_boxed_answers.py \\
    /capstor/store/cscs/swissai/infra01/posttrain_data/04_decontaminated/tulu-3-sft-mixture-ai2-adapt-dev_tulu_v3_9_open_math_2_gsm8k_50k \\
    --output data/05-annotations/
  # Creates: data/05-annotations/tulu-3-sft-mixture-ai2-adapt-dev_tulu_v3_9_open_math_2_gsm8k_50k-boxedRemoved

  # Specify custom output name
  venv/bin/python 05-annotations/extract_boxed_answers.py \\
    data/02-standardised/large-math-dataset \\
    --output data/05-annotations/my-custom-name \\
    --batch-size 50000 \\
    --num-proc 16

  # Single process (for debugging)
  venv/bin/python 05-annotations/extract_boxed_answers.py \\
    data/02-standardised/math-dataset \\
    --output data/05-annotations/math-extracted \\
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
        help="Path to output directory (if ends with /, auto-generates name with -boxedRemoved suffix)"
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
        # Auto-generate name with -boxedRemoved suffix
        dataset_name = input_path.name
        output_path = Path(output_arg) / f"{dataset_name}-boxedRemoved"
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
            total_extracted = sum(stats['total_answers_extracted'] 
                                 for stats in all_statistics.values())
            
            combined_stats = {
                'total_samples_all_splits': total_samples,
                'total_answers_extracted_all_splits': total_extracted,
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
    print(f"\n=== Extraction Complete ===")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    
    if isinstance(dataset, DatasetDict):
        print(f"Splits processed: {', '.join(available_splits)}")
        for split_name in available_splits:
            stats = all_statistics[f"{split_name}_stats"]
            print(f"\n{split_name} split:")
            print(f"  Samples: {stats['samples_processed']:,}")
            print(f"  Answers extracted: {stats['total_answers_extracted']:,}")
            print(f"  Messages with multiple answers: {stats['messages_with_multiple']:,}")
    else:
        print(f"Samples processed: {combined_stats['samples_processed']:,}")
        print(f"Total answers extracted: {combined_stats['total_answers_extracted']:,}")
        print(f"Messages with multiple answers: {combined_stats['messages_with_multiple']:,}")


if __name__ == "__main__":
    main()