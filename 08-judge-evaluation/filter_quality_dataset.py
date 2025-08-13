#!/usr/bin/env python3
"""
Filter dataset to keep only high-quality samples with specific model responses.

Filters based on:
- Initial prompt complexity x quality score >= 20
- Keep only conversational branches from "Qwen/Qwen3-32B" model
- Collect up to 500 samples
"""

import argparse
from pathlib import Path
from datasets import load_from_disk, Dataset, DatasetDict
from tqdm import tqdm
import json


def filter_dataset(dataset_path, output_path, target_model="Qwen/Qwen3-32B", min_score=20, max_samples=500, max_completion_length=None):
    """Filter dataset based on quality score and model."""
    print(f"Loading dataset from: {dataset_path}")
    dataset = load_from_disk(dataset_path)
    
    # Handle DatasetDict or single Dataset
    if hasattr(dataset, 'keys'):
        print(f"Dataset splits: {list(dataset.keys())}")
        split_name = list(dataset.keys())[0]
        data = dataset[split_name]
        print(f"Using split: {split_name}")
    else:
        data = dataset
    
    print(f"Total samples in dataset: {len(data)}")
    print(f"Target model: {target_model}")
    print(f"Minimum score: {min_score}")
    print(f"Target samples to collect: {max_samples}")
    if max_completion_length:
        print(f"Maximum completion length: {max_completion_length} characters")
    
    filtered_samples = []
    processed = 0
    
    for sample in tqdm(data, desc="Processing samples"):
        processed += 1
        
        # Check if we have enough samples
        if len(filtered_samples) >= max_samples:
            break
        
        # Check initial prompt score
        initial_prompt = sample.get('initial_prompt', {})
        metadata = initial_prompt.get('metadata', {})
        
        # Look for complexity x quality score in the specific nested structure
        score = None
        complexity_classification = metadata.get('complexity_classification', {})
        if target_model in complexity_classification:
            model_data = complexity_classification[target_model]
            if 'complexity_x_quality' in model_data:
                score_str = model_data['complexity_x_quality']
                if score_str is not None:
                    try:
                        score = float(score_str)
                    except (ValueError, TypeError):
                        continue
        
        if score is None or score < min_score:
            continue
        
        # Find the target model's conversational branch
        target_branch = None
        target_completion_content = None
        for branch in sample.get('conversation_branches', []):
            for message in branch.get('messages', []):
                if message.get('role') == 'assistant' and 'parts' in message:
                    for part in message.get('parts', []):
                        if isinstance(part, dict) and 'metadata' in part:
                            model = part['metadata'].get('model')
                            if model == target_model and part.get('type') == 'response':
                                target_branch = branch
                                target_completion_content = part.get('content', '')
                                break
                    if target_branch:
                        break
            if target_branch:
                break
        
        if target_branch is None:
            continue
        
        # Check completion length if specified
        if max_completion_length and target_completion_content:
            if len(target_completion_content) > max_completion_length:
                continue
        
        # Create filtered sample with only the target branch
        filtered_sample = {
            'conversation_id': sample['conversation_id'],
            'dataset_source': sample['dataset_source'],
            'original_metadata': sample.get('original_metadata', {}),
            'system_prompt': sample.get('system_prompt', {}),
            'initial_prompt': sample['initial_prompt'],
            'available_functions': sample.get('available_functions', []),
            'conversation_branches': [target_branch],  # Only keep target model branch
            'created_timestamp': sample.get('created_timestamp', ''),
            'filter_metadata': {
                'score': score,
                'model': target_model,
                'original_branch_count': len(sample.get('conversation_branches', [])),
                'completion_length': len(target_completion_content) if target_completion_content else 0
            }
        }
        
        filtered_samples.append(filtered_sample)
        
        if len(filtered_samples) % 100 == 0:
            print(f"Collected {len(filtered_samples)} samples so far...")
    
    print(f"\nProcessed {processed} samples")
    print(f"Collected {len(filtered_samples)} samples meeting criteria")
    
    if not filtered_samples:
        print("No samples found meeting the criteria!")
        return False
    
    # Create new dataset
    print(f"Creating filtered dataset...")
    filtered_dataset = Dataset.from_list(filtered_samples)
    
    # Wrap in DatasetDict to maintain consistency
    filtered_dataset_dict = DatasetDict({'train': filtered_dataset})
    
    # Save the filtered dataset
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving filtered dataset to: {output_path}")
    filtered_dataset_dict.save_to_disk(output_path)
    
    print(f"Successfully created filtered dataset with {len(filtered_samples)} samples")
    
    # Show some statistics
    scores = [sample['filter_metadata']['score'] for sample in filtered_samples]
    lengths = [sample['filter_metadata']['completion_length'] for sample in filtered_samples]
    
    print(f"Score statistics:")
    print(f"  Min score: {min(scores)}")
    print(f"  Max score: {max(scores)}")
    print(f"  Average score: {sum(scores) / len(scores):.2f}")
    
    print(f"Completion length statistics:")
    print(f"  Min length: {min(lengths)} characters")
    print(f"  Max length: {max(lengths)} characters")
    print(f"  Average length: {sum(lengths) / len(lengths):.1f} characters")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Filter dataset for high-quality samples with specific model")
    parser.add_argument("dataset_path", help="Path to the input dataset directory")
    parser.add_argument("output_path", help="Path for the filtered dataset output")
    parser.add_argument("--model", "-m", default="Qwen/Qwen3-32B", 
                       help="Target model to keep (default: Qwen/Qwen3-32B)")
    parser.add_argument("--min-score", "-s", type=float, default=20.0,
                       help="Minimum complexity x quality score (default: 20)")
    parser.add_argument("--max-samples", "-n", type=int, default=500,
                       help="Maximum number of samples to collect (default: 500)")
    parser.add_argument("--max-completion-length", "-l", type=int, default=None,
                       help="Maximum completion length in characters (skip longer completions)")
    
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        print(f"Error: Dataset path {dataset_path} does not exist")
        return 1
    
    try:
        success = filter_dataset(
            args.dataset_path, 
            args.output_path, 
            args.model, 
            args.min_score, 
            args.max_samples,
            args.max_completion_length
        )
        return 0 if success else 1
        
    except Exception as e:
        print(f"Error processing dataset: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())