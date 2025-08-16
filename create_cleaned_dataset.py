#!/usr/bin/env python3
"""
Create cleaned version of olmo-2-preference-quality-short-1100-synthetic dataset.

Filters:
1. Remove all branches with iterations >= 9 (keep only 0-8)
2. Remove sample 563 which has None content in iteration 8
3. Keep only samples that have all iterations 0-8 with valid content

Output: cleaned dataset with exactly 9 iterations per sample (0-8)
"""

from datasets import load_from_disk, Dataset, DatasetDict
import os
from pathlib import Path

# Paths
INPUT_PATH = "/capstor/store/cscs/swissai/infra01/posttrain_data/04_decontaminated_newformat/olmo-2-preference-quality-short-1100-synthetic"
OUTPUT_PATH = "/capstor/store/cscs/swissai/infra01/posttrain_data/04_decontaminated_newformat/olmo-2-preference-quality-short-1100-synthetic-v2"

def filter_sample(sample, sample_idx):
    """Filter a single sample to keep only iterations 0-8 with valid content."""
    
    # Skip the problematic sample 563
    if sample_idx == 563:
        return None
    
    filtered_branches = []
    kept_iterations = set()
    
    for branch in sample['conversation_branches']:
        if branch['messages'] and 'parts' in branch['messages'][0]:
            part = branch['messages'][0]['parts'][0]
            if 'metadata' in part:
                iteration = part['metadata'].get('iteration', -1)
                
                # Keep only iterations 0-8
                if 0 <= iteration <= 8:
                    # Check for valid content (not None)
                    content = part.get('content')
                    if content is not None:
                        filtered_branches.append(branch)
                        kept_iterations.add(iteration)
    
    # Only keep samples that have all 9 iterations (0-8)
    if len(kept_iterations) == 9 and kept_iterations == set(range(9)):
        # Create new sample with filtered branches
        return {
            'conversation_id': sample['conversation_id'],
            'dataset_source': sample['dataset_source'],
            'original_metadata': sample['original_metadata'],
            'system_prompt': sample['system_prompt'],
            'initial_prompt': sample['initial_prompt'],
            'conversation_branches': filtered_branches,
            'created_timestamp': sample['created_timestamp']
        }
    
    return None

def main():
    print(f"Loading dataset from: {INPUT_PATH}")
    ds = load_from_disk(INPUT_PATH)
    
    print(f"Original dataset size: {len(ds['train'])} samples")
    
    # Filter samples
    filtered_samples = []
    removed_count = 0
    
    for i, sample in enumerate(ds['train']):
        filtered_sample = filter_sample(sample, i)
        
        if filtered_sample is not None:
            filtered_samples.append(filtered_sample)
        else:
            removed_count += 1
            if i == 563:
                print(f"  Removed sample {i} (problematic sample with None content)")
            else:
                print(f"  Removed sample {i} (incomplete iterations)")
    
    print(f"\nFiltering complete:")
    print(f"  Kept: {len(filtered_samples)} samples")
    print(f"  Removed: {removed_count} samples")
    
    # Create new dataset
    if filtered_samples:
        # Verify each sample has exactly 9 branches (iterations 0-8)
        for i, sample in enumerate(filtered_samples):
            branch_count = len(sample['conversation_branches'])
            if branch_count != 9:
                print(f"WARNING: Sample {i} has {branch_count} branches instead of 9")
        
        # Create new DatasetDict
        new_dataset = Dataset.from_list(filtered_samples)
        new_dataset_dict = DatasetDict({'train': new_dataset})
        
        # Save to disk
        print(f"\nSaving cleaned dataset to: {OUTPUT_PATH}")
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        new_dataset_dict.save_to_disk(OUTPUT_PATH)
        
        print("Dataset saved successfully!")
        print(f"Final dataset size: {len(new_dataset)} samples")
        
        # Verify iteration distribution
        from collections import defaultdict
        iteration_counts = defaultdict(int)
        
        for sample in filtered_samples:
            for branch in sample['conversation_branches']:
                if branch['messages'] and 'parts' in branch['messages'][0]:
                    part = branch['messages'][0]['parts'][0]
                    if 'metadata' in part:
                        iteration = part['metadata'].get('iteration', -1)
                        iteration_counts[iteration] += 1
        
        print(f"\nIteration distribution in cleaned dataset:")
        for iteration in sorted(iteration_counts.keys()):
            print(f"  Iteration {iteration}: {iteration_counts[iteration]} branches")
            
    else:
        print("ERROR: No samples remained after filtering!")

if __name__ == "__main__":
    main()