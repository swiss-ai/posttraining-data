#!/usr/bin/env python3
import json
import multiprocessing as mp
from pathlib import Path
from datasets import load_from_disk, Dataset
import time
from tqdm import tqdm
import copy

def transform_to_uniform_schema(sample, uniform_schema):
    """Transform a sample to conform to the uniform schema"""
    
    def apply_schema_to_value(value, schema_template):
        """Recursively apply schema template to a value"""
        if isinstance(schema_template, dict):
            # Handle dictionary structures
            result = {}
            for key, template_value in schema_template.items():
                if isinstance(value, dict) and key in value and value[key] is not None:
                    # Key exists in sample and is not None, recursively apply schema
                    result[key] = apply_schema_to_value(value[key], template_value)
                else:
                    # Key missing in sample or is None, use default from template
                    result[key] = get_default_value(template_value)
            return result
            
        elif isinstance(schema_template, list) and schema_template:
            # Handle list structures - apply template to each element
            if not isinstance(value, list) or value is None:
                return []
            
            result = []
            template_item = schema_template[0]
            for item in value:
                result.append(apply_schema_to_value(item, template_item))
            return result
            
        else:
            # Handle primitive types
            if value is None:
                return get_default_value(schema_template)
            
            if schema_template == "str":
                return str(value)
            elif schema_template == "int":
                if isinstance(value, (int, float)):
                    return int(value)
                elif isinstance(value, str) and value.isdigit():
                    return int(value)
                else:
                    return 0
            elif schema_template == "bool":
                return bool(value)
            else:
                return value
    
    def get_default_value(template_value):
        """Get default value for a template"""
        if template_value == "str":
            return ""
        elif template_value == "int":
            return 0
        elif template_value == "bool":
            return False
        elif isinstance(template_value, list):
            return []
        elif isinstance(template_value, dict):
            return {}
        else:
            return ""
    
    # Apply uniform schema to the sample
    transformed = apply_schema_to_value(sample, uniform_schema)
    return transformed

def should_keep_sample(sample):
    """Check if sample should be kept based on filtering criteria.
    
    Returns True if sample should be kept, False if it should be filtered out.
    """
    # Filter out samples with empty conversation messages
    if "conversation_branches" in sample:
        for branch in sample.get("conversation_branches", []):
            messages = branch.get("messages", [])
            if len(messages) == 0:
                return False  # Filter out samples with no messages
    
    return True  # Keep sample

# Global variables for multiprocessing
uniform_schema_global = None
progress_counter = None
dataset_path_global = None
split_name_global = None

def init_worker(uniform_schema_arg, progress_counter_arg, dataset_path_arg, split_name_arg):
    """Initialize worker process with shared objects"""
    global uniform_schema_global, progress_counter, dataset_path_global, split_name_global
    uniform_schema_global = uniform_schema_arg
    progress_counter = progress_counter_arg
    dataset_path_global = dataset_path_arg
    split_name_global = split_name_arg

def process_chunk_indices(chunk_start_end):
    """Process a chunk of samples by indices and return transformed samples"""
    # Load dataset in worker process to avoid sharing large objects
    from datasets import load_from_disk
    import datasets
    # Disable tqdm for dataset loading in workers
    datasets.disable_progress_bar()
    dataset = load_from_disk(str(dataset_path_global))
    
    # Handle both Dataset and DatasetDict
    if hasattr(dataset, 'keys'):  # DatasetDict
        dataset_split = dataset[split_name_global]
    else:  # Single Dataset
        dataset_split = dataset
    
    start_idx, end_idx = chunk_start_end
    transformed_samples = []
    
    for idx in range(start_idx, end_idx):
        sample = dataset_split[idx]
        
        # Filter out samples with empty conversations
        if not should_keep_sample(sample):
            continue  # Skip this sample
        
        transformed_sample = transform_to_uniform_schema(sample, uniform_schema_global)
        transformed_samples.append(transformed_sample)
    
    # Update progress counter when chunk is complete
    with progress_counter.get_lock():
        progress_counter.value += 1
    
    return transformed_samples

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Transform dataset to uniform schema in parallel")
    parser.add_argument("input_path", help="Path to input dataset")
    parser.add_argument("output_path", help="Path to save transformed dataset")
    parser.add_argument("--uniform-schema", default="uniform_schema.json", help="Path to uniform schema JSON file")
    args = parser.parse_args()
    
    # Load uniform schema
    print("Loading uniform schema...")
    with open(args.uniform_schema, 'r') as f:
        uniform_schema = json.load(f)
    
    print("Loading dataset...")
    dataset = load_from_disk(str(args.input_path))
    
    # Handle both Dataset and DatasetDict
    print("Determine dataset type and size")
    if hasattr(dataset, 'keys'):  # DatasetDict
        split_name = 'train' if 'train' in dataset else list(dataset.keys())[0]
        dataset_size = dataset.num_rows[split_name] if hasattr(dataset, 'num_rows') else len(dataset[split_name])
    else:  # Single Dataset
        split_name = None  # Not needed for single dataset
        dataset_size = dataset.num_rows if hasattr(dataset, 'num_rows') else len(dataset)
    
    # Limit to 200 CPU processes max
    max_processes = min(200, mp.cpu_count())
    print(f"Processing {dataset_size} samples with {max_processes} cores...")
    
    # Create progress counter
    print("Start MP manager...")
    progress_counter = mp.Value('i', 0)
    
    # Split into index ranges instead of materializing chunks
    chunk_size = max(1, dataset_size // (max_processes * 4))
    print(f"Split data into {chunk_size} sized chunks")
    chunk_indices = [(i, min(i + chunk_size, dataset_size)) for i in range(0, dataset_size, chunk_size)]
    total_chunks = len(chunk_indices)
    
    print(f"Split into {total_chunks} chunks of ~{chunk_size} samples each")
    
    # Process in parallel with progress tracking
    with mp.Pool(max_processes, initializer=init_worker, 
                 initargs=(uniform_schema, progress_counter, args.input_path, split_name)) as pool:
        
        # Start async processing
        result = pool.map_async(process_chunk_indices, chunk_indices)
        
        # Progress bar
        with tqdm(total=total_chunks, desc="Processing chunks", unit="chunk") as pbar:
            while not result.ready():
                current_progress = progress_counter.value
                pbar.n = current_progress
                pbar.refresh()
                time.sleep(0.1)
            
            # Final update
            pbar.n = total_chunks
            pbar.refresh()
        
        # Get results - this may take a minute with large datasets
        print("Waiting for workers to complete and collecting results...")
        chunk_results = result.get()
    
    # Flatten results into single list
    print("Flattening results from all workers...")
    all_transformed_samples = []
    with tqdm(total=len(chunk_results), desc="Collecting chunks", unit="chunk") as pbar:
        for chunk_samples in chunk_results:
            all_transformed_samples.extend(chunk_samples)
            pbar.update(1)
    
    print(f"Collected {len(all_transformed_samples)} transformed samples")
    
    # Create new dataset from transformed samples - this may take several minutes
    print("Creating new dataset from transformed samples (this may take several minutes)...")
    new_dataset = Dataset.from_list(all_transformed_samples)
    
    # Save the new dataset
    print(f"Saving transformed dataset to {args.output_path}...")
    new_dataset.save_to_disk(args.output_path)
    
    # Calculate filtering statistics
    samples_filtered = dataset_size - len(all_transformed_samples)
    
    print(f"Transformation complete! New dataset saved to {args.output_path}")
    print(f"Original samples: {dataset_size:,}")
    print(f"Samples filtered out (empty conversations): {samples_filtered:,}")
    print(f"Final dataset samples: {len(all_transformed_samples):,}")
    
    if samples_filtered > 0:
        filter_percentage = (samples_filtered / dataset_size) * 100
        print(f"Filtering rate: {filter_percentage:.4f}%")

if __name__ == "__main__":
    main()