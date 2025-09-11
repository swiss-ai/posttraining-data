#!/usr/bin/env python3
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
from datasets import load_from_disk
from functools import partial
from tqdm import tqdm
import time

def strip_values(obj):
    """Strip values, keep only structure/schema"""
    if isinstance(obj, dict):
        return {k: strip_values(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [strip_values(obj[0])] if obj else []
    else:
        return type(obj).__name__

# Global variables for multiprocessing
shared_dict = None
lock = None
progress_counter = None
dataset_path_global = None
split_name_global = None

def init_worker(shared_dict_arg, lock_arg, progress_counter_arg, dataset_path_arg, split_name_arg):
    """Initialize worker process with shared objects"""
    global shared_dict, lock, progress_counter, dataset_path_global, split_name_global
    shared_dict = shared_dict_arg
    lock = lock_arg
    progress_counter = progress_counter_arg
    dataset_path_global = dataset_path_arg
    split_name_global = split_name_arg

def process_chunk_indices(chunk_start_end):
    """Process a chunk of samples by indices"""
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
    local_hashes = set()
    
    for idx in range(start_idx, end_idx):
        sample = dataset_split[idx]
        schema = strip_values(sample)
        schema_str = json.dumps(schema, sort_keys=True)
        hash_val = hashlib.sha256(schema_str.encode()).hexdigest()
        
        if hash_val not in local_hashes:
            with lock:
                if hash_val not in shared_dict:
                    shared_dict[hash_val] = schema_str
            local_hashes.add(hash_val)
    
    # Update progress counter when chunk is complete
    with progress_counter.get_lock():
        progress_counter.value += 1

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Extract unique schemas from dataset in parallel")
    parser.add_argument("input_dataset", help="Path to input dataset")
    parser.add_argument("output_schemas", help="Path to save unique schemas JSON file")
    args = parser.parse_args()
    
    dataset_path = Path(args.input_dataset)
    output_path = Path(args.output_schemas)
    
    print(f"Loading dataset from: {dataset_path}")
    dataset = load_from_disk(str(dataset_path))
    
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
    
    # Create shared dict, lock, and progress counter
    print("Start MP manager...")
    manager = mp.Manager()
    shared_dict = manager.dict()
    lock = manager.Lock()
    progress_counter = mp.Value('i', 0)
    
    # Split into index ranges instead of materializing chunks
    chunk_size = max(1, dataset_size // (max_processes * 4))
    print(f"Split data into {chunk_size} sized chunks")
    chunk_indices = [(i, min(i + chunk_size, dataset_size)) for i in range(0, dataset_size, chunk_size)]
    total_chunks = len(chunk_indices)
    
    print(f"Split into {total_chunks} chunks of ~{chunk_size} samples each")
    
    # Process in parallel with progress tracking
    with mp.Pool(max_processes, initializer=init_worker, 
                 initargs=(shared_dict, lock, progress_counter, str(dataset_path), split_name)) as pool:
        
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
        
        # Wait for completion
        result.get()
    
    print(f"Found {len(shared_dict)} unique schemas")
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save results
    with open(str(output_path), "w") as f:
        json.dump(dict(shared_dict), f, indent=2)
    
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()