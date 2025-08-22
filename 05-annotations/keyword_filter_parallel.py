#!/usr/bin/env python3
"""
High-performance parallel keyword filtering for large datasets.

Uses transform-and-reconstruct approach to avoid Arrow schema conflicts when
adding new metadata fields. Based on the schema_transform_parallel.py pattern.
"""

import json
import multiprocessing as mp
from pathlib import Path
from datasets import load_from_disk, Dataset
import time
from tqdm import tqdm
import copy
import re
from datetime import datetime

# Keyword list for filtering AI model references
FILTER_KEYWORDS = [
    "chatgpt", "chat gpt", "Chat-gpt", "gpt3", "gpt4", "gpt5",
    "gpt-3", "gpt-4", "gpt-5", "openai", "open-ai", "open ai", 
    "openassistant", "open assistant", "open-assistant", "eurollm", "euroblock"
]

def find_keywords_in_text(text: str, keywords: list) -> list:
    """Find keywords in text using case-insensitive word boundary matching."""
    if not text or not keywords:
        return []
    
    found_keywords = []
    text_lower = text.lower()
    
    for keyword in keywords:
        keyword_lower = keyword.lower()
        pattern = r'\b' + re.escape(keyword_lower) + r'\b'
        if re.search(pattern, text_lower):
            found_keywords.append(keyword)
    
    return found_keywords

def annotate_content_with_keywords(content: str, keywords: list) -> dict:
    """Analyze content for keywords and return metadata."""
    found_keywords = find_keywords_in_text(content, keywords)
    
    return {
        "assistant_keywords_found": found_keywords,
        "has_assistant_keywords": len(found_keywords) > 0,
        "timestamp": datetime.now().isoformat(),
        "filter_type": "assistant_keyword_detection",
        "total_keywords_checked": len(keywords)
    }

def transform_sample_with_keywords(sample, keywords):
    """Transform a sample to add keyword detection metadata consistently."""
    
    # Track if any content in this sample has keywords
    sample_has_keywords = False
    
    def get_empty_keyword_result():
        return {
            "assistant_keywords_found": [],
            "has_assistant_keywords": False,
            "timestamp": datetime.now().isoformat(),
            "filter_type": "assistant_keyword_detection",
            "total_keywords_checked": len(keywords)
        }
    
    def process_content(content, existing_metadata=None):
        """Process content and add keyword detection to metadata."""
        nonlocal sample_has_keywords
        metadata = copy.deepcopy(existing_metadata) if existing_metadata else {}
        
        if content and isinstance(content, str):
            content = content.strip()
            if content:
                keyword_result = annotate_content_with_keywords(content, keywords)
            else:
                keyword_result = get_empty_keyword_result()
        else:
            keyword_result = get_empty_keyword_result()
        
        # Update sample-level flag if keywords found
        if keyword_result["has_assistant_keywords"]:
            sample_has_keywords = True
        
        metadata["keyword_detection"] = keyword_result
        return metadata
    
    # Create deep copy to avoid modifying original
    result = copy.deepcopy(sample)
    
    # Process system prompt
    if "system_prompt" in result and result["system_prompt"]:
        content = result["system_prompt"].get("content", "")
        existing_metadata = result["system_prompt"].get("metadata", {})
        result["system_prompt"]["metadata"] = process_content(content, existing_metadata)
    
    # Process initial prompt
    if "initial_prompt" in result and result["initial_prompt"]:
        content = result["initial_prompt"].get("content", "")
        existing_metadata = result["initial_prompt"].get("metadata", {})
        result["initial_prompt"]["metadata"] = process_content(content, existing_metadata)
    
    # Process conversation branches
    if "conversation_branches" in result:
        for branch in result["conversation_branches"]:
            if "messages" in branch:
                for message in branch["messages"]:
                    # Ensure message has metadata
                    if "metadata" not in message:
                        message["metadata"] = {}
                    
                    # Process parts if they exist (new format)
                    if "parts" in message and isinstance(message["parts"], list):
                        for part in message["parts"]:
                            if isinstance(part, dict):
                                content = part.get("content", "")
                                existing_metadata = part.get("metadata", {})
                                part["metadata"] = process_content(content, existing_metadata)
                    else:
                        # Process direct content (old format)
                        content = message.get("content", "")
                        existing_metadata = message.get("metadata", {})
                        message["metadata"] = process_content(content, existing_metadata)
    
    # Add sample-level keyword detection flag
    result["sample_has_assistant_keywords"] = sample_has_keywords
    
    return result

# Global variables for multiprocessing
keywords_global = None
progress_counter = None
dataset_path_global = None
split_name_global = None

def init_worker(keywords_arg, progress_counter_arg, dataset_path_arg, split_name_arg):
    """Initialize worker process with shared objects"""
    global keywords_global, progress_counter, dataset_path_global, split_name_global
    keywords_global = keywords_arg
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
        transformed_sample = transform_sample_with_keywords(sample, keywords_global)
        transformed_samples.append(transformed_sample)
    
    # Update progress counter when chunk is complete
    with progress_counter.get_lock():
        progress_counter.value += 1
    
    return transformed_samples

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Add keyword detection to datasets using parallel transform-and-reconstruct",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  # Default keywords (AI model references)
  venv/bin/python 05-annotations/keyword_filter_parallel.py input_dataset output_dataset
  
  # Custom keywords
  venv/bin/python 05-annotations/keyword_filter_parallel.py input_dataset output_dataset --keywords word1 word2 word3

Default keywords: {', '.join(FILTER_KEYWORDS)}
        """
    )
    parser.add_argument("input_path", help="Path to input dataset")
    parser.add_argument("output_path", help="Path to save annotated dataset")
    parser.add_argument("--keywords", nargs="*", default=FILTER_KEYWORDS, 
                        help="Custom keywords to search for (default: AI model keywords)")
    parser.add_argument("--max-processes", type=int, default=200,
                        help="Maximum number of parallel processes (default: 200)")
    args = parser.parse_args()
    
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
    
    # Display keyword configuration
    keywords = args.keywords
    print(f"\nKeyword filtering with {len(keywords)} keywords:")
    for i, keyword in enumerate(keywords, 1):
        print(f"  {i:2d}. {keyword}")
    print()
    
    # Limit CPU processes
    max_processes = min(args.max_processes, mp.cpu_count())
    print(f"Processing {dataset_size:,} samples with {max_processes} cores...")
    
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
                 initargs=(keywords, progress_counter, args.input_path, split_name)) as pool:
        
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
    
    print(f"Collected {len(all_transformed_samples):,} transformed samples")
    
    # Create new dataset from transformed samples - this may take several minutes
    print("Creating new dataset from transformed samples (this may take several minutes)...")
    new_dataset = Dataset.from_list(all_transformed_samples)
    
    # Save the new dataset
    print(f"Saving annotated dataset to {args.output_path}...")
    new_dataset.save_to_disk(args.output_path)
    
    # Calculate statistics using the sample-level flag
    print("\nCalculating statistics...")
    samples_with_keywords = 0
    
    for sample in tqdm(new_dataset, desc="Counting samples with keywords"):
        if sample.get("sample_has_assistant_keywords", False):
            samples_with_keywords += 1
    
    # Count total individual keyword detections for detailed reporting
    total_keywords_found = 0
    for sample in tqdm(new_dataset, desc="Counting individual detections"):
        # Check system prompt
        if (sample.get("system_prompt", {}).get("metadata", {}).get("keyword_detection", {}).get("has_assistant_keywords")):
            total_keywords_found += 1
        
        # Check initial prompt
        if (sample.get("initial_prompt", {}).get("metadata", {}).get("keyword_detection", {}).get("has_assistant_keywords")):
            total_keywords_found += 1
        
        # Check conversation branches
        for branch in sample.get("conversation_branches", []):
            for message in branch.get("messages", []):
                if (message.get("metadata", {}).get("keyword_detection", {}).get("has_assistant_keywords")):
                    total_keywords_found += 1
                
                # Check parts
                for part in message.get("parts", []):
                    if (part.get("metadata", {}).get("keyword_detection", {}).get("has_assistant_keywords")):
                        total_keywords_found += 1
    
    # Save metadata
    metadata = {
        "operation": "assistant_keyword_annotation",
        "script": "keyword_filter_parallel.py",
        "timestamp": datetime.now().isoformat(),
        "input_path": str(args.input_path),
        "output_path": str(args.output_path),
        "total_samples": len(all_transformed_samples),
        "samples_with_keywords": samples_with_keywords,
        "sample_level_keyword_percentage": (samples_with_keywords/len(all_transformed_samples)*100) if all_transformed_samples else 0.0,
        "total_keyword_detections": total_keywords_found,
        "keywords_searched": keywords,
        "format": "new_chat_format_with_parts_and_sample_level_flag"
    }
    
    metadata_path = Path(args.output_path) / "dataset_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nTransformation complete! Dataset saved to {args.output_path}")
    print(f"Total samples: {len(all_transformed_samples):,}")
    print(f"Samples with keywords: {samples_with_keywords:,} ({(samples_with_keywords/len(all_transformed_samples)*100):.2f}%)")
    print(f"Total keyword detections: {total_keywords_found:,}")
    print(f"Keywords searched: {keywords}")
    print(f"Metadata saved to: {metadata_path}")

if __name__ == "__main__":
    main()