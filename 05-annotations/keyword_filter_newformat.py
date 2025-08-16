#!/usr/bin/env python3
"""
Keyword filtering script for new chat format datasets (with parts structure).

This script searches for specific keywords in system prompts, initial prompts, and 
all message parts, adding metadata about found keywords and a boolean flag. Designed 
to efficiently identify and filter assistant responses containing AI model references.
"""

import sys
import json
import argparse
import re
import gc
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Set
from datasets import load_from_disk, Dataset, DatasetDict, concatenate_datasets
from tqdm import tqdm

# Keyword list for filtering AI model references
FILTER_KEYWORDS = [
    "claude", "chatgpt", "chat gpt", "gpt3", "gpt4", "gpt5", 
    "gpt-3", "gpt-4", "perplexity", "openai", "open-ai", "open ai", 
    "gemini", "openassistant", "eurollm", "euroblock"
]


def find_keywords_in_text(text: str, keywords: List[str]) -> List[str]:
    """
    Find keywords in text using case-insensitive word boundary matching.
    
    Args:
        text: Text to search in
        keywords: List of keywords to search for
        
    Returns:
        List of found keywords (original case from keyword list)
    """
    if not text or not keywords:
        return []
    
    found_keywords = []
    text_lower = text.lower()
    
    for keyword in keywords:
        keyword_lower = keyword.lower()
        
        # Use word boundary regex for exact matches
        # \b ensures we match whole words, not substrings
        pattern = r'\b' + re.escape(keyword_lower) + r'\b'
        
        if re.search(pattern, text_lower):
            found_keywords.append(keyword)
    
    return found_keywords


def annotate_content_with_keywords(content: str, keywords: List[str]) -> Dict[str, Any]:
    """
    Analyze content for keywords and return metadata.
    
    Args:
        content: Text content to analyze
        keywords: List of keywords to search for
        
    Returns:
        Dictionary with keyword detection results
    """
    found_keywords = find_keywords_in_text(content, keywords)
    
    return {
        "assistant_keywords_found": found_keywords,
        "has_assistant_keywords": len(found_keywords) > 0,
        "timestamp": datetime.now().isoformat(),
        "filter_type": "assistant_keyword_detection",
        "total_keywords_checked": len(keywords)
    }


def annotate_message_parts_keywords(message: Dict[str, Any], keywords: List[str]) -> Dict[str, Any]:
    """
    Add keyword detection to ALL parts in a message (new format with parts structure).
    CRITICAL: Adds keyword_detection to ALL parts for Arrow schema consistency.
    
    Args:
        message: Message dictionary with parts structure
        keywords: List of keywords to search for
        
    Returns:
        Updated message with keyword annotations in ALL parts
    """
    import copy
    
    # Create deep copy to avoid in-place modification (prevents Arrow corruption)
    message_copy = copy.deepcopy(message)
    
    # Helper function to create empty keyword result
    def empty_keyword_result():
        return {
            "assistant_keywords_found": [],
            "has_assistant_keywords": False,
            "timestamp": datetime.now().isoformat(),
            "filter_type": "assistant_keyword_detection",
            "total_keywords_checked": len(keywords)
        }
    
    # Process parts if they exist
    if "parts" in message_copy and isinstance(message_copy["parts"], list):
        for i, part in enumerate(message_copy["parts"]):
            # Ensure part is a dict
            if not isinstance(part, dict):
                continue
            
            # Ensure metadata exists for ALL parts
            if "metadata" not in message_copy["parts"][i]:
                message_copy["parts"][i]["metadata"] = {}
            
            # Check if part has content
            if "content" in part and isinstance(part["content"], str):
                content = part["content"].strip()
                if content:
                    # Analyze keywords
                    keyword_result = annotate_content_with_keywords(content, keywords)
                else:
                    # Empty content - add empty result
                    keyword_result = empty_keyword_result()
            else:
                # No content field - add empty result
                keyword_result = empty_keyword_result()
            
            # ALWAYS add keyword_detection field
            message_copy["parts"][i]["metadata"]["keyword_detection"] = keyword_result
    
    return message_copy


def annotate_sample_keywords(sample: Dict[str, Any], keywords: List[str]) -> Dict[str, Any]:
    """
    Add keyword detection to all content in a sample (new format).
    CRITICAL: Adds keyword_detection field to ALL messages for Arrow schema consistency.
    
    Args:
        sample: Chat format sample (may have parts structure in messages)
        keywords: List of keywords to search for
        
    Returns:
        Updated sample with keyword annotations on ALL messages
    """
    # Create a deep copy to avoid in-place modification (critical for Arrow compatibility)
    import copy
    sample_copy = copy.deepcopy(sample)
    
    # Helper function to create empty keyword result
    def empty_keyword_result():
        return {
            "assistant_keywords_found": [],
            "has_assistant_keywords": False,
            "timestamp": datetime.now().isoformat(),
            "filter_type": "assistant_keyword_detection",
            "total_keywords_checked": len(keywords)
        }
    
    # Annotate system prompt - ALWAYS add field
    if "system_prompt" in sample_copy and sample_copy["system_prompt"]:
        if "metadata" not in sample_copy["system_prompt"]:
            sample_copy["system_prompt"]["metadata"] = {}
        
        content = sample_copy["system_prompt"].get("content", "").strip()
        if content:
            keyword_result = annotate_content_with_keywords(content, keywords)
        else:
            keyword_result = empty_keyword_result()
        
        sample_copy["system_prompt"]["metadata"]["keyword_detection"] = keyword_result
    
    # Annotate initial prompt - ALWAYS add field
    if "initial_prompt" in sample_copy and sample_copy["initial_prompt"]:
        if "metadata" not in sample_copy["initial_prompt"]:
            sample_copy["initial_prompt"]["metadata"] = {}
        
        content = sample_copy["initial_prompt"].get("content", "").strip()
        if content:
            keyword_result = annotate_content_with_keywords(content, keywords)
        else:
            keyword_result = empty_keyword_result()
        
        sample_copy["initial_prompt"]["metadata"]["keyword_detection"] = keyword_result
    
    # Annotate conversation branches (messages may have parts structure)
    if "conversation_branches" in sample_copy:
        for branch in sample_copy["conversation_branches"]:
            if "messages" in branch:
                for i, message in enumerate(branch["messages"]):
                    # Ensure metadata exists
                    if "metadata" not in branch["messages"][i]:
                        branch["messages"][i]["metadata"] = {}
                    
                    # Check if message has parts (new format) or direct content
                    if "parts" in message:
                        # New format with parts - use parts annotation
                        branch["messages"][i] = annotate_message_parts_keywords(message, keywords)
                    else:
                        # Direct content or no content - ALWAYS add field
                        content = message.get("content", "").strip() if "content" in message else ""
                        if content:
                            keyword_result = annotate_content_with_keywords(content, keywords)
                        else:
                            keyword_result = empty_keyword_result()
                        
                        branch["messages"][i]["metadata"]["keyword_detection"] = keyword_result
    
    return sample_copy


def prepare_batch_schema(examples: Dict[str, List]) -> Dict[str, List]:
    """
    First pass: Add empty metadata structures to ensure consistent schema.
    This prepares the dataset for parallel keyword detection.
    
    Args:
        examples: Batch of samples in HuggingFace format
        
    Returns:
        Batch with empty keyword_detection metadata added
    """
    import copy
    
    # Deep copy to avoid modifying input
    result = copy.deepcopy(examples)
    
    batch_size = len(result[next(iter(result))])
    
    # Helper function to create empty keyword result
    def empty_keyword_result():
        return {
            "assistant_keywords_found": [],
            "has_assistant_keywords": False,
            "timestamp": datetime.now().isoformat(),
            "filter_type": "assistant_keyword_detection",
            "total_keywords_checked": 0
        }
    
    # Process each sample in the batch
    for idx in range(batch_size):
        # Handle system_prompt
        if "system_prompt" in result and result["system_prompt"][idx] is not None:
            if "metadata" not in result["system_prompt"][idx]:
                result["system_prompt"][idx]["metadata"] = {}
            result["system_prompt"][idx]["metadata"]["keyword_detection"] = empty_keyword_result()
        
        # Handle initial_prompt
        if "initial_prompt" in result and result["initial_prompt"][idx] is not None:
            if "metadata" not in result["initial_prompt"][idx]:
                result["initial_prompt"][idx]["metadata"] = {}
            result["initial_prompt"][idx]["metadata"]["keyword_detection"] = empty_keyword_result()
        
        # Handle conversation branches
        if "conversation_branches" in result and result["conversation_branches"][idx] is not None:
            for branch in result["conversation_branches"][idx]:
                if "messages" in branch:
                    for message in branch["messages"]:
                        if "metadata" not in message:
                            message["metadata"] = {}
                        message["metadata"]["keyword_detection"] = empty_keyword_result()
                        
                        # Handle parts if they exist
                        if "parts" in message and isinstance(message["parts"], list):
                            for part in message["parts"]:
                                if isinstance(part, dict):
                                    if "metadata" not in part:
                                        part["metadata"] = {}
                                    part["metadata"]["keyword_detection"] = empty_keyword_result()
    
    return result


def process_batch_simple(examples: Dict[str, List], keywords: List[str]) -> Dict[str, List]:
    """
    Simple batch processing for keyword annotation.
    
    Args:
        examples: Batch of samples in HuggingFace format
        keywords: List of keywords to search for
        
    Returns:
        Batch with keyword annotations added
    """
    import copy
    
    # Deep copy to avoid modifying input
    result = copy.deepcopy(examples)
    
    batch_size = len(result[next(iter(result))])
    
    # Process each sample
    for idx in range(batch_size):
        sample = {key: result[key][idx] for key in result}
        annotated_sample = annotate_sample_keywords(sample, keywords)
        
        # Put back into batch
        for key in result:
            result[key][idx] = annotated_sample[key]
    
    return result


def fill_batch_keywords(examples: Dict[str, List], keywords: List[str]) -> Dict[str, List]:
    """
    Second pass: Fill in keyword detection in existing metadata structures.
    Assumes keyword_detection fields already exist from first pass.
    
    Args:
        examples: Batch of samples with prepared schema
        keywords: List of keywords to search for
        
    Returns:
        Batch with filled keyword detection
    """
    import copy
    
    # Deep copy to avoid modifying input
    result = copy.deepcopy(examples)
    
    batch_size = len(result[next(iter(result))])
    
    # Process each sample in the batch
    for idx in range(batch_size):
        # Handle system_prompt
        if "system_prompt" in result and result["system_prompt"][idx] is not None:
            content = result["system_prompt"][idx].get("content", "").strip()
            if content and "metadata" in result["system_prompt"][idx]:
                keyword_result = annotate_content_with_keywords(content, keywords)
                result["system_prompt"][idx]["metadata"]["keyword_detection"] = keyword_result
        
        # Handle initial_prompt
        if "initial_prompt" in result and result["initial_prompt"][idx] is not None:
            content = result["initial_prompt"][idx].get("content", "").strip()
            if content and "metadata" in result["initial_prompt"][idx]:
                keyword_result = annotate_content_with_keywords(content, keywords)
                result["initial_prompt"][idx]["metadata"]["keyword_detection"] = keyword_result
        
        # Handle conversation branches
        if "conversation_branches" in result and result["conversation_branches"][idx] is not None:
            for branch in result["conversation_branches"][idx]:
                if "messages" in branch:
                    for message in branch["messages"]:
                        # Handle parts if they exist
                        if "parts" in message and isinstance(message["parts"], list):
                            for part in message["parts"]:
                                if isinstance(part, dict) and "content" in part and "metadata" in part:
                                    content = part["content"].strip()
                                    if content:
                                        keyword_result = annotate_content_with_keywords(content, keywords)
                                        part["metadata"]["keyword_detection"] = keyword_result
                        # Handle direct content
                        elif "content" in message and "metadata" in message:
                            content = message.get("content", "").strip()
                            if content:
                                keyword_result = annotate_content_with_keywords(content, keywords)
                                message["metadata"]["keyword_detection"] = keyword_result
    
    return result


def annotate_dataset_streaming(dataset, keywords: List[str], chunk_size: int = 10000):
    """
    Annotate dataset using streaming approach with chunking for memory efficiency.
    
    Args:
        dataset: HuggingFace Dataset
        keywords: List of keywords to search for
        chunk_size: Number of samples to process in each chunk
        
    Returns:
        Annotated Dataset and processing stats
    """
    print(f"Annotating dataset in chunks of {chunk_size:,} samples using streaming approach...")
    total_samples = len(dataset)
    total_messages = 0
    total_keywords_found = 0
    
    chunk_datasets = []
    
    # Process dataset in chunks
    with tqdm(total=total_samples, desc="Assistant keyword annotation", unit="samples") as pbar:
        for chunk_start in range(0, total_samples, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_samples)
            chunk_samples = []
            chunk_messages = 0
            chunk_keywords = 0
            
            # Process chunk
            for idx in range(chunk_start, chunk_end):
                sample = dataset[idx]
                
                # Count messages before annotation
                message_count = 0
                if sample.get("system_prompt"):
                    message_count += 1
                if sample.get("initial_prompt"):
                    message_count += 1
                for branch in sample.get("conversation_branches", []):
                    message_count += len(branch.get("messages", []))
                
                # Annotate sample
                annotated_sample = annotate_sample_keywords(sample, keywords)
                chunk_samples.append(annotated_sample)
                chunk_messages += message_count
                
                # Count keywords found in this sample (for stats)
                sample_keywords = count_keywords_in_sample(annotated_sample)
                chunk_keywords += sample_keywords
                
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix(
                    messages=f"{total_messages + chunk_messages:,}",
                    keywords=f"{total_keywords_found + chunk_keywords:,}"
                )
            
            # Create chunk dataset
            if chunk_samples:
                chunk_dataset = Dataset.from_list(chunk_samples)
                chunk_datasets.append(chunk_dataset)
                total_messages += chunk_messages
                total_keywords_found += chunk_keywords
                
                # Clear chunk samples from memory
                del chunk_samples
                gc.collect()
    
    print(f"Annotation complete: {total_samples:,} samples, {total_messages:,} messages, {total_keywords_found:,} keyword detections")
    
    if not chunk_datasets:
        return Dataset.from_list([]), {"total_messages": 0, "total_keywords_found": 0}
    
    # Combine using concatenate_datasets for memory efficiency
    print("Combining chunk datasets...")
    final_dataset = concatenate_datasets(chunk_datasets)
    
    # Clear chunk datasets from memory
    del chunk_datasets
    gc.collect()
    
    return final_dataset, {"total_messages": total_messages, "total_keywords_found": total_keywords_found}


def annotate_dataset(dataset, keywords: List[str], num_proc: int = None, batch_size: int = 1000):
    """
    Annotate entire dataset sequentially.
    
    Sequential processing is required due to Arrow schema constraints when adding
    new nested metadata fields. Still efficient at ~3000+ samples/sec.
    
    Args:
        dataset: HuggingFace Dataset
        keywords: List of keywords to search for
        num_proc: Number of processes (ignored - always uses 1)
        batch_size: Size of batches for processing
        
    Returns:
        Tuple of (annotated_dataset, stats_dict)
    """
    if num_proc and num_proc > 1:
        print(f"Note: Parallel processing not supported when adding nested metadata.")
        print(f"      Using sequential processing (~3000+ samples/sec).")
    
    print(f"Processing dataset sequentially with batch size {batch_size:,}")
    print(f"Dataset size: {len(dataset):,} samples")
    
    # Process with single process - clean and simple
    annotated_dataset = dataset.map(
        lambda examples: process_batch_simple(examples, keywords),
        batched=True,
        batch_size=batch_size,
        num_proc=1,
        load_from_cache_file=False,
        desc="Assistant keyword annotation"
    )
    
    # Calculate statistics after processing
    print("Calculating statistics...")
    total_messages = 0
    total_keywords_found = 0
    
    for sample in tqdm(annotated_dataset, desc="Counting statistics"):
        # Count messages
        if sample.get("system_prompt"):
            total_messages += 1
        if sample.get("initial_prompt"):
            total_messages += 1
        for branch in sample.get("conversation_branches", []):
            total_messages += len(branch.get("messages", []))
        
        # Count keywords
        total_keywords_found += count_keywords_in_sample(sample)
    
    print(f"Annotation complete: {len(annotated_dataset):,} samples, {total_messages:,} messages, {total_keywords_found:,} keyword detections")
    
    return annotated_dataset, {"total_messages": total_messages, "total_keywords_found": total_keywords_found}


def count_keywords_in_sample(sample: Dict[str, Any]) -> int:
    """
    Count total keyword detections in a sample for statistics.
    
    Args:
        sample: Annotated sample
        
    Returns:
        Total number of keyword detections found
    """
    count = 0
    
    # Check system prompt
    if (sample.get("system_prompt", {}).get("metadata", {}).get("keyword_detection", {}).get("has_assistant_keywords")):
        count += 1
    
    # Check initial prompt
    if (sample.get("initial_prompt", {}).get("metadata", {}).get("keyword_detection", {}).get("has_assistant_keywords")):
        count += 1
    
    # Check conversation branches
    for branch in sample.get("conversation_branches", []):
        for message in branch.get("messages", []):
            # Check direct content
            if (message.get("metadata", {}).get("keyword_detection", {}).get("has_assistant_keywords")):
                count += 1
            
            # Check parts
            for part in message.get("parts", []):
                if (part.get("metadata", {}).get("keyword_detection", {}).get("has_assistant_keywords")):
                    count += 1
    
    return count


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


def save_dataset_and_metadata(dataset, output_path: Path, input_path: Path, stats: Dict[str, int], keywords: List[str]):
    """Save annotated dataset with metadata."""
    # Ensure output directory exists
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save dataset
    print(f"Saving annotated dataset to {output_path}...")
    dataset.save_to_disk(str(output_path))
    
    # Load or create metadata
    metadata = load_existing_metadata(input_path) or {}
    
    # Add processing log entry
    processing_entry = {
        "operation": "assistant_keyword_annotation",
        "script": "keyword_filter_newformat.py",
        "timestamp": datetime.now().isoformat(),
        "input_path": str(input_path),
        "output_path": str(output_path),
        "total_messages_processed": stats["total_messages"],
        "total_assistant_keyword_detections": stats["total_keywords_found"],
        "assistant_keywords_searched": keywords,
        "annotation_success": True,
        "format": "new_chat_format_with_parts"
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
        description="Add keyword detection to new chat format datasets (with parts structure)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  # Annotate new format dataset for AI model keywords
  venv/bin/python 05-annotations/keyword_filter_newformat.py data/02-standardised-newformat/dataset data/05-annotations/dataset-annotated
  
  # With custom parallel processing settings
  venv/bin/python 05-annotations/keyword_filter_newformat.py data/04-decontaminated-newformat/dataset data/05-annotations/dataset-annotated --num-proc 64 --batch-size 2000

Keywords searched: {', '.join(FILTER_KEYWORDS)}
        """
    )
    
    parser.add_argument(
        "input_path",
        help="Path to input new chat format dataset directory (with parts structure)"
    )
    parser.add_argument(
        "output_path", 
        help="Path for output filtered dataset"
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=None,
        help="Number of processes for parallel processing (default: auto-detect, max 128)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for parallel processing (default: 1000)"
    )
    
    return parser.parse_args()


def main():
    """Main filtering function."""
    args = parse_arguments()
    
    # Validate input path
    input_path = Path(args.input_path)
    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}")
        sys.exit(1)
    
    output_path = Path(args.output_path)
    
    # Display keyword configuration
    print(f"Keyword filtering with {len(FILTER_KEYWORDS)} keywords:")
    for i, keyword in enumerate(FILTER_KEYWORDS, 1):
        print(f"  {i:2d}. {keyword}")
    print()
    
    # Load dataset  
    print(f"Loading dataset from: {input_path}")
    try:
        dataset = load_from_disk(str(input_path))
        
        # Handle DatasetDict vs single Dataset
        if hasattr(dataset, 'keys'):
            available_splits = list(dataset.keys())
            print(f"Found DatasetDict with splits: {available_splits}")
            
            if 'train' in available_splits:
                dataset = dataset['train']
                print(f"Using 'train' split")
            else:
                first_split = available_splits[0]
                dataset = dataset[first_split]
                print(f"Using '{first_split}' split")
        
        print(f"Dataset size: {len(dataset):,} samples")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)
    
    # Annotate dataset using parallel processing
    print(f"Processing dataset with keyword detection...")
    
    annotated_dataset, stats = annotate_dataset(
        dataset, FILTER_KEYWORDS, args.num_proc, args.batch_size
    )
    
    # Save annotated dataset
    save_dataset_and_metadata(annotated_dataset, output_path, input_path, stats, FILTER_KEYWORDS)
    
    print(f"\nAssistant keyword annotation completed!")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Samples: {len(annotated_dataset):,}")
    print(f"Messages processed: {stats['total_messages']:,}")
    print(f"Assistant keyword detections: {stats['total_keywords_found']:,}")
    print(f"Keywords searched: {len(FILTER_KEYWORDS)}")


if __name__ == "__main__":
    main()