#!/usr/bin/env python3
"""
Convert OLMO-2-0325-32b-preference-mix Dataset to New Chat Format (Prompts Only)

Extracts only the prompts from the OLMO preference dataset, discarding the chosen
and rejected responses. Converts to the unified new chat format with parts structure.

Original OLMO format:
{
  "chosen": [
    {"role": "user", "content": "prompt"},
    {"role": "assistant", "content": "chosen response"}
  ],
  "rejected": [
    {"role": "user", "content": "prompt"},
    {"role": "assistant", "content": "rejected response"}
  ],
  "chosen_model": "model_name",
  "rejected_model": "model_name",
  "id": "sample_id",
  "source": "data_source"
}

Converts to new chat format with only the prompt, no responses.
"""

import sys
import json
import argparse
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from datasets import load_from_disk, DatasetDict, Dataset
from tqdm import tqdm


def generate_conversation_id(dataset_source: str, content: str) -> str:
    """Generate a unique conversation ID based on dataset source and content."""
    content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()[:12]
    return f"{dataset_source.replace('/', '_').replace('-', '_')}_{content_hash}"


def extract_prompt_from_chosen(chosen: List[Dict[str, str]]) -> Optional[str]:
    """
    Extract the user prompt from the chosen conversation.
    
    Args:
        chosen: List of message dictionaries with 'role' and 'content'
        
    Returns:
        The user prompt content or None if not found
    """
    if not chosen or not isinstance(chosen, list):
        return None
    
    # Find the first user message
    for msg in chosen:
        if isinstance(msg, dict) and msg.get("role") == "user":
            return msg.get("content", "")
    
    return None


def convert_olmo_sample(sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Convert a single OLMO sample to new chat format (prompts only).
    
    Args:
        sample: Original OLMO format sample
        
    Returns:
        Converted sample in new chat format or None if conversion fails
    """
    try:
        dataset_source = "olmo-2-0325-32b-preference-mix"
        
        # Extract prompt from chosen field
        chosen = sample.get("chosen", [])
        prompt_text = extract_prompt_from_chosen(chosen)
        
        if not prompt_text:
            return None
        
        # Generate conversation ID
        conversation_id = generate_conversation_id(dataset_source, prompt_text)
        
        # Build original metadata (selective preservation)
        original_metadata = {}
        if "id" in sample:
            original_metadata["id"] = sample["id"]
        if "source" in sample:
            original_metadata["source"] = sample["source"]
        
        # Create the converted sample in new chat format
        converted = {
            "conversation_id": conversation_id,
            "dataset_source": dataset_source,
            "original_metadata": original_metadata,
            "system_prompt": {
                "content": "",
                "metadata": {}
            },
            "initial_prompt": {
                "role": "user",
                "content": prompt_text,
                "metadata": {}
            },
            "available_functions": [],
            "conversation_branches": [],  # No responses, prompts only
            "created_timestamp": datetime.now().isoformat()
        }
        
        return converted
        
    except Exception as e:
        print(f"Error converting sample: {e}")
        return None


def convert_dataset(dataset: Dataset) -> Dataset:
    """
    Convert entire dataset to new chat format.
    
    Args:
        dataset: Original OLMO dataset
        
    Returns:
        Converted dataset in new chat format
    """
    print(f"Converting {len(dataset)} samples...")
    
    # Convert samples with progress bar
    converted_samples = []
    skipped = 0
    
    for sample in tqdm(dataset, desc="Converting samples"):
        converted = convert_olmo_sample(sample)
        if converted:
            converted_samples.append(converted)
        else:
            skipped += 1
    
    if skipped > 0:
        print(f"Skipped {skipped} samples due to conversion errors")
    
    print(f"Successfully converted {len(converted_samples)} samples")
    
    # Create new dataset from converted samples
    if not converted_samples:
        raise ValueError("No samples were successfully converted")
    
    return Dataset.from_list(converted_samples)


def main():
    parser = argparse.ArgumentParser(
        description="Convert OLMO preference dataset to new chat format (prompts only)"
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to input OLMO dataset (load_from_disk compatible)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for converted dataset. If directory, will create subfolder with -promptsOnly suffix"
    )
    
    args = parser.parse_args()
    
    # Load the dataset
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}")
        sys.exit(1)
    
    print(f"Loading dataset from {input_path}...")
    dataset = load_from_disk(str(input_path))
    
    # Handle DatasetDict vs Dataset
    if isinstance(dataset, DatasetDict):
        if "train" in dataset:
            dataset_to_convert = dataset["train"]
            print(f"Using 'train' split with {len(dataset_to_convert)} samples")
        else:
            # Use the first available split
            split_name = list(dataset.keys())[0]
            dataset_to_convert = dataset[split_name]
            print(f"Using '{split_name}' split with {len(dataset_to_convert)} samples")
    else:
        dataset_to_convert = dataset
        print(f"Processing dataset with {len(dataset_to_convert)} samples")
    
    # Convert the dataset
    converted_dataset = convert_dataset(dataset_to_convert)
    
    # Determine output path
    output_path = Path(args.output)
    if output_path.is_dir() or not output_path.suffix:
        # If output is a directory, create subfolder with dataset name
        dataset_name = "olmo-2-0325-32b-preference-mix-promptsOnly"
        final_output_path = output_path / dataset_name
    else:
        final_output_path = output_path
    
    # Create output directory if needed
    final_output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as DatasetDict to maintain consistency
    dataset_dict = DatasetDict({
        "train": converted_dataset
    })
    
    print(f"Saving converted dataset to {final_output_path}...")
    dataset_dict.save_to_disk(str(final_output_path))
    
    print(f"✓ Successfully converted {len(converted_dataset)} samples")
    print(f"✓ Dataset saved to {final_output_path}")
    
    # Print sample for verification
    print("\nSample converted entry:")
    print(json.dumps(converted_dataset[0], indent=2, default=str)[:2000])


if __name__ == "__main__":
    main()