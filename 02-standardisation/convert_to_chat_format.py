#!/usr/bin/env python3
"""
Convert HuggingFace datasets to standardized chat format.

This script converts datasets from various formats (chat messages, ShareGPT, 
instruction-response, preference pairs) into a unified chat format suitable 
for downstream processing.
"""

import sys
import json
import argparse
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
from datasets import load_from_disk
from tqdm import tqdm


def generate_conversation_id(dataset_source: str, content: str) -> str:
    """Generate a unique conversation ID based on dataset source and content."""
    content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()[:12]
    return f"{dataset_source.replace('/', '_').replace('-', '_')}_{content_hash}"


def convert_chat_messages(row: Dict[str, Any], dataset_source: str) -> Dict[str, Any]:
    """
    Convert standard chat format (messages array with role/content).
    Used for: smoltalk, tulu-3-sft-mixture, etc.
    """
    messages = row.get("messages", [])
    if not messages or not isinstance(messages, list):
        raise ValueError("No valid messages array found")
    
    # Parse messages if stored as JSON string
    if isinstance(messages, str):
        messages = json.loads(messages)
    
    # Find system prompt, initial user prompt, and conversation messages
    system_prompt = None
    initial_prompt = None
    conversation_messages = []
    
    for msg in messages:
        if not isinstance(msg, dict):
            continue
            
        role = msg.get("role", "").strip()
        content = msg.get("content", "").strip()
        
        if not role or not content:
            continue
        
        # Identify message types
        if role == "system" and system_prompt is None:
            system_prompt = {
                "content": content,
                "metadata": {}
            }
        elif role == "user" and initial_prompt is None:
            initial_prompt = {
                "role": "user",
                "content": content,
                "metadata": {}
            }
        else:
            # Part of conversation (assistant, user follow-ups, etc.)
            conversation_messages.append({
                "role": role,
                "content": content,
                "metadata": {}
            })
    
    if initial_prompt is None:
        raise ValueError("No initial user prompt found")
    
    # Generate conversation ID
    conversation_id = generate_conversation_id(dataset_source, initial_prompt["content"])
    
    # Create single conversation branch
    conversation_branches = [{
        "messages": conversation_messages
    }]
    
    # Preserve original metadata (exclude messages field)
    original_metadata = {k: v for k, v in row.items() if k != "messages"}
    
    result = {
        "conversation_id": conversation_id,
        "dataset_source": dataset_source,
        "original_metadata": original_metadata,
        "initial_prompt": initial_prompt,
        "conversation_branches": conversation_branches,
        "created_timestamp": datetime.now().isoformat()
    }
    
    if system_prompt:
        result["system_prompt"] = system_prompt
    
    return result


def convert_nemotron_format(row: Dict[str, Any], dataset_source: str) -> Dict[str, Any]:
    """
    Convert Nemotron format (input array + output string + system_prompt).
    Used for: Llama-Nemotron-Post-Training-Dataset
    
    Format:
    - input: [{"role": "user", "content": "..."}] (list with single user message)
    - output: "..." (assistant response string)
    - system_prompt: "..." (optional system prompt)
    - Additional metadata: category, license, reasoning, generator, etc.
    """
    input_messages = row.get("input", [])
    output = row.get("output", "").strip()
    system_prompt_content = row.get("system_prompt", "").strip()
    
    if not input_messages or not isinstance(input_messages, list) or len(input_messages) == 0:
        raise ValueError("No valid input messages array found")
    if not output:
        raise ValueError("No output found")
    
    # Extract user message from nested input structure
    user_msg_data = input_messages[0]
    if not isinstance(user_msg_data, dict) or "content" not in user_msg_data:
        raise ValueError("Invalid input message structure")
        
    user_content = user_msg_data.get("content", "").strip()
    if not user_content:
        raise ValueError("No user content found")
    
    # Generate conversation ID
    conversation_id = generate_conversation_id(dataset_source, user_content)
    
    # Create initial user prompt
    initial_prompt = {
        "role": "user",
        "content": user_content,
        "metadata": {}
    }
    
    # Create conversation with assistant response
    conversation_messages = [{
        "role": "assistant",
        "content": output,
        "metadata": {}
    }]
    
    # Create single conversation branch
    conversation_branches = [{
        "messages": conversation_messages
    }]
    
    # Preserve original metadata (exclude input, output, system_prompt)
    original_metadata = {k: v for k, v in row.items() 
                        if k not in ["input", "output", "system_prompt"]}
    
    result = {
        "conversation_id": conversation_id,
        "dataset_source": dataset_source,
        "original_metadata": original_metadata,
        "initial_prompt": initial_prompt,
        "conversation_branches": conversation_branches,
        "created_timestamp": datetime.now().isoformat()
    }
    
    # Add system prompt if present
    if system_prompt_content:
        result["system_prompt"] = {
            "content": system_prompt_content,
            "metadata": {}
        }
    
    return result


def convert_sharegpt_format(row: Dict[str, Any], dataset_source: str) -> Dict[str, Any]:
    """
    Convert ShareGPT format (conversations with from/value fields).
    Used for: The-Tome, EuroBlocks-SFT-Synthetic-1124, etc.
    """
    conversations = row.get("conversations", [])
    if not conversations or not isinstance(conversations, list):
        raise ValueError("No valid conversations array found")
    
    # Role mapping for ShareGPT format
    role_map = {"human": "user", "gpt": "assistant"}
    
    # Find system prompt, initial user prompt, and conversation messages
    system_prompt = None
    initial_prompt = None
    conversation_messages = []
    
    for msg in conversations:
        if not isinstance(msg, dict):
            continue
            
        from_role = msg.get("from", "").strip()
        value_content = msg.get("value", "").strip()
        
        if not from_role or not value_content:
            continue
        
        # Map role
        role = role_map.get(from_role, from_role)
        
        # Identify message types
        if role == "system" and system_prompt is None:
            system_prompt = {
                "content": value_content,
                "metadata": {}
            }
        elif role == "user" and initial_prompt is None:
            initial_prompt = {
                "role": "user",
                "content": value_content,
                "metadata": {}
            }
        else:
            # Part of conversation
            conversation_messages.append({
                "role": role,
                "content": value_content,
                "metadata": {}
            })
    
    if initial_prompt is None:
        raise ValueError("No initial user prompt found")
    
    # Generate conversation ID
    conversation_id = generate_conversation_id(dataset_source, initial_prompt["content"])
    
    # Create single conversation branch
    conversation_branches = [{
        "messages": conversation_messages
    }]
    
    # Preserve original metadata (exclude conversations field)
    original_metadata = {k: v for k, v in row.items() if k != "conversations"}
    
    result = {
        "conversation_id": conversation_id,
        "dataset_source": dataset_source,
        "original_metadata": original_metadata,
        "initial_prompt": initial_prompt,
        "conversation_branches": conversation_branches,
        "created_timestamp": datetime.now().isoformat()
    }
    
    if system_prompt:
        result["system_prompt"] = system_prompt
    
    return result


def convert_preference_format(row: Dict[str, Any], dataset_source: str) -> Dict[str, Any]:
    """
    Convert preference format (prompt with chosen/rejected responses).
    Used for: DPO datasets, preference pairs, etc.
    """
    prompt = row.get("prompt", "").strip()
    chosen = row.get("chosen", "").strip()
    rejected = row.get("rejected", "").strip()
    
    if not prompt:
        raise ValueError("No prompt found")
    if not chosen or not rejected:
        raise ValueError("Missing chosen or rejected response")
    
    # Generate conversation ID
    conversation_id = generate_conversation_id(dataset_source, prompt)
    
    # Create initial user prompt
    initial_prompt = {
        "role": "user",
        "content": prompt,
        "metadata": {}
    }
    
    # Create conversation branches (chosen first = most preferred)
    conversation_branches = [
        {
            "messages": [{
                "role": "assistant",
                "content": chosen,
                "metadata": {}
            }]
        },
        {
            "messages": [{
                "role": "assistant", 
                "content": rejected,
                "metadata": {}
            }]
        }
    ]
    
    # Preserve original metadata (exclude prompt, chosen, rejected)
    original_metadata = {k: v for k, v in row.items() 
                        if k not in ["prompt", "chosen", "rejected"]}
    
    return {
        "conversation_id": conversation_id,
        "dataset_source": dataset_source,
        "original_metadata": original_metadata,
        "initial_prompt": initial_prompt,
        "conversation_branches": conversation_branches,
        "created_timestamp": datetime.now().isoformat()
    }


def convert_instruction_response(row: Dict[str, Any], dataset_source: str) -> Dict[str, Any]:
    """
    Convert instruction-response format (instruction/input → output).
    Used for: alpaca-style datasets, etc.
    """
    # Combine instruction and input into user prompt
    instruction = row.get("instruction", "").strip()
    input_text = row.get("input", "").strip()
    output = row.get("output", "").strip()
    
    if not instruction:
        raise ValueError("No instruction found")
    if not output:
        raise ValueError("No output found")
    
    # Create user prompt (combine instruction and input)
    user_content_parts = [instruction]
    if input_text:
        user_content_parts.append(input_text)
    user_content = "\n\n".join(user_content_parts)
    
    # Generate conversation ID
    conversation_id = generate_conversation_id(dataset_source, user_content)
    
    # Create initial user prompt
    initial_prompt = {
        "role": "user",
        "content": user_content,
        "metadata": {}
    }
    
    # Create single conversation branch with assistant response
    conversation_branches = [{
        "messages": [{
            "role": "assistant",
            "content": output,
            "metadata": {}
        }]
    }]
    
    # Preserve original metadata (exclude instruction, input, output)
    original_metadata = {k: v for k, v in row.items() 
                        if k not in ["instruction", "input", "output"]}
    
    return {
        "conversation_id": conversation_id,
        "dataset_source": dataset_source,
        "original_metadata": original_metadata,
        "initial_prompt": initial_prompt,
        "conversation_branches": conversation_branches,
        "created_timestamp": datetime.now().isoformat()
    }


# Converter registry - maps dataset names to converter functions
CONVERTERS = {
    # Standard chat format with messages array
    "smoltalk": convert_chat_messages,
    "smoltalk2": convert_chat_messages, 
    "tulu-3-sft-mixture": convert_chat_messages,
    
    # ShareGPT format with from/value fields
    "The-Tome": convert_sharegpt_format,
    "EuroBlocks-SFT-Synthetic-1124": convert_sharegpt_format,
    
    # Nemotron format (input array + output string + system_prompt)
    "Llama-Nemotron-Post-Training-Dataset": convert_nemotron_format,
    
    # Add more as needed - typically 1 line each
}


def get_converter(dataset_name: str):
    """Get appropriate converter function for dataset."""
    return CONVERTERS.get(dataset_name)


def convert_dataset(dataset, converter, dataset_source: str):
    """Convert entire dataset using specified converter."""
    
    def convert_row(row, idx):
        try:
            return converter(row, dataset_source)
        except Exception as e:
            tqdm.write(f"Warning: Failed to convert row {idx}: {e}")
            return None
    
    # Convert dataset and filter out failed conversions
    print("Converting rows...")
    converted_rows = []
    failed_count = 0
    
    # Use tqdm for progress bar
    with tqdm(total=len(dataset), desc="Converting", unit="rows") as pbar:
        for idx, row in enumerate(dataset):
            converted_row = convert_row(row, idx)
            if converted_row:
                converted_rows.append(converted_row)
            else:
                failed_count += 1
            
            # Update progress bar
            pbar.update(1)
            if failed_count > 0:
                pbar.set_postfix(failed=failed_count)
    
    print(f"Conversion complete: {len(converted_rows)} successful, {failed_count} failed")
    
    # Create new dataset from converted rows
    from datasets import Dataset
    return Dataset.from_list(converted_rows)


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


def save_dataset_and_metadata(dataset, output_path: Path, dataset_name: str, input_path: Path, split_name: str = None):
    """Save converted dataset and update metadata."""
    # Ensure output directory exists
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save dataset
    print(f"Saving dataset to {output_path}...")
    dataset.save_to_disk(str(output_path))
    
    # Load or create metadata
    metadata = load_existing_metadata(input_path) or {}
    
    # Add processing log entry
    processing_entry = {
        "operation": "standardisation",
        "script": "convert_to_chat_format.py",
        "timestamp": datetime.now().isoformat(),
        "input_path": str(input_path),
        "output_path": str(output_path),
        "samples_processed": len(dataset),
        "conversion_success": True,
        "target_schema": "chat_format_v1.0"
    }
    
    # Add split information if provided
    if split_name:
        processing_entry["split_name"] = split_name
    
    if "processing_log" not in metadata:
        metadata["processing_log"] = []
    metadata["processing_log"].append(processing_entry)
    
    # Save updated metadata
    metadata_file = output_path / "dataset_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {metadata_file}")


def determine_output_path(output_arg: str, dataset_name: str, split_name: str = None) -> Path:
    """Determine the final output path."""
    output_path = Path(output_arg)
    
    # If we have a split name, create nested structure
    if split_name:
        return output_path / dataset_name / split_name
    
    # If output_arg ends with dataset name, use it directly
    # Otherwise, append dataset name
    if output_path.name == dataset_name:
        return output_path
    else:
        return output_path / dataset_name


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace datasets to standardized chat format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python convert_to_chat_format.py ../data/01-hf-data/smoltalk ../data/02-standardised/smoltalk
  python convert_to_chat_format.py ../data/01-hf-data/tulu-3-sft-mixture ../data/02-standardised/
  python convert_to_chat_format.py ../data/01-hf-data/The-Tome ../data/02-standardised/The-Tome
        """
    )
    
    parser.add_argument(
        "input_path",
        help="Path to input HuggingFace dataset directory"
    )
    parser.add_argument(
        "output_path", 
        help="Path for output dataset (directory name or parent directory)"
    )
    
    return parser.parse_args()


def main():
    """Main conversion function."""
    args = parse_arguments()
    
    # Validate input path
    input_path = Path(args.input_path)
    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}")
        sys.exit(1)
    
    # Get dataset name from input path
    dataset_name = input_path.name
    print(f"Converting dataset: {dataset_name}")
    
    # Get converter function
    converter = get_converter(dataset_name)
    if not converter:
        print(f"Error: No converter found for dataset: {dataset_name}")
        print(f"Supported datasets: {list(CONVERTERS.keys())}")
        sys.exit(1)
    
    # Load input dataset
    print(f"Loading dataset from: {input_path}")
    try:
        dataset = load_from_disk(str(input_path))
        
        # Handle DatasetDict (has multiple splits) vs single Dataset
        if hasattr(dataset, 'keys'):
            # DatasetDict - process all splits and preserve structure
            available_splits = list(dataset.keys())
            print(f"Found DatasetDict with {len(available_splits)} splits: {available_splits}")
            
            converted_splits = {}
            total_samples_processed = 0
            
            for split_name in available_splits:
                print(f"\n{'='*60}")
                print(f"Processing split: {split_name}")
                print('='*60)
                
                input_dataset = dataset[split_name]
                print(f"Split '{split_name}' has {len(input_dataset)} samples")
                
                # Convert this split
                try:
                    converted_dataset = convert_dataset(input_dataset, converter, dataset_name)
                    converted_splits[split_name] = converted_dataset
                    total_samples_processed += len(converted_dataset)
                    print(f"✅ Split '{split_name}' converted: {len(converted_dataset)} samples")
                except Exception as e:
                    print(f"Error converting split '{split_name}': {e}")
                    continue
            
            if converted_splits:
                # Reconstruct DatasetDict with converted splits
                from datasets import DatasetDict
                converted_datasetdict = DatasetDict(converted_splits)
                
                # Determine output path (no split name for DatasetDict)
                output_path = determine_output_path(args.output_path, dataset_name)
                
                # Save the entire DatasetDict
                try:
                    save_dataset_and_metadata(converted_datasetdict, output_path, dataset_name, input_path)
                    print(f"\n{'='*60}")
                    print(f"✅ DatasetDict conversion complete!")
                    print(f"Dataset: {dataset_name}")
                    print(f"Total splits: {len(converted_splits)}")
                    print(f"Total samples processed: {total_samples_processed}")
                    print(f"Output directory: {output_path}")
                    
                    for split_name, split_dataset in converted_splits.items():
                        print(f"  - {split_name}: {len(split_dataset):,} samples")
                except Exception as e:
                    print(f"Error saving DatasetDict: {e}")
                    sys.exit(1)
            else:
                print("❌ No splits were successfully converted")
                sys.exit(1)
        else:
            # Single Dataset - process as before
            input_dataset = dataset
            print(f"Loaded single dataset with {len(input_dataset)} samples")
            
            # Convert dataset
            try:
                converted_dataset = convert_dataset(input_dataset, converter, dataset_name)
            except Exception as e:
                print(f"Error during conversion: {e}")
                sys.exit(1)
            
            # Determine output path
            output_path = determine_output_path(args.output_path, dataset_name)
            
            # Save output
            try:
                save_dataset_and_metadata(converted_dataset, output_path, dataset_name, input_path)
                print(f"\n✅ Conversion complete!")
                print(f"Input:  {input_path}")
                print(f"Output: {output_path}")
                print(f"Samples: {len(converted_dataset)}")
            except Exception as e:
                print(f"Error saving dataset: {e}")
                sys.exit(1)
            
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()