#!/usr/bin/env python3
"""
Convert APIGen-MT-5k Dataset to Standardized Chat Format

Converts APIGen-MT-5k dataset from its multi-turn conversation format to the unified
chat format with parts structure. The dataset contains multi-turn function calling 
conversations where users interact with an airline agent that has access to various tools.

Original APIGen format:
{
  "conversations": [
    {"from": "human", "value": "user message"},
    {"from": "gpt", "value": "assistant message"}, 
    {"from": "function_call", "value": '{"name": "func", "arguments": {...}}'},
    {"from": "observation", "value": "function result"},
    {"from": "gpt", "value": "assistant response"}
  ],
  "tools": '[{"name": "func", "description": "...", "parameters": {...}}]',
  "system": "# System prompt..."
}

Converts to new chat format with:
- Parts structure for all messages (user and assistant)
- Multi-turn conversation preservation
- Function calls as assistant parts with function-call/function-output types
- Proper conversation flow and turn alternation
"""

import sys
import json
import argparse
import hashlib
import copy
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from datasets import load_from_disk, DatasetDict, Dataset


def generate_conversation_id(dataset_source: str, content: str) -> str:
    """Generate a unique conversation ID based on dataset source and content."""
    content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()[:12]
    return f"{dataset_source.replace('/', '_').replace('-', '_')}_{content_hash}"


def create_unified_part(part_type: str, content: str = "", name: str = "", 
                       args: str = "", metadata: Dict = None) -> Dict[str, Any]:
    """
    Create a unified part with all required fields for Arrow schema compatibility.
    
    All parts must have identical field structure to prevent Arrow schema conflicts
    when merging datasets. Uses empty strings instead of None for unused fields.
    """
    return {
        "type": part_type,
        "content": content,
        "metadata": metadata or {},
        "name": name,
        "args": args
    }


def parse_function_call(msg_value: str) -> Tuple[str, str, str]:
    """
    Parse function_call message value to extract function name, args, and content.
    
    Args:
        msg_value: JSON string like '{"name": "func", "arguments": {...}}'
        
    Returns:
        (function_name, args_json_string, content_for_thought)
    """
    try:
        func_call = json.loads(msg_value)
        func_name = func_call.get("name", "")
        func_args = func_call.get("arguments", {})
        
        # Special handling for think function
        if func_name == "think":
            thought_content = func_args.get("thought", "")
            return "think", "", thought_content
        
        # Regular function call
        args_json = json.dumps(func_args) if func_args else ""
        return func_name, args_json, ""
        
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Warning: Failed to parse function call: {msg_value[:100]}... Error: {e}")
        return "", "", ""


def parse_conversation_turns(conversations: List[Dict[str, str]]) -> List[List[Dict[str, str]]]:
    """
    Split APIGen conversation list into logical message groups.
    
    Groups consecutive assistant-related messages (gpt, function_call, observation)
    into single assistant turns, while keeping human messages as separate user turns.
    
    Returns list of message groups, each representing one user or assistant turn.
    """
    if not conversations:
        return []
    
    groups = []
    current_group = []
    current_role = None
    
    for msg in conversations:
        msg_type = msg.get("from", "")
        
        if msg_type == "human":
            # Start new user turn
            if current_group:
                groups.append(current_group)
            current_group = [msg]
            current_role = "user"
            
        elif msg_type in ["gpt", "function_call", "observation"]:
            if current_role == "user":
                # Transition from user to assistant - start new assistant group
                if current_group:
                    groups.append(current_group)
                current_group = [msg]
                current_role = "assistant"
            else:
                # Continue assistant group
                current_group.append(msg)
        else:
            print(f"Warning: Unknown message type '{msg_type}', adding to current group")
            current_group.append(msg)
    
    # Add final group
    if current_group:
        groups.append(current_group)
    
    return groups


def create_user_message(human_msgs: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Convert human message(s) to user message with response parts.
    
    Typically expects single human message, but handles multiple if present.
    """
    parts = []
    
    for msg in human_msgs:
        if msg.get("from") == "human":
            content = msg.get("value", "")
            part = create_unified_part("response", content=content)
            parts.append(part)
    
    return {
        "role": "user",
        "parts": parts
    }


def create_assistant_message(assistant_msgs: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Convert mixed gpt/function_call/observation sequence to assistant message with parts.
    
    Maps APIGen message types to parts:
    - gpt → response parts
    - function_call → function-call parts (or thought parts for think function)
    - observation → function-output parts
    """
    parts = []
    
    for msg in assistant_msgs:
        msg_type = msg.get("from", "")
        msg_value = msg.get("value", "")
        
        if msg_type == "gpt":
            # Regular assistant response
            part = create_unified_part("response", content=msg_value)
            parts.append(part)
            
        elif msg_type == "function_call":
            # Parse function call
            func_name, args_json, thought_content = parse_function_call(msg_value)
            
            if func_name == "think":
                # Special think function becomes thought part
                part = create_unified_part("thought", content=thought_content)
            elif func_name:
                # Regular function call
                part = create_unified_part("function-call", name=func_name, args=args_json)
            else:
                # Failed to parse, skip
                continue
                
            parts.append(part)
            
        elif msg_type == "observation":
            # Function execution result
            part = create_unified_part("function-output", content=msg_value)
            parts.append(part)
    
    return {
        "role": "assistant", 
        "parts": parts
    }


def convert_apigen_tools(tools_json_str: str) -> List[Dict[str, Any]]:
    """
    Convert APIGen tools JSON string to available_functions list.
    
    APIGen tools are already in OpenAI-compatible format, so this is mostly
    parsing and validation.
    """
    try:
        tools = json.loads(tools_json_str)
    except json.JSONDecodeError as e:
        print(f"Warning: Failed to parse tools JSON: {e}")
        return []
    
    if not isinstance(tools, list):
        print(f"Warning: Tools is not a list: {type(tools)}")
        return []
    
    # Filter out any malformed tools and ensure required fields
    valid_tools = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
            
        name = tool.get("name", "").strip()
        if not name:
            continue
            
        # Ensure tool has required structure
        valid_tool = {
            "name": name,
            "description": tool.get("description", ""),
            "parameters": tool.get("parameters", {
                "type": "object",
                "properties": {},
                "required": []
            })
        }
        valid_tools.append(valid_tool)
    
    return valid_tools


def convert_apigen_sample(sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert a single APIGen sample to new chat format."""
    try:
        dataset_source = "apigen-mt-5k"
        conversations = sample.get("conversations", [])
        
        if not conversations:
            return None
        
        # Generate conversation ID from first user message
        first_human_msg = next((msg.get("value", "") for msg in conversations 
                              if msg.get("from") == "human"), "")
        conversation_id = generate_conversation_id(dataset_source, first_human_msg)
        
        # Parse tools
        tools_str = sample.get("tools", "[]")
        available_functions = convert_apigen_tools(tools_str)
        
        # Get system prompt
        system_content = sample.get("system", "")
        
        # Parse conversation into turns
        turn_groups = parse_conversation_turns(conversations)
        if not turn_groups:
            return None
        
        # Convert each turn group to messages
        messages = []
        for group in turn_groups:
            if not group:
                continue
                
            first_msg_type = group[0].get("from", "")
            
            if first_msg_type == "human":
                # User message
                user_msg = create_user_message(group)
                messages.append(user_msg)
            else:
                # Assistant message 
                assistant_msg = create_assistant_message(group)
                if assistant_msg["parts"]:  # Only add if has parts
                    messages.append(assistant_msg)
        
        if not messages:
            return None
        
        # Extract initial prompt (should be first user message)
        initial_prompt_content = ""
        if messages and messages[0]["role"] == "user" and messages[0]["parts"]:
            initial_prompt_content = messages[0]["parts"][0].get("content", "")
        
        # Build final chat format
        chat_sample = {
            "conversation_id": conversation_id,
            "dataset_source": dataset_source,
            "original_metadata": {},
            "system_prompt": {
                "content": system_content,
                "metadata": {}
            },
            "initial_prompt": {
                "role": "user",
                "content": initial_prompt_content,
                "metadata": {}
            },
            "available_functions": available_functions,
            "conversation_branches": [
                {
                    "messages": messages
                }
            ],
            "created_timestamp": datetime.now().isoformat()
        }
        
        return chat_sample
        
    except Exception as e:
        print(f"Error converting sample: {e}")
        return None


def process_dataset(dataset: Dataset, chunk_size: int = 1000) -> Dataset:
    """Process APIGen dataset to convert to chat format with chunked processing."""
    print(f"Converting {len(dataset):,} samples in chunks of {chunk_size:,}...")
    
    converted_rows = []
    failed_count = 0
    total_rows = len(dataset)
    
    # Use tqdm for progress bar
    from tqdm import tqdm
    with tqdm(total=total_rows, desc="Converting APIGen samples", unit="samples") as pbar:
        for chunk_start in range(0, total_rows, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_rows)
            chunk_rows = []
            
            # Process chunk
            for idx in range(chunk_start, chunk_end):
                try:
                    sample = dataset[idx]
                    converted_sample = convert_apigen_sample(sample)
                    if converted_sample:
                        chunk_rows.append(converted_sample)
                    else:
                        failed_count += 1
                except Exception as e:
                    print(f"Error processing sample {idx}: {e}")
                    failed_count += 1
                
                # Update progress bar
                pbar.update(1)
                if failed_count > 0:
                    pbar.set_postfix(failed=failed_count)
            
            # Add chunk to results
            converted_rows.extend(chunk_rows)
            
            # Force garbage collection after each chunk
            import gc
            gc.collect()
    
    print(f"Conversion complete: {len(converted_rows):,} successful, {failed_count} failed")
    
    # Create new dataset from converted rows
    if not converted_rows:
        return Dataset.from_list([])
    
    print("Creating Dataset with explicit schema...")
    
    # Critical: Use deepcopy to avoid Arrow corruption
    # NEVER modify input data in-place with dataset.map()
    processed_rows = copy.deepcopy(converted_rows)
    
    try:
        return Dataset.from_list(processed_rows)
    except Exception as e:
        print(f"Dataset creation failed: {e}")
        return None


def save_dataset_and_metadata(dataset_dict: DatasetDict, output_path: Path, 
                             input_path: Path, args: argparse.Namespace):
    """Save converted dataset with processing metadata."""
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save dataset
    dataset_dict.save_to_disk(str(output_path))
    
    # Create processing metadata
    metadata = {
        "processing_log": [
            {
                "operation": "convert_apigen_function_calling",
                "script": "convert_apigen_function_calling.py",
                "timestamp": datetime.now().isoformat(),
                "input_path": str(input_path),
                "output_path": str(output_path),
                "chunk_size": args.batch_size,
                "description": "Converted APIGen-MT-5k dataset to standardized chat format with multi-turn conversations and function calling support"
            }
        ],
        "format": "chat_format_v1_new",
        "source_dataset": "apigen-mt-5k",
        "conversion_details": {
            "function_format": "openai_compatible", 
            "conversation_type": "multi_turn_function_calling",
            "parts_structure": "unified_schema",
            "special_features": ["multi_turn", "function_calling", "think_reasoning"]
        }
    }
    
    # Save metadata
    metadata_file = output_path / "dataset_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Dataset saved to {output_path}")
    print(f"Metadata saved to {metadata_file}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert APIGen-MT-5k dataset to standardized chat format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert APIGen dataset to chat format
  venv/bin/python 02-standardisation/convert_apigen_function_calling.py \\
    /capstor/store/cscs/swissai/infra01/posttrain_data/01_raw_hf_data/APIGen-MT-5k \\
    --output data/02-standardised/apigen-mt-5k
  
  # With custom chunk size
  venv/bin/python 02-standardisation/convert_apigen_function_calling.py \\
    /path/to/apigen/dataset \\
    --output /path/to/output \\
    --batch-size 500
        """
    )
    
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to APIGen dataset directory (HuggingFace format)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output directory for converted dataset"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Chunk size for processing (default: 1000)"
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
    
    # Setup output path - if output is a directory, append input folder name
    output_path = Path(args.output)
    input_folder_name = input_path.name
    
    # Check if output_arg ends with '/' or is clearly a directory
    if args.output.endswith('/') or args.output.endswith('\\'):
        # It's a folder path, append input folder name
        output_path = output_path / input_folder_name
    elif output_path.exists() and output_path.is_dir():
        # Existing directory without trailing slash
        output_path = output_path / input_folder_name
    # Otherwise use the path as-is (specific name provided)
    
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    
    # Load dataset
    try:
        dataset = load_from_disk(str(input_path))
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)
    
    # Ensure we have a DatasetDict
    if not isinstance(dataset, DatasetDict):
        dataset = DatasetDict({"train": dataset})
    
    # Process each split
    processed_splits = {}
    for split_name, split_dataset in dataset.items():
        print(f"Processing {split_name} split: {len(split_dataset):,} samples")
        processed_splits[split_name] = process_dataset(
            split_dataset,
            chunk_size=args.batch_size
        )
        
        if processed_splits[split_name] is None:
            print(f"Failed to process {split_name} split")
            sys.exit(1)
    
    processed_dataset = DatasetDict(processed_splits)
    
    # Save results
    save_dataset_and_metadata(processed_dataset, output_path, input_path, args)
    
    # Print summary
    total_samples = sum(len(split) for split in processed_dataset.values())
    print(f"\nConversion completed successfully!")
    print(f"Total samples processed: {total_samples:,}")
    print(f"Output format: Standardized chat format with multi-turn conversations and function calling")
    
    # Show sample statistics
    if "train" in processed_dataset:
        train_split = processed_dataset["train"]
        print(f"\nDataset structure:")
        print(f"- Split: train")
        print(f"- Samples: {len(train_split):,}")
        print(f"- Features: {list(train_split.features.keys())}")


if __name__ == "__main__":
    main()