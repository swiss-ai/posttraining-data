#!/usr/bin/env python3
"""
Convert HuggingFace datasets to new standardized chat format with parts structure.

This script converts datasets from various formats (chat messages, ShareGPT, 
instruction-response, preference pairs) into the new unified chat format with
parts structure for function calls, thinking, and verifiable responses.
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


def extract_sample_id(row: Dict[str, Any], row_index: Optional[int] = None) -> Optional[str]:
    """Extract sample ID from row, with fallback to row index."""
    # Try common ID field names
    for id_field in ['id', 'sample_id', 'conversation_id', 'idx', 'index']:
        if id_field in row and row[id_field] is not None:
            return str(row[id_field])
    
    # Fallback to row index if provided
    if row_index is not None:
        return f"row_{row_index}"
    
    return None


def create_schema_compliant_part(part_type: str, content: str = "", metadata: Optional[Dict] = None, 
                                 name: str = "", args: str = "") -> Dict[str, Any]:
    """Create a schema-compliant part with all required fields for Arrow compatibility."""
    return {
        "type": part_type,
        "content": content,
        "metadata": metadata or {},
        "name": name,
        "args": args
    }


def validate_conversation_pattern(conversation_messages: List[Dict], conversation_id: str = "unknown") -> bool:
    """Validate conversation message pattern and return True if valid, False if should skip."""
    if not conversation_messages:
        return True  # Empty conversation is valid (prompt-only)
    
    try:
        # First message should be assistant (responding to initial user prompt)
        if conversation_messages[0]["role"] != "assistant":
            print(f"Warning: Skipping sample {conversation_id} - first message must be from assistant, got '{conversation_messages[0]['role']}'")
            return False
        
        # Check alternating pattern and valid roles
        for i in range(1, len(conversation_messages)):
            current_role = conversation_messages[i]["role"]
            previous_role = conversation_messages[i-1]["role"]
            
            # Check for valid roles
            if current_role not in ["user", "assistant"]:
                print(f"Warning: Skipping sample {conversation_id} - invalid role '{current_role}' at message {i}")
                return False
            
            # Check alternating pattern
            if current_role == previous_role:
                print(f"Warning: Skipping sample {conversation_id} - consecutive messages from same role '{current_role}' at messages {i-1}-{i}")
                return False
        
        return True
        
    except Exception as e:
        print(f"Warning: Skipping sample {conversation_id} - validation error: {e}")
        return False


def validate_standardized_sample(sample: Dict[str, Any]) -> bool:
    """Validate a sample in the standardized new chat format."""
    if not sample or sample is None:
        return False
    
    # Check for required fields
    if "conversation_id" not in sample:
        print(f"Warning: Sample missing conversation_id field")
        return False
    
    conversation_id = sample.get("conversation_id", "unknown")
    
    # Validate each conversation branch
    for i, branch in enumerate(sample.get("conversation_branches", [])):
        messages = branch.get("messages", [])
        branch_id = f"{conversation_id}_branch{i}" if len(sample.get("conversation_branches", [])) > 1 else conversation_id
        
        if not validate_conversation_pattern(messages, branch_id):
            return False
    
    return True


def generate_conversation_id(dataset_source: str, content: str, sample_id: Optional[str] = None) -> str:
    """Generate a globally unique conversation ID."""
    dataset_prefix = dataset_source.replace('/', '_').replace('-', '_')
    
    if sample_id:
        # Use provided ID + short content hash for verification
        content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()[:8]
        return f"{dataset_prefix}_{sample_id}_{content_hash}"
    else:
        # Fallback to longer content hash for uniqueness
        content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]
        return f"{dataset_prefix}_{content_hash}"


def convert_chat_messages(row: Dict[str, Any], dataset_source: str, row_index: Optional[int] = None) -> Dict[str, Any]:
    """
    Convert standard chat format (messages array with role/content) to new format with parts.
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
        content = msg.get("content", "")
        
        if not role:
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
            # Convert to parts structure with schema compliance
            message_parts = [create_schema_compliant_part("response", content, {})]
            
            conversation_messages.append({
                "role": role,
                "parts": message_parts
            })
    
    if initial_prompt is None:
        raise ValueError("No initial user prompt found")
    
    # Extract sample ID
    sample_id = extract_sample_id(row, row_index)
    
    # Generate conversation ID
    conversation_id = generate_conversation_id(dataset_source, initial_prompt["content"], sample_id)
    
    
    # Preserve original metadata (exclude messages field)
    original_metadata = {k: v for k, v in row.items() if k != "messages"}
    
    # Create single conversation branch
    conversation_branches = [{
        "messages": conversation_messages
    }]
    
    result = {
        "conversation_id": conversation_id,
        "dataset_source": dataset_source,
        "original_metadata": original_metadata,
        "system_prompt": system_prompt or {"content": "", "metadata": {}},
        "initial_prompt": initial_prompt,
        "available_functions": [],
        "conversation_branches": conversation_branches,
        "created_timestamp": datetime.now().isoformat()
    }
    
    return result


def convert_nemotron_format(row: Dict[str, Any], dataset_source: str, row_index: Optional[int] = None) -> Dict[str, Any]:
    """
    Convert Nemotron format (input array + output string + system_prompt) to new parts format.
    Used for: Llama-Nemotron-Post-Training-Dataset
    
    Format:
    - input: [{"role": "user", "content": "..."}] (list with single user message)
    - output: "..." (assistant response string)
    - system_prompt: "..." (optional system prompt)
    - Additional metadata: category, license, reasoning, generator, etc.
    """
    input_messages = row.get("input", [])
    output = row.get("output", "")
    system_prompt_content = row.get("system_prompt", "")
    
    if not input_messages or not isinstance(input_messages, list) or len(input_messages) == 0:
        raise ValueError("No valid input messages array found")
    
    # Extract user message from nested input structure
    user_msg_data = input_messages[0]
    if not isinstance(user_msg_data, dict) or "content" not in user_msg_data:
        raise ValueError("Invalid input message structure")
        
    user_content = user_msg_data.get("content", "")
    
    # Extract sample ID
    sample_id = extract_sample_id(row, row_index)
    
    # Generate conversation ID
    conversation_id = generate_conversation_id(dataset_source, user_content, sample_id)
    
    # Preserve original metadata (exclude input, output, system_prompt)
    original_metadata = {k: v for k, v in row.items() 
                        if k not in ["input", "output", "system_prompt"]}
    
    # Create initial user prompt
    initial_prompt = {
        "role": "user",
        "content": user_content,
        "metadata": {}
    }
    
    # Create conversation with assistant response using parts
    conversation_messages = [{
        "role": "assistant",
        "parts": [create_schema_compliant_part("response", output, {})]
    }]
    
    
    # Create single conversation branch
    conversation_branches = [{
        "messages": conversation_messages
    }]
    
    result = {
        "conversation_id": conversation_id,
        "dataset_source": dataset_source,
        "original_metadata": original_metadata,
        "system_prompt": {"content": system_prompt_content, "metadata": {}} if system_prompt_content else {"content": "", "metadata": {}},
        "initial_prompt": initial_prompt,
        "available_functions": [],
        "conversation_branches": conversation_branches,
        "created_timestamp": datetime.now().isoformat()
    }
    
    return result


def convert_sharegpt_format(row: Dict[str, Any], dataset_source: str, row_index: Optional[int] = None) -> Dict[str, Any]:
    """
    Convert ShareGPT format (conversations with from/value fields) to new parts format.
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
        value_content = msg.get("value", "")
        
        if not from_role:
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
            # Convert to parts structure with schema compliance
            message_parts = [create_schema_compliant_part("response", value_content, {})]
            
            conversation_messages.append({
                "role": role,
                "parts": message_parts
            })
    
    if initial_prompt is None:
        raise ValueError("No initial user prompt found")
    
    # Extract sample ID
    sample_id = extract_sample_id(row, row_index)
    
    # Generate conversation ID
    conversation_id = generate_conversation_id(dataset_source, initial_prompt["content"], sample_id)
    
    
    # Preserve original metadata (exclude conversations field)
    original_metadata = {k: v for k, v in row.items() if k != "conversations"}
    
    # Create single conversation branch
    conversation_branches = [{
        "messages": conversation_messages
    }]
    
    result = {
        "conversation_id": conversation_id,
        "dataset_source": dataset_source,
        "original_metadata": original_metadata,
        "system_prompt": system_prompt or {"content": "", "metadata": {}},
        "initial_prompt": initial_prompt,
        "available_functions": [],
        "conversation_branches": conversation_branches,
        "created_timestamp": datetime.now().isoformat()
    }
    
    return result


def convert_preference_format(row: Dict[str, Any], dataset_source: str, row_index: Optional[int] = None) -> Dict[str, Any]:
    """
    Convert preference format (prompt with chosen/rejected responses) to new parts format.
    Used for: DPO datasets, preference pairs, etc.
    """
    prompt = row.get("prompt", "")
    chosen = row.get("chosen", "")
    rejected = row.get("rejected", "")
    
    # Extract sample ID
    sample_id = extract_sample_id(row, row_index)
    
    # Generate conversation ID
    conversation_id = generate_conversation_id(dataset_source, prompt, sample_id)
    
    # Preserve original metadata (exclude prompt, chosen, rejected)
    original_metadata = {k: v for k, v in row.items() 
                        if k not in ["prompt", "chosen", "rejected"]}
    
    # Create initial user prompt
    initial_prompt = {
        "role": "user",
        "content": prompt,
        "metadata": {}
    }
    
    # Create conversation branches with parts structure (chosen first = most preferred)
    conversation_branches = [
        {
            "messages": [{
                "role": "assistant",
                "parts": [create_schema_compliant_part("response", chosen, {})]
            }]
        },
        {
            "messages": [{
                "role": "assistant", 
                "parts": [create_schema_compliant_part("response", rejected, {})]
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
        "system_prompt": {"content": "", "metadata": {}},
        "initial_prompt": initial_prompt,
        "available_functions": [],
        "conversation_branches": conversation_branches,
        "created_timestamp": datetime.now().isoformat()
    }


def convert_instruction_response(row: Dict[str, Any], dataset_source: str, row_index: Optional[int] = None) -> Dict[str, Any]:
    """
    Convert instruction-response format (instruction/input → output) to new parts format.
    Used for: alpaca-style datasets, etc.
    """
    # Combine instruction and input into user prompt
    instruction = row.get("instruction", "")
    input_text = row.get("input", "")
    output = row.get("output", "")
    
    # Create user prompt (combine instruction and input)
    user_content_parts = [instruction]
    if input_text:
        user_content_parts.append(input_text)
    user_content = "\n\n".join(user_content_parts)
    
    # Extract sample ID
    sample_id = extract_sample_id(row, row_index)
    
    # Generate conversation ID
    conversation_id = generate_conversation_id(dataset_source, user_content, sample_id)
    
    # Create initial user prompt
    initial_prompt = {
        "role": "user",
        "content": user_content,
        "metadata": {}
    }
    
    # Create single conversation branch with assistant response using parts
    conversation_branches = [{
        "messages": [{
            "role": "assistant",
            "parts": [create_schema_compliant_part("response", output, {})]
        }]
    }]
    
    # Preserve original metadata (exclude instruction, input, output)
    original_metadata = {k: v for k, v in row.items() 
                        if k not in ["instruction", "input", "output"]}
    
    return {
        "conversation_id": conversation_id,
        "dataset_source": dataset_source,
        "original_metadata": original_metadata,
        "system_prompt": {"content": "", "metadata": {}},
        "initial_prompt": initial_prompt,
        "available_functions": [],
        "conversation_branches": conversation_branches,
        "created_timestamp": datetime.now().isoformat()
    }


def convert_inputs_labels_format(row: Dict[str, Any], dataset_source: str, row_index: Optional[int] = None) -> Dict[str, Any]:
    """
    Convert inputs/labels format to new parts format.
    Used for: DataProvenanceInitiative Commercial-Flan-Collection datasets.
    """
    inputs = row.get("inputs", "")
    labels = row.get("labels", "")
    
    if not inputs or not labels:
        raise ValueError("Missing required fields: inputs or labels")
    
    # Extract sample ID
    sample_id = extract_sample_id(row, row_index)
    
    # Generate conversation ID
    conversation_id = generate_conversation_id(dataset_source, inputs, sample_id)
    
    # Create initial user prompt
    initial_prompt = {
        "role": "user",
        "content": inputs,
        "metadata": {}
    }
    
    # Create single conversation branch with assistant response using parts
    conversation_branches = [{
        "messages": [{
            "role": "assistant",
            "parts": [create_schema_compliant_part("response", labels, {})]
        }]
    }]
    
    # Preserve original metadata (exclude inputs, labels)
    original_metadata = {k: v for k, v in row.items() 
                        if k not in ["inputs", "labels"]}
    
    return {
        "conversation_id": conversation_id,
        "dataset_source": dataset_source,
        "original_metadata": original_metadata,
        "system_prompt": {"content": "", "metadata": {}},
        "initial_prompt": initial_prompt,
        "available_functions": [],
        "conversation_branches": conversation_branches,
        "created_timestamp": datetime.now().isoformat()
    }


def convert_input_output_format(row: Dict[str, Any], dataset_source: str, row_index: Optional[int] = None) -> Dict[str, Any]:
    """
    Convert input/output format to new parts format.
    Used for: muri-it and similar datasets with simple input->output structure.
    """
    input_text = row.get("input", "")
    output_text = row.get("output", "")
    
    if not input_text or not output_text:
        raise ValueError("Missing required fields: input or output")
    
    # Extract sample ID
    sample_id = extract_sample_id(row, row_index)
    
    # Generate conversation ID
    conversation_id = generate_conversation_id(dataset_source, input_text, sample_id)
    
    # Create initial user prompt
    initial_prompt = {
        "role": "user",
        "content": input_text,
        "metadata": {}
    }
    
    # Create single conversation branch with assistant response using parts
    conversation_branches = [{
        "messages": [{
            "role": "assistant",
            "parts": [create_schema_compliant_part("response", output_text, {})]
        }]
    }]
    
    # Preserve original metadata (exclude input, output)
    original_metadata = {k: v for k, v in row.items() 
                        if k not in ["input", "output"]}
    
    return {
        "conversation_id": conversation_id,
        "dataset_source": dataset_source,
        "original_metadata": original_metadata,
        "system_prompt": {"content": "", "metadata": {}},
        "initial_prompt": initial_prompt,
        "available_functions": [],
        "conversation_branches": conversation_branches,
        "created_timestamp": datetime.now().isoformat()
    }














def auto_detect_converter(dataset):
    """Auto-detect appropriate converter based on dataset structure."""
    # Get a sample row to inspect structure
    if hasattr(dataset, 'keys'):  # DatasetDict
        sample = dataset[list(dataset.keys())[0]][0]
    else:  # Single Dataset
        sample = dataset[0]
    
    # Check for common field patterns
    if "messages" in sample:
        return convert_chat_messages
    elif "conversations" in sample:
        return convert_sharegpt_format
    elif "input" in sample and "output" in sample and isinstance(sample["input"], list):
        return convert_nemotron_format
    elif "prompt" in sample and "chosen" in sample and "rejected" in sample:
        return convert_preference_format
    elif "instruction" in sample and "output" in sample:
        return convert_instruction_response
    elif "inputs" in sample and "labels" in sample:
        return convert_inputs_labels_format
    elif "input" in sample and "output" in sample:
        return convert_input_output_format
    else:
        raise ValueError(f"Cannot auto-detect format for sample with keys: {list(sample.keys())}")


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


def convert_dataset_simple(dataset, converter, dataset_source: str):
    """Convert dataset using simple dataset.map() approach with validation."""
    def convert_with_validation(example, idx):
        try:
            # Convert first
            converted = converter(example, dataset_source, idx)
            
            if converted is None:
                return None  # Return None for filtering
            
            # Check if conversion produced valid structure
            if not isinstance(converted, dict):
                print(f"Warning: Converter returned non-dict for sample {idx}: {type(converted)}")
                return None
            
            # Validate the standardized format
            try:
                if not validate_standardized_sample(converted):
                    return None  # Return None for filtering
            except Exception as validation_error:
                print(f"Warning: Validation failed for sample {idx}: {validation_error}")
                print(f"Sample keys: {list(converted.keys()) if isinstance(converted, dict) else 'Not a dict'}")
                return None
            
            return converted
            
        except Exception as e:
            print(f"Warning: Failed to convert sample {idx}: {e}")
            return None  # Return None for filtering
    
    print(f"Converting and validating {len(dataset):,} samples...")
    
    # Use dataset.map with enumeration for index
    converted = dataset.map(
        convert_with_validation,
        with_indices=True,
        desc="Converting",
        remove_columns=dataset.column_names  # Remove original columns
    )
    
    # Filter out None results (invalid/failed conversions)
    initial_count = len(converted)
    converted = converted.filter(lambda x: x is not None)
    final_count = len(converted)
    
    if initial_count > final_count:
        print(f"Filtered out {initial_count - final_count:,} invalid samples ({final_count:,} remain)")
    
    return converted


def save_dataset_and_metadata(dataset, output_path: Path, dataset_name: str, input_path: Path):
    """Save converted dataset and update metadata."""
    from datasets import DatasetDict
    
    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Always save as DatasetDict with 'train' split for pipeline compatibility
    if isinstance(dataset, DatasetDict):
        output_dataset = dataset
    else:
        output_dataset = DatasetDict({"train": dataset})
    
    # Save dataset
    print(f"Saving dataset to {output_path}...")
    output_dataset.save_to_disk(str(output_path))
    
    # Load or create metadata
    metadata = load_existing_metadata(input_path) or {}
    
    # Add processing log entry
    processing_entry = {
        "operation": "standardisation",
        "script": "convert_to_chat_newformat.py",
        "timestamp": datetime.now().isoformat(),
        "input_path": str(input_path),
        "output_path": str(output_path),
        "samples_processed": len(dataset) if not isinstance(dataset, DatasetDict) else sum(len(split) for split in dataset.values()),
        "conversion_success": True,
        "target_schema": "chat_format_newformat_v1.0"
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
        description="Convert HuggingFace datasets to new chat format with parts structure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python convert_to_chat_newformat.py ../data/01-hf-data/smoltalk ../data/02-standardised/smoltalk
  python convert_to_chat_newformat.py ../data/01-hf-data/tulu-3-sft-mixture ../data/02-standardised/tulu-newformat
        """
    )
    
    parser.add_argument(
        "input_path",
        help="Path to input HuggingFace dataset directory"
    )
    parser.add_argument(
        "output_path", 
        help="Path for output dataset directory"
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
    print(f"Input:  {input_path}")
    
    print(f"Output: {args.output_path}")
    
    # Load input dataset
    print(f"Loading dataset from: {input_path}")
    try:
        dataset = load_from_disk(str(input_path))
        
        # Auto-detect converter function
        try:
            converter = auto_detect_converter(dataset)
            print(f"Auto-detected format: {converter.__name__}")
        except ValueError as e:
            print(f"Error: {e}")
            print("Supported formats: messages, conversations, input+output (Nemotron), prompt+chosen+rejected, instruction+output, inputs+labels")
            sys.exit(1)
        
        # Handle DatasetDict vs single Dataset
        if hasattr(dataset, 'keys'):
            # DatasetDict - use 'train' split or first available split
            available_splits = list(dataset.keys())
            print(f"Found DatasetDict with splits: {available_splits}")
            
            if 'train' in available_splits:
                input_dataset = dataset['train']
                print(f"Using 'train' split with {len(input_dataset):,} samples")
            else:
                split_name = available_splits[0]
                input_dataset = dataset[split_name]
                print(f"Using '{split_name}' split with {len(input_dataset):,} samples")
        else:
            # Single Dataset
            input_dataset = dataset
            print(f"Loaded single dataset with {len(input_dataset):,} samples")
        
        # Convert dataset
        print(f"Converting using {converter.__name__}...")
        converted_dataset = convert_dataset_simple(input_dataset, converter, dataset_name)
        
        # Save output
        output_path = Path(args.output_path)
        save_dataset_and_metadata(converted_dataset, output_path, dataset_name, input_path)
        
        print(f"\n✅ Conversion complete!")
        print(f"Input:  {input_path}")
        print(f"Output: {output_path}")
        print(f"Samples: {len(converted_dataset):,}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()