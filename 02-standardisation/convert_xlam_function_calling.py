#!/usr/bin/env python3
"""
Convert XLAM Function Calling Dataset to Standardized Chat Format

Converts xlam-function-calling-60k dataset from its custom format to the unified
chat format with OpenAI-compatible function definitions. The dataset contains
single-turn function calling examples where users ask questions and the model
should respond with appropriate function calls.

Original XLAM format:
{
  "id": 0,
  "query": "user question",
  "answers": "[{\"name\": \"func\", \"arguments\": {...}}]",
  "tools": "[{\"name\": \"func\", \"description\": \"...\", \"parameters\": {...}}]"
}

Converts to new chat format with:
- OpenAI-standard function definitions in available_functions
- Function calls as assistant parts with function-call type
- Proper conversation structure
"""

import sys
import json
import argparse
import hashlib
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from datasets import load_from_disk, DatasetDict, Dataset


def generate_conversation_id(dataset_source: str, content: str) -> str:
    """Generate a unique conversation ID based on dataset source and content."""
    content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()[:12]
    return f"{dataset_source.replace('/', '_').replace('-', '_')}_{content_hash}"


def parse_xlam_type(type_str: str) -> tuple[str, bool]:
    """
    Parse XLAM type string to OpenAI format.
    
    Args:
        type_str: XLAM type like "str", "int", "str, optional", "List[str]"
    
    Returns:
        (openai_type, is_required)
    """
    # Handle optional types
    if ", optional" in type_str:
        base_type = type_str.replace(", optional", "").strip()
        is_required = False
    else:
        base_type = type_str.strip()
        is_required = True
    
    # Handle complex types like List[...], Dict[...], etc.
    if base_type.startswith(('List[', 'list[')):
        return "array", is_required
    elif base_type.startswith(('Dict[', 'dict[', 'Mapping[')):
        return "object", is_required
    
    # Convert basic XLAM types to OpenAI/JSON Schema types
    type_mapping = {
        "str": "string",
        "string": "string",
        "int": "integer", 
        "integer": "integer",
        "float": "number",
        "number": "number",
        "bool": "boolean",
        "boolean": "boolean"
    }
    
    openai_type = type_mapping.get(base_type, "string")
    return openai_type, is_required


def convert_xlam_parameters(xlam_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert XLAM parameter format to OpenAI function calling standard.
    
    XLAM format:
    {
      "param1": {"type": "str", "description": "...", "default": "..."},
      "param2": {"type": "int, optional", "description": "..."}
    }
    
    OpenAI format:
    {
      "type": "object",
      "properties": {
        "param1": {"type": "string", "description": "...", "default": "..."},
        "param2": {"type": "integer", "description": "..."}
      },
      "required": ["param1"]
    }
    """
    if not xlam_params:
        return {
            "type": "object",
            "properties": {},
            "required": []
        }
    
    properties = {}
    required = []
    
    for param_name, param_spec in xlam_params.items():
        if not isinstance(param_spec, dict):
            continue
            
        # Parse type and determine if required
        xlam_type = param_spec.get("type", "str")
        openai_type, is_required = parse_xlam_type(xlam_type)
        
        # Build property definition
        prop = {
            "type": openai_type,
            "description": param_spec.get("description", "")
        }
        
        # Add default value if present (convert to string for consistency)
        if "default" in param_spec:
            default_val = param_spec["default"]
            # Convert all defaults to strings to avoid Arrow schema conflicts
            if default_val is not None and default_val != "":
                prop["default"] = str(default_val)
            # Skip empty string defaults as they're not useful
        
        # Add enum if present (though rare in XLAM)
        if "enum" in param_spec:
            prop["enum"] = param_spec["enum"]
            
        properties[param_name] = prop
        
        if is_required:
            required.append(param_name)
    
    return {
        "type": "object", 
        "properties": properties,
        "required": required
    }


def convert_xlam_tools(tools_json_str: str) -> List[Dict[str, Any]]:
    """Convert XLAM tools JSON string to OpenAI function format."""
    try:
        xlam_tools = json.loads(tools_json_str)
    except json.JSONDecodeError:
        return []
    
    if not isinstance(xlam_tools, list):
        return []
    
    openai_functions = []
    
    for tool in xlam_tools:
        if not isinstance(tool, dict):
            continue
            
        func_name = tool.get("name", "").strip()
        if not func_name:  # Skip if function name is empty
            continue
            
        # Convert to OpenAI function format
        function = {
            "name": func_name,
            "description": tool.get("description", ""),
            "parameters": convert_xlam_parameters(tool.get("parameters", {}))
        }
        
        openai_functions.append(function)
    
    return openai_functions


def convert_xlam_answers(answers_json_str: str) -> List[Dict[str, Any]]:
    """Convert XLAM answers JSON string to function-call parts."""
    try:
        xlam_answers = json.loads(answers_json_str)
    except json.JSONDecodeError:
        return []
    
    if not isinstance(xlam_answers, list):
        return []
    
    function_call_parts = []
    
    for answer in xlam_answers:
        if not isinstance(answer, dict):
            continue
            
        func_name = answer.get("name", "").strip()
        if not func_name:  # Skip if function name is empty
            continue
            
        # Convert all argument values to strings for type consistency
        args = answer.get("arguments", {})
        normalized_args = {}
        for key, value in args.items():
            # Only include non-null values to avoid massive parameter lists
            if value is not None and value != "":
                normalized_args[key] = str(value)
            
        part = {
            "type": "function-call",
            "name": func_name,
            "args": normalized_args
        }
        
        function_call_parts.append(part)
    
    return function_call_parts


def convert_xlam_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a single XLAM sample to new chat format."""
    dataset_source = "xlam-function-calling-60k"
    query = sample.get("query", "")
    
    # Generate conversation ID
    conversation_id = generate_conversation_id(dataset_source, query)
    
    # Convert tools to OpenAI format
    available_functions = convert_xlam_tools(sample.get("tools", "[]"))
    
    # Convert answers to function-call parts
    function_call_parts = convert_xlam_answers(sample.get("answers", "[]"))
    
    # Build chat format
    chat_sample = {
        "conversation_id": conversation_id,
        "dataset_source": dataset_source,
        "original_metadata": {
            "original_id": sample.get("id", 0)
        },
        "system_prompt": {
            "content": "You are a helpful assistant with access to tools. Use them to answer the user's questions effectively.",
            "metadata": {}
        },
        "initial_prompt": {
            "role": "user",
            "content": query,
            "metadata": {}
        },
        "available_functions": available_functions,
        "conversation_branches": [
            {
                "messages": [
                    {
                        "role": "assistant",
                        "parts": function_call_parts
                    }
                ] if function_call_parts else []
            }
        ],
        "created_timestamp": datetime.now().isoformat()
    }
    
    return chat_sample


def create_chat_format_schema():
    """Create explicit Arrow schema for chat format to avoid inference overhead."""
    import pyarrow as pa
    
    # Define schema for chat format structure
    schema = pa.schema([
        pa.field("conversation_id", pa.string()),
        pa.field("dataset_source", pa.string()),
        pa.field("original_metadata", pa.struct([
            pa.field("original_id", pa.int64())
        ])),
        pa.field("system_prompt", pa.struct([
            pa.field("content", pa.string()),
            pa.field("metadata", pa.struct([]))
        ])),
        pa.field("initial_prompt", pa.struct([
            pa.field("role", pa.string()),
            pa.field("content", pa.string()),
            pa.field("metadata", pa.struct([]))
        ])),
        pa.field("available_functions", pa.list_(pa.struct([
            pa.field("name", pa.string()),
            pa.field("description", pa.string()),
            pa.field("parameters", pa.struct([
                pa.field("type", pa.string()),
                pa.field("properties", pa.string()),  # JSON string for flexibility
                pa.field("required", pa.list_(pa.string()))
            ]))
        ]))),
        pa.field("conversation_branches", pa.list_(pa.struct([
            pa.field("messages", pa.list_(pa.struct([
                pa.field("role", pa.string()),
                pa.field("parts", pa.list_(pa.struct([
                    pa.field("type", pa.string()),
                    pa.field("name", pa.string(), nullable=True),
                    pa.field("args", pa.string(), nullable=True)  # JSON string for flexibility
                ])))
            ])))
        ]))),
        pa.field("created_timestamp", pa.string())
    ])
    
    return schema


def process_dataset(dataset: Dataset, chunk_size: int = 1000) -> Dataset:
    """Process XLAM dataset to convert to chat format."""
    print(f"Converting {len(dataset):,} samples in chunks of {chunk_size:,}...")
    
    converted_rows = []
    failed_count = 0
    total_rows = len(dataset)
    
    # Use tqdm for progress bar
    from tqdm import tqdm
    with tqdm(total=total_rows, desc="Converting XLAM samples", unit="samples") as pbar:
        for chunk_start in range(0, total_rows, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_rows)
            chunk_rows = []
            
            # Process chunk
            for idx in range(chunk_start, chunk_end):
                try:
                    sample = dataset[idx]
                    converted_sample = convert_xlam_sample(sample)
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
    
    # Create new dataset from converted rows with explicit schema
    if not converted_rows:
        return Dataset.from_list([])
    
    print("Creating Dataset with explicit schema...")
    
    # Flatten complex nested structures for Arrow compatibility
    flattened_rows = []
    for row in converted_rows:
        flattened_row = {
            "conversation_id": row["conversation_id"],
            "dataset_source": row["dataset_source"],
            "original_metadata": row["original_metadata"],
            "system_prompt": row["system_prompt"],
            "initial_prompt": row["initial_prompt"],
            "available_functions": [
                {
                    "name": func["name"],
                    "description": func["description"],
                    "parameters": {
                        "type": func["parameters"]["type"],
                        "properties": json.dumps(func["parameters"]["properties"]),
                        "required": func["parameters"]["required"]
                    }
                } for func in row["available_functions"]
            ],
            "conversation_branches": [
                {
                    "messages": [
                        {
                            "role": msg["role"],
                            "parts": [
                                {
                                    "type": part["type"],
                                    "name": part.get("name"),
                                    "args": json.dumps(part.get("args", {})) if part.get("args") else None
                                } for part in msg["parts"]
                            ]
                        } for msg in branch["messages"]
                    ]
                } for branch in row["conversation_branches"]
            ],
            "created_timestamp": row["created_timestamp"]
        }
        flattened_rows.append(flattened_row)
    
    try:
        return Dataset.from_list(flattened_rows)
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
                "operation": "convert_xlam_function_calling",
                "script": "convert_xlam_function_calling.py",
                "timestamp": datetime.now().isoformat(),
                "input_path": str(input_path),
                "output_path": str(output_path),
                "chunk_size": args.batch_size,
                "description": "Converted XLAM function calling dataset to standardized chat format with OpenAI-compatible function definitions"
            }
        ],
        "format": "chat_format_v1",
        "source_dataset": "xlam-function-calling-60k",
        "conversion_details": {
            "function_format": "openai_compatible", 
            "parameter_schema": "json_schema",
            "conversation_type": "single_turn_function_calling"
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
        description="Convert XLAM function calling dataset to standardized chat format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert XLAM dataset to chat format
  venv/bin/python 02-standardisation/convert_xlam_function_calling.py \\
    /users/schlag/store/posttrain_data/01_raw_hf_data/xlam-function-calling-60k \\
    --output data/02-standardised/xlam-function-calling-60k
  
  # With custom chunk size
  venv/bin/python 02-standardisation/convert_xlam_function_calling.py \\
    /path/to/xlam/dataset \\
    --output /path/to/output \\
    --batch-size 5000
        """
    )
    
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to XLAM dataset directory (HuggingFace format)"
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
    
    processed_dataset = DatasetDict(processed_splits)
    
    # Save results
    save_dataset_and_metadata(processed_dataset, output_path, input_path, args)
    
    # Print summary
    total_samples = sum(len(split) for split in processed_dataset.values())
    print(f"\nConversion completed successfully!")
    print(f"Total samples processed: {total_samples:,}")
    print(f"Output format: Standardized chat format with OpenAI-compatible functions")


if __name__ == "__main__":
    main()