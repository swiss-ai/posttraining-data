#!/usr/bin/env python3
"""
convert_lipengcs_table_gpt.py
───────────────────────
Convert the LipengCS/Table-GPT dataset from JSON format into the unified chat-tool schema.

The Table-GPT dataset contains table QA problems structured as:
• task - The type of the task, can be one of ColumnAugmentation, DataImputation, EntityMatching, 
         ErrorDetection, HeaderValueMatching, ListExtraction, NL2SQL, Row2RowTransformation, 
         RowAugmentation, RowColumnFiltering, RowColumnSorting, RowColumnSwapping, SchemaMatching,
         TableSummarization.
• dataset - The name of the dataset from which the example originates.
• prompt - The input prompt provided to the model for generating a response.
• completion - The generated output response corresponding to the given prompt.
• messages - A list of messages that combine the prompt and completion, typically used in chat-oriented models.
• prompt_setting - 'few-shot' or 'zero-shot'
• metadata - A dict for other information about the example.

This converter:
1. Converts the original metadata
2. Drops the system prompt
3. Cleans the user question by removing recurrent formatting biases
4. Removes the reasoning traces of the answers
5. Extracts the answer as a verifiable answer when given as a JSON object
"""

import re
import sys
import json
import argparse
from datetime import datetime, UTC
from typing import Dict, Any, Optional
from pathlib import Path
from datasets import Dataset, DatasetDict, load_dataset

SRC = "lipengcs_table_gpt"

# ───────────── helpers ────────────── #

regex_question_bias = re.compile(r'(?:\#\s*)?[\w\s\[\]]+?:\s*')
regex_last_json = re.compile(r"\{[^{}]*\}(?=[^}]*$)")

def clean_question(question: str) -> str:
    lines = ": ".join(question.split(": ")[1:]).splitlines()
    question = "\n".join(lines[:-3])
    return question

def clean_answer(answer: str) -> str:
    match = re.search(regex_last_json, answer)
    if match:
        return match.group(0)
    return answer

def extract_verifiable_answer(answer: str) -> Optional[str]:
    """Tries to find a JSON object in the answer. Picks the last one."""
    verifiable_answer = None

    match = re.search(regex_last_json, answer)
    if match:
        verifiable_answer = match.group(0)
        
    try:
        verifiable_answer = json.loads(verifiable_answer)
    except:
        pass
    else:
        keys = verifiable_answer.keys()
        assert len(keys) == 1, "LipengCS/Table-GPT has only JSON objects with a single key"
        verifiable_answer = verifiable_answer[next(iter(keys))]

    return verifiable_answer

def convert_sample(sample: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """Convert a single sample to the new format."""
    
    # Extracts original metadata.
    original_metadata = json.loads(sample["metadata"])
    
    # Start with existing fields
    converted: Dict[str, Any] = {
        "conversation_id": f"{SRC}_{idx}",
        "dataset_source": SRC,
        "original_metadata": {
            "dataset": original_metadata["dataset"],
            "num_fewshots": int(original_metadata["num_fewshots"]),
            "seed": int(original_metadata["seed"]),
            "table": original_metadata["table"],
            "task": original_metadata["task"],
        },
        "created_timestamp": datetime.now(UTC).isoformat(),
    }
    
    # System prompt is always the same, we don't want it
    converted["system_prompt"] = {
        "content": "",
        "metadata": {},
    }
    
    # Extracts parts
    question = next(m for m in sample["messages"] if m["role"] == "user")
    answer = next(m for m in sample["messages"] if m["role"] == "assistant")
    
    # Process initial_prompt
    cleaned_question = clean_question(question["content"])
    converted["initial_prompt"] = {
        "role": "user",
        "content": cleaned_question,
        "metadata": 
            {} if question["content"] == cleaned_question 
            else {"original_content": question["content"]
        },
    }
    
    # Add missing available_functions field
    converted["available_functions"] = []
    
    # Attempts to extract verifiable answer.
    verifiable_answer = extract_verifiable_answer(answer["content"])
    
    # Gets the answer and potentially a verifiable response.
    cleaned_response = clean_answer(answer["content"])
    parts: list[Dict] = [
        {
            "type": "response",
            "content": cleaned_response,
            "metadata": 
                {} if answer["content"] == cleaned_response
                else {"original_content": answer["content"]}
        }
    ]
    
    if verifiable_answer is not None:
        parts.append({
            "type": "verifiable-responses",
            "answers": [str(verifiable_answer)],
        })
    
    # Process conversation branches    
    converted["conversation_branches"] = [
        {
            "messages": [
                {
                    "role": "assistant",
                    "parts": parts,
                },
            ],
        },
    ]
    
    return converted

def load_existing_metadata(output_path: Path) -> Optional[Dict[str, Any]]:
    """Load existing dataset metadata if it exists."""
    meta_file = output_path / "dataset_metadata.json"
    if meta_file.exists():
        try:
            with open(meta_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return None

def save_dataset_and_metadata(dataset_dict: DatasetDict, output_path: Path, args: argparse.Namespace):
    """Save converted dataset with processing metadata."""
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save dataset
    dataset_dict.save_to_disk(str(output_path))
    
    # Load existing metadata or create new
    metadata = load_existing_metadata(output_path) or {}
    
    # Create processing entry
    processing_entry = {
        "operation": f"convert_{SRC}",
        "script": f"convert_{SRC}.py",
        "timestamp": datetime.now(UTC).isoformat(),
        "input_path": args.input,
        "output_path": str(output_path),
        "num_processes": args.num_proc,
        "limit": args.limit,
        "description": f"Converted {SRC} dataset from JSON to unified chat format"
    }
    
    # Add to processing log
    if "processing_log" not in metadata:
        metadata["processing_log"] = []
    metadata["processing_log"].append(processing_entry)
    
    # Add format metadata if not already present
    if "format" not in metadata:
        metadata["format"] = "chat_format_v1"
    if "source_dataset" not in metadata:
        metadata["source_dataset"] = SRC
    if "conversion_details" not in metadata:
        metadata["conversion_details"] = {
            "conversation_type": "table_task_qa",
            "added_fields": ["system_prompt", "available_functions", "conversation_branches"],
            "cleaned_elements": ["Assistant messages' content"],
            "field_normalization": "Removed question biases",
            "format": "new_chat_format_with_parts"
        }
    
    # Save metadata
    metadata_file = output_path / "dataset_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Dataset saved to {output_path}")
    print(f"Metadata saved to {metadata_file}")

# ──────────── CLI / main ───────────── #
def cli():
    p = argparse.ArgumentParser(description=f"Convert {SRC} dataset to unified chat format")
    p.add_argument("-i", "--input", default=None, help="Input file path. If None, will be loaded from the Hub.")
    p.add_argument("-o", "--output", required=True, help="Output directory path")
    p.add_argument("--num-proc", type=int, default=8, help="Number of processes for dataset operations")
    p.add_argument("--limit", type=int, default=None, help="Limit number of samples to process")
    return p.parse_args()

def main():
    args = cli()
    output_path = Path(args.output)
    
    # Check if output exists
    if output_path.exists():
        response = input(f"{output_path} exists. Overwrite? [y/N]: ")
        if response.lower() != "y":
            sys.exit(0)
    
    if args.input is None:
        dataset = load_dataset("LipengCS/Table-GPT", "All", cache_dir="/tmp")
    else:
        # Load JSON data
        print(f"Loading data from {args.input}")
        try:
            with open(args.input, 'r') as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading input file: {e}")
            sys.exit(1)
        dataset = DatasetDict({'train': Dataset.from_list(data)})
    
    if not isinstance(dataset, DatasetDict):
        print(f"Unexpected dataset type.")
        sys.exit(1)
        
    data = dataset["train"]
    print(f"Loaded {len(data)} samples")
    
    # Apply limit if specified
    if args.limit and args.limit > 0:
        data = data[:args.limit]
        print(f"Limited to {len(data)} samples")
    
    # Convert samples
    print("Converting samples to new format...")
    converted_samples = []
    for i, sample in enumerate(data):
        if i % 1000 == 0:
            print(f"Processing sample {i}/{len(data)}")
        converted_samples.append(convert_sample(sample, i))
    
    # Create Dataset and DatasetDict
    print("Creating DatasetDict...")    
    dataset = Dataset.from_list(converted_samples)
    dataset_dict = DatasetDict({"train": dataset})
    
    print(f"Converted {len(converted_samples)} samples")
    
    # Save dataset and metadata
    save_dataset_and_metadata(dataset_dict, output_path, args)
    
    print("Conversion complete!")

if __name__ == "__main__":
    main()
