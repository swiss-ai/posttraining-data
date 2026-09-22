#!/usr/bin/env python3
"""
convert_dongfujiang_fetaqa.py
───────────────────────
Convert the DongfuJiang/FeTaQA dataset from JSON format into the unified chat-tool schema.

Required additional libraries:
- jsonlines
- tabulate

To install:
> pip install jsonlines tabulate

The FeTaQA dataset contains table QA problems structured as:
• feta_id - the numerical sample ID
• table_source_json - relative path to the JSON source table
• page_wikipedia_url - the URL to the Wikipedia page the table is from
• table_page_title - the Wikipedia page title
• table_section_title - the Wikipedia section title
• table_array - a 2D array representation of a table
• highlighted_cell_ids - a list of pairs (x, y) of highlighted cells
• question - a question that can be answered with the table
• answer - the answer to the question

This converter:
1. Adds unmatched fields to the metadata dict
2. Converts the table to one of the random text table format defined in TABLE_FORMATS
3. Appends the text table to the question as initial user prompt
4. Converts the answer field as assistant response
"""

import re
import sys
import json
import argparse
import jsonlines
import random
from tabulate import tabulate
from datetime import datetime, UTC
from typing import Dict, Any, Optional
from pathlib import Path
from datasets import Dataset, DatasetDict
from huggingface_hub import hf_hub_download

SRC = "dongfujiang_fetaqa"

TABLE_FORMATS = [
    # tabulate formats
    # https://github.com/astanin/python-tabulate?tab=readme-ov-file#table-format
    "plain",
    "simple",
    "github",
    "grid",
    "simple_grid",
    "rounded_grid",
    # "heavy_grid",
    "mixed_grid",
    "double_grid",
    # "fancy_grid",
    "outline",
    "simple_outline",
    # "rounded_outline",
    # "heavy_outline",
    "mixed_outline",
    "double_outline",
    # "fancy_outline",
    "pipe",
    "orgtbl",
    "asciidoc",
    "jira",
    "presto",
    "pretty",
    # "psql", # similar to pretty
    "rst",
    "mediawiki",
    "moinmoin",
    # "youtrack", # similar to jira
    "html",
    "unsafehtml",
    "latex",
    "latex_raw",
    "latex_booktabs",
    "latex_longtable",
    "textile",
    "tsv",
    
    # custom formats
    "raw",
    "json",
    "row_array"
]

# ───────────── helpers ────────────── #

def convert_table(table: list[list[str]], use_header: bool, fmt: str) -> str:
    if fmt == "raw":
        return str(table)
    if fmt == "json":
        return json.dumps(table)
    if fmt == "row_array":
        return "\n".join(json.dumps(r) for r in table)
    if use_header:
        return tabulate(table[1:], headers=table[0], tablefmt=fmt)
    return tabulate(table, tablefmt=fmt)

def convert_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a single sample to the new format."""
    
    # Start with existing fields
    converted: Dict[str, Any] = {
        "conversation_id": sample["feta_id"],
        "dataset_source": SRC,
        "original_metadata": {
            "feta_id": sample["feta_id"],
            "table_source_json": sample["table_source_json"],
            "page_wikipedia_url": sample["page_wikipedia_url"],
            "table_page_title": sample["table_page_title"],
            "table_section_title": sample["table_section_title"],
            "highlighted_cell_ids": sample["highlighted_cell_ids"],
            "table_array": sample["table_array"],
        },
        "created_timestamp": datetime.now(UTC).isoformat(),
    }
    
    # System prompt is always the same, we don't want it
    converted["system_prompt"] = {
        "content": "",
        "metadata": {},
    }
    
    # Random conversion
    add_header_row = not random.getrandbits(1)
    title = random.choice([f'# {sample["table_page_title"]}\n## {sample["table_section_title"]}\n', f'{sample["table_page_title"]}, {sample["table_section_title"]}\n', ''])
    table_format = random.choice(TABLE_FORMATS)
    separator = "\n" * random.randint(1, 2)
    
    # Final prompt
    prompt = f'{sample["question"]}{separator}{title}{convert_table(sample["table_array"], add_header_row, table_format)}'
    
    # Process initial_prompt
    converted["initial_prompt"] = {
        "role": "user",
        "content": prompt,
        "metadata": {
            "table_format": table_format,
            "separator": separator,
            "has_header_row": add_header_row,
            "title": title,
        },
    }
    
    # Add missing available_functions field
    converted["available_functions"] = []
    
    # Gets the answer and potentially a verifiable response.
    parts: list[Dict] = [
        {
            "type": "response",
            "content": sample["answer"],
            "metadata": {},
        }
    ]
    
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
            "added_fields": ["system_prompt", "available_functions", "conversation_branches", "initial_prompt"],
            "cleaned_elements": [],
            "field_normalization": "converted table array to string",
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
        dataset_path_train = hf_hub_download(repo_id="DongfuJiang/FeTaQA", filename="fetaQA-v1_train.jsonl", repo_type="dataset", cache_dir="/tmp")
        dataset_path_test = hf_hub_download(repo_id="DongfuJiang/FeTaQA", filename="fetaQA-v1_test.jsonl", repo_type="dataset", cache_dir="/tmp")
        dataset_path_dev = hf_hub_download(repo_id="DongfuJiang/FeTaQA", filename="fetaQA-v1_dev.jsonl", repo_type="dataset", cache_dir="/tmp")
    else:
        dataset_path_train = args.input
        dataset_path_test = args.input
        dataset_path_dev = args.input
    
    # Load JSON data
    print(f"Loading data from {dataset_path_train}")
    try:
        loaded_data: list[Dict] = []
        for dataset_path in set([dataset_path_train, dataset_path_test, dataset_path_dev]):
            with jsonlines.open(dataset_path) as f:
                for l in f:
                    loaded_data.append(l)
    except (FileNotFoundError, json.JSONDecodeError, jsonlines.InvalidLineError) as e:
        print(f"Error loading input file: {e}")
        sys.exit(1)
    dataset = DatasetDict({'train': Dataset.from_list(loaded_data)})
    
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
        converted_samples.append(convert_sample(sample))
    
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