#!/usr/bin/env python3
"""
convert_numina_math.py
──────────────────────
Convert the HuggingFace *NuminaMath-1.5* dataset into the unified chat-tool schema 
we use for XLAM.

The NuminaMath dataset contains mathematical problems with solutions, structured as:
• `problem` - The mathematical problem statement
• `solution` - The detailed solution/answer
• `answer` - Short answer type
• `problem_type` - Category of the problem
• `question_type` - Type of question
• Additional metadata fields

Turn mapping
------------
Problem statement    → role="user"          part.type="response"
Solution/Answer     → role="assistant"     part.type="response"
"""

import re, json, sys, argparse, hashlib
from pathlib import Path
from datetime import datetime, UTC
from typing import List, Dict, Any, Optional

from datasets import Dataset, DatasetDict, load_from_disk

SRC = "numina-math-1.5"

# ───────────── helpers ────────────── #
def conv_id(seed: str) -> str:
    return f"{SRC}_{hashlib.sha256(seed.encode()).hexdigest()[:12]}"

def make_part(ptype: str,
              content: str = "",
              name: str = "",
              args: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    return {
        "type": ptype,
        "content": content,
        "name": name,
        "args": json.dumps(args, ensure_ascii=False) if args else ""
    }

def clean_text(text: str) -> str:
    """Clean and normalize text content."""
    if not text:
        return ""
    # Remove excessive whitespace and normalize
    text = re.sub(r'\n\s*\n', '\n\n', text.strip())
    text = re.sub(r' +', ' ', text)
    return text

def create_system_prompt(problem_type: str, question_type: str) -> str:
    """Create a system prompt based on the problem type and question type."""
    base_prompt = "You are a helpful mathematics tutor. Solve the given mathematical problem step by step, showing your work clearly."
    
    if problem_type:
        base_prompt += f" This is a {problem_type.lower()} problem."
    
    if question_type:
        if "word-problem" in question_type.lower():
            base_prompt += " Read the problem carefully and identify the key information needed to solve it."
        elif "proof" in question_type.lower():
            base_prompt += " Provide a rigorous mathematical proof with clear logical steps."
    
    return base_prompt

# ───────────— sample parser ───────── #
def parse_sample(row: Dict[str, Any]) -> Dict[str, Any]:
    """Parse a NuminaMath sample into our chat format."""
    
    # Extract fields
    problem = clean_text(row.get("problem", ""))
    solution = clean_text(row.get("solution", ""))
    answer = clean_text(row.get("answer", ""))
    problem_type = row.get("problem_type", "")
    question_type = row.get("question_type", "")
    
    # Create system prompt
    system_text = create_system_prompt(problem_type, question_type)
    
    # Create messages - assistant message with only the solution
    messages = []
    
    if solution:
        # Create assistant message with only the solution (no answer appended)
        messages.append({
            "role": "assistant",
            "parts": [make_part("response", solution)]
        })
    
    return {
        "system": "",
        "functions": [],  # No functions for math problems
        "initial": {
            "role": "user",
            "content": problem,
            "metadata": {
                "problem_type": problem_type,
                "question_type": question_type,
                "answer": answer,
                "source": row.get("source", ""),
                "synthetic": row.get("synthetic", False)
            }
        },
        "messages": messages,
    }

# ───────────— map row ─────────────── #
def convert_row(row: Dict[str, Any], idx: int) -> Dict[str, Any]:
    p = parse_sample(row)
    return {
        "conversation_id": conv_id(p["initial"]["content"] + str(idx)),
        "dataset_source": SRC,
        "original_metadata": {
            "row_index": idx,
            "problem_type": row.get("problem_type", ""),
            "question_type": row.get("question_type", ""),
            "answer": row.get("answer", ""),
            "source": row.get("source", ""),
            "synthetic": row.get("synthetic", False),
            "problem_is_valid": row.get("problem_is_valid", ""),
            "solution_is_valid": row.get("solution_is_valid", "")
        },
        "system_prompt": {"content": p["system"], "metadata": {}},
        "initial_prompt": p["initial"],
        "available_functions": p["functions"],
        "conversation_branches": [{"messages": p["messages"]}],
        "created_timestamp": datetime.now(UTC).isoformat()
    }

def process_split(ds: Dataset, num_proc: int) -> Dataset:
    return ds.map(convert_row,
                  with_indices=True,
                  num_proc=num_proc,
                  desc="Converting NuminaMath")

def subset(ds: Dataset, lim: Optional[int]):
    return ds if not lim or lim <= 0 or lim >= ds.num_rows else ds.select(range(lim))

def load_existing_metadata(input_path: Path) -> Optional[Dict[str, Any]]:
    """Load existing dataset metadata if it exists."""
    meta_file = Path(input_path) / "dataset_metadata.json"
    if meta_file.exists():
        try:
            with open(meta_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return None

def save_dataset_and_metadata(dataset_dict: DatasetDict, output_path: Path, 
                             input_path: Path, args: argparse.Namespace):
    """Save converted dataset with processing metadata."""
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save dataset
    dataset_dict.save_to_disk(str(output_path))
    
    # Load existing metadata or create new
    metadata = load_existing_metadata(input_path) or {}
    
    # Create processing entry
    processing_entry = {
        "operation": "convert_numina_math",
        "script": "convert_numina_math.py",
        "timestamp": datetime.now(UTC).isoformat(),
        "input_path": str(input_path),
        "output_path": str(output_path),
        "num_processes": args.num_proc,
        "limit": args.limit,
        "description": "Converted NuminaMath dataset to standardized chat format for mathematical problem solving"
    }
    
    # Add to processing log
    if "processing_log" not in metadata:
        metadata["processing_log"] = []
    metadata["processing_log"].append(processing_entry)
    
    # Add format metadata if not already present
    if "format" not in metadata:
        metadata["format"] = "chat_format_v1"
    if "source_dataset" not in metadata:
        metadata["source_dataset"] = "AI-MO/NuminaMath-1.5"
    if "conversion_details" not in metadata:
        metadata["conversion_details"] = {
            "conversation_type": "mathematical_problem_solving",
            "problem_format": "text_based_math_problems",
            "solution_format": "step_by_step_explanations",
            "metadata_preservation": "full_original_metadata"
        }
    
    # Save metadata
    metadata_file = output_path / "dataset_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Dataset saved to {output_path}")
    print(f"Metadata saved to {metadata_file}")

# ───────────— CLI / main ───────────── #
def cli():
    p = argparse.ArgumentParser()
    p.add_argument("input_path")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--num-proc", type=int, default=8)
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()

def main():
    a = cli()
    inp = Path(a.input_path)
    out = Path(a.output if not a.output.endswith("/")
               else a.output + inp.name + "-converted")

    if out.exists() and input(f"{out} exists. overwrite? [y/N]: ").lower() != "y":
        sys.exit(0)

    ds = load_from_disk(str(inp))
    if not isinstance(ds, DatasetDict):
        ds = DatasetDict({"train": ds})

    out_ds = DatasetDict()
    for split, d in ds.items():
        print(f"{split}: {d.num_rows:,} rows")
        d = subset(d, a.limit)
        out_ds[split] = process_split(d, a.num_proc)

    save_dataset_and_metadata(out_ds, out, inp, a)

if __name__ == "__main__":
    main()
