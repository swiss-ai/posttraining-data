#!/usr/bin/env python3
"""
convert_numinamath_variation.py
──────────────────────────────
Convert the HuggingFace *NuminaMath-1.5* dataset into the unified chat-tool schema 
we use for XLAM.

This is a variation of convert_numinamath.py that uses different answer formats
instead of just "The answer is: {answer}".

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

import re, json, sys, argparse, hashlib, random
from pathlib import Path
from datetime import datetime, UTC
from typing import List, Dict, Any, Optional

from datasets import Dataset, DatasetDict, load_from_disk

SRC = "numina-math-1.5"

# ───────────── answer format variations ───────────── #
ANSWER_FORMATS = [
    "The {label} is: {answer}",
    "{label}: {answer}",
    "{label} — {answer}",
    "{label} → {answer}",
    "{label} ⇒ {answer}",
    "{label} = {answer}",
    "I conclude the {label} is: {answer}",
    "By inspection, the {label} is: {answer}",
    "After analysis, the {label} is: {answer}",
    "This evaluates to: {answer}",
    "This resolves to: {answer}",
    "This computes to: {answer}",
    "Therefore: {answer}",
    "Thus: {answer}",
    "Hence: {answer}",
    "In one line: {answer}",
    "Plainly put, the {label} is: {answer}",
    "In summary, the {label} is: {answer}",
    "Strictly speaking, the {label} is: {answer}",
    "Calculated {label}: {answer}",
    "Computed {label}: {answer}",
    "Evaluated {label}: {answer}",
    "Derived {label}: {answer}",
    "Inferred {label}: {answer}",
    "Final {label}: {answer}",
    "The computed {label} is: {answer}",
    "The derived {label} is: {answer}",
    "The final {label} is: {answer}",
    "The resulting {label} is: {answer}",
    '"{answer}" — {label}',
    'The {label} is: "{answer}"',
    "The {label} is: '{answer}'"
]

LABELS = [
    "Answer",
    "Solution", 
    "Result",
    "Evaluation",
    "Conclusion",
    "Expression"
]

# ───────────── helpers ────────────── #
def conv_id(seed: str) -> str:
    return f"{SRC}_{hashlib.sha256(seed.encode()).hexdigest()[:12]}"

def make_part(ptype: str,
              content: str = "",
              name: str = "",
              args: Optional[Dict[str, Any]] = None,
              metadata: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    return {
        "type": ptype,
        "content": content,
        "metadata": metadata or {},
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

def contains_links_or_images(text: str) -> bool:
    """Check if text contains links or images."""
    if not text:
        return False
    # Check for image links, http links, .com domains, etc.
    patterns = [
        r'!\[.*?\]\(.*?\)',  # Markdown images
        r'https?://',        # HTTP/HTTPS links
        r'\.com',           # .com domains
        r'\.org',           # .org domains
        r'\.net',           # .net domains
        r'\.edu',           # .edu domains
        r'cdn\.',           # CDN links
        r'mathpix\.com',    # Mathpix specifically
    ]
    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False

def clean_problem_text(text: str) -> str:
    """Clean problem text by removing headers and short prefixes."""
    if not text:
        return text
    
    # Remove markdown headers
    text = re.sub(r'^#+\s*', '', text)
    
    # Remove common problem prefixes
    prefixes_to_remove = [
        r'^Problem\s+\d+\.?\s*',
        r'^NT\d+\s*',
        r'^G\d+\s*',
        r'^##\s*Problem\s+\d+\.?\s*',
        r'^#\s*Problem\s+\d+\.?\s*',
    ]
    
    for prefix in prefixes_to_remove:
        text = re.sub(prefix, '', text, flags=re.IGNORECASE)
    
    # Remove first sentence if it's very short (less than 20 chars)
    lines = text.split('\n')
    if lines and len(lines[0].strip()) < 20:
        # Check if it ends with a period
        first_line = lines[0].strip()
        if first_line.endswith('.') and len(first_line) < 20:
            lines = lines[1:]
    
    return '\n'.join(lines).strip()

def clean_solution_text(text: str) -> str:
    """Clean solution text by removing headers and short prefixes."""
    if not text:
        return text
    
    # Remove markdown headers
    text = re.sub(r'^#+\s*', '', text)
    
    # Remove common solution prefixes
    prefixes_to_remove = [
        r'^Solution\s*\d*\.?\s*',
        r'^##\s*Solution\s*\d*\.?\s*',
        r'^#\s*Solution\s*\d*\.?\s*',
    ]
    
    for prefix in prefixes_to_remove:
        text = re.sub(prefix, '', text, flags=re.IGNORECASE)
    
    # Remove first sentence if it's very short (less than 20 chars)
    lines = text.split('\n')
    if lines and len(lines[0].strip()) < 20:
        # Check if it ends with a period
        first_line = lines[0].strip()
        if first_line.endswith('.') and len(first_line) < 20:
            lines = lines[1:]
    
    return '\n'.join(lines).strip()

def get_random_answer_format(answer: str) -> str:
    """Get a random answer format with appropriate label."""
    format_template = random.choice(ANSWER_FORMATS)
    label = random.choice(LABELS)
    
    # Check if the label appears at the beginning of the sentence or in the middle
    # Look for patterns like "The {label}", "I conclude the {label}", etc.
    sentence_start_patterns = [
        "The {label}",
        "I conclude the {label}",
        "By inspection, the {label}",
        "After analysis, the {label}",
        "Plainly put, the {label}",
        "In summary, the {label}",
        "Strictly speaking, the {label}",
        "The computed {label}",
        "The derived {label}",
        "The final {label}",
        "The resulting {label}",
        "The {label} is:",
        "{label} ⇒",
        "{label} →",
        "{label} —",
        "{label} =",
        "{label}:"
    ]
    
    # Check if this format template matches any sentence-start pattern
    # For patterns that start with {label}, we need to check differently
    is_sentence_start = False
    
    # Check for patterns that start with {label}
    label_start_patterns = ["{label} ⇒", "{label} →", "{label} —", "{label} =", "{label}:"]
    if any(pattern in format_template for pattern in label_start_patterns):
        is_sentence_start = True
    else:
        # Check for other sentence-start patterns
        is_sentence_start = any(pattern.format(label="PLACEHOLDER") in format_template for pattern in sentence_start_patterns)
    
    # If it's not a sentence start pattern, make label lowercase
    if not is_sentence_start:
        label = label.lower()
    
    return format_template.format(label=label, answer=answer)

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
    
    # Clean problem and solution text
    problem = clean_problem_text(problem)
    solution = clean_solution_text(solution)
    
    # Create system prompt
    system_text = create_system_prompt(problem_type, question_type)
    
    # Determine answer type and create appropriate response
    answer_lower = answer.lower() if answer else ""
    
    if answer_lower == "proof":
        # For proof answers, use the solution as assistant text
        assistant_text = solution
        reasoning = None
        verifiable_answer = None
    else:
        # For regular answers, use a random answer format
        assistant_text = get_random_answer_format(answer)
        reasoning = solution
        verifiable_answer = answer
    
    # Create messages - assistant message with reasoning and response
    messages = []
    
    if assistant_text:
        parts = []
        
        # Add reasoning as THOUGHT if it exists
        if reasoning:
            parts.append(make_part("thought", reasoning))
        
        # Add the answer as response
        parts.append(make_part("response", assistant_text))
        
        messages.append({
            "role": "assistant",
            "parts": parts
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
                "synthetic": row.get("synthetic", False),
                "verifiable_answer": verifiable_answer
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
            "solution_is_valid": row.get("solution_is_valid", ""),
            "verifiable_answer": p["initial"]["metadata"]["verifiable_answer"]
        },
        "system_prompt": {"content": p["system"], "metadata": {}},
        "initial_prompt": p["initial"],
        "available_functions": p["functions"],
        "conversation_branches": [{"messages": p["messages"]}],
        "created_timestamp": datetime.now(UTC).isoformat()
    }

def process_split(ds: Dataset, num_proc: int) -> Dataset:
    # First filter out samples that would be None, then convert
    def should_keep(row):
        problem = clean_text(row.get("problem", ""))
        solution = clean_text(row.get("solution", ""))
        answer = clean_text(row.get("answer", ""))
        
        # Filter out samples with links or images
        if contains_links_or_images(problem) or contains_links_or_images(solution):
            return False
        
        # Filter out notfound answers
        if answer.lower() == "notfound":
            return False
        
        # Clean problem and solution text
        problem = clean_problem_text(problem)
        solution = clean_solution_text(solution)
        
        # Skip if problem or solution is empty after cleaning
        if not problem or not solution:
            return False
        
        return True
    
    # Filter first
    filtered_ds = ds.filter(should_keep, desc="Filtering samples")
    
    # Then convert
    converted = filtered_ds.map(convert_row,
                               with_indices=True,
                               num_proc=num_proc,
                               desc="Converting NuminaMath")
    
    return converted

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
        "operation": "convert_numinamath_variation",
        "script": "convert_numinamath_variation.py",
        "timestamp": datetime.now(UTC).isoformat(),
        "input_path": str(input_path),
        "output_path": str(output_path),
        "num_processes": args.num_proc,
        "limit": args.limit,
        "description": "Converted NuminaMath dataset with filtering, cleaning, and varied answer formats"
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
            "metadata_preservation": "full_original_metadata",
            "filtering": "removed_links_images_notfound",
            "cleaning": "removed_headers_short_prefixes",
            "field_mapping": "reasoning_verifiable_answer",
            "answer_format": "varied_formats_with_random_labels"
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
