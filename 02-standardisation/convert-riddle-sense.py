"""
convert_riddle_sense.py
───────────────────────
Convert the INK-USC/riddle_sense dataset from HuggingFace into the unified chat-tool schema.

The RiddleSense dataset contains riddle-solving problems structured as:
• answerKey - The correct answer's label (e.g., "A", "B", "C", "D", "E")
• choices - A dictionary containing:
    - label: List of choice labels (e.g., ["A", "B", "C", "D", "E"])
    - text: List of possible answers corresponding to the labels
• question - The riddle text that needs to be solved

Example structure:
{
    "answerKey": "E",
    "choices": {
        "label": ["A", "B", "C", "D", "E"],
        "text": ["throw", "bit", "gallow", "mouse", "hole"]
    },
    "question": "A riddle text here..."
}

This converter:
1. Converts the question into an initial prompt
2. Generates a structured response using templates that include:
   - Various answer formats ("The answer is...", "Solution:", etc.)
   - Different label variations (capitalized and lowercase)
3. Includes the correct answer as a verifiable response
4. Adds creation timestamp in ISO format with UTC timezone

The converted format follows a chat-tool schema with:
- Conversation metadata (ID, source, timestamp)
- System prompt (empty for this dataset)
- Initial prompt (the riddle question)
- Available functions (empty for this dataset)
- Conversation branches containing the assistant's response and verifiable answer
"""


import enum
import re
import sys
import json
import argparse
import random
from datetime import datetime, UTC
from typing import Dict, Any, Optional
from pathlib import Path
from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets

SRC = "riddle_sense"


def check_for_tooling(ds: Dataset):
    # Prints all questions that mention tools or using tools
    data = ds['train']
    for i, sample in enumerate(data):
        question_lower = sample['question'].lower()
        if ('tool' in question_lower or 'tools' in question_lower or 
            'use' in question_lower or 'using' in question_lower or 
            'used' in question_lower or 'uses' in question_lower):
            print(sample['question'])
            print("\n")
        if (i%100 ==0 ):
            print(f"{i} done")

def generate_response(answer: str) -> str:
    content = [
        # Only lowercase label
        "The {label} is: {answer}",
        'The {label} is: "{answer}"',
        "The {label} is: '{answer}'",
        "I conclude the {label} is: {answer}",
        "Strictly speaking, the {label} is: {answer}",
    ]
    mixed_content = [
        # Lowercase and first letter capitalized label
        "{label}: {answer}",
        "{label} — {answer}",
        "{label} → {answer}",
        "{label} ⇒ {answer}",
        "{label} = {answer}",
        '"{answer}" — {label}'
    ]
    label = ["answer", "the right choice", "solution"]
    mixed_label = ["Answer", "The right choice", "Solution", "answer", "the right choice", "solution"]

    go_choice = random.choice([0,1])
    if go_choice == 0:
        gen_response = random.choice(content)
        gen_response = gen_response.format(label=random.choice(label), answer=answer)
    else:
        gen_response = random.choice(mixed_content)
        gen_response = gen_response.format(label=random.choice(mixed_label), answer=answer)
    return gen_response

def convert_sample(sample: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """Convert a single sample to the new format."""
    
    # Start with existing fields
    converted: Dict[str, Any] = {
        "conversation_id": f"{SRC}_{idx}",
        "dataset_source": SRC,
        "original_metadata": {},
        "created_timestamp": datetime.now(UTC).isoformat(),
    }
    
    # System prompt is always the same, we don't want it
    converted["system_prompt"] = {
        "content": "",
        "metadata": {},
    }
    
    # Extracts parts.
    question = sample['question']
    answer = sample['answerKey']
    labels = sample['choices']['label']
    texts = sample['choices']['text']

    # Merge question with choices with some randomness
    if random.random() < 0.5:
        labels = [label.lower() for label in labels]
        answer = answer.lower()
    if random.random() < 0.5:
        texts = [text.capitalize() for text in texts]
    punct = random.choice(['.', ')'])
    spacing = f"{random.choice([' ', '\n'])}"
    formatted_choices = [
        f"{spacing}"  # Random spacing
        f"{label}{punct} {text}"  # Label format, punctuation, and Answer text
        for label, text in zip(labels, texts)
    ]
    question += " ".join(formatted_choices)
    question = question.strip()
    
    # Process initial_prompt
    converted["initial_prompt"] = {
        "role": "user",
        "content": question,
        "metadata": {}
        }
    
    # No available functions in this dataset
    converted["available_functions"] = []
    
    # Gets the answer and potentially a verifiable response.
    gen_response = generate_response(answer)

    parts: list[Dict] = [
        {
            "type": "response",
            "content": gen_response,
            "metadata": 
                {}
        }
    ]

    if answer is not None:
        parts.append({
            "type": "verifiable-responses",
            "answers": [str(answer)],
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
            "conversation_type": "riddle_sense",
            "added_fields": ["system_prompt", "conversation_branches"],
            "edited elements": ["Assistant messages' content"],
            "format": "new_chat_format_with_parts"
        }
    
    # Save metadata
    metadata_file = output_path / "dataset_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Dataset saved to {output_path}")
    print(f"Metadata saved to {metadata_file}")

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
        dataset = concatenate_datasets([load_dataset("INK-USC/riddle_sense", revision="refs/convert/parquet", split="train"), load_dataset("INK-USC/riddle_sense", revision="refs/convert/parquet", split="validation")])
        dataset = DatasetDict({"train": dataset})
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
