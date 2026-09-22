"""
convert_LEXam_mcq.py
───────────────────────
Convert the LEXam-Benchmark/LEXam dataset (mcq_4_choices subset) from HuggingFace into the unified chat-tool schema.

The LEXam MCQ dataset contains multiple-choice questions structured as:
    - question: The multiple-choice question
    - choices: List of answer choices
    - gold: Position of the correct answer within the choices list
    - course: Title of the law course from which the question was derived
    - language: Language of the question (en or de)
    - area: Legal area covered by the question (criminal, public, private, or interdisciplinary)
    - jurisdiction: Legal jurisdiction of the question (Swiss, international, or generic)
    - year: Year when the exam was administered (2016 to 2022)
    - n_statements: Number of statements contained in the question (2 to 5)
    - none_as_an_option: Binary indicator specifying whether None of the statements (or Keine der Aussagen) is included among the answer choices
    - id: Unique identifier for the question
    - negative_question: Binary indicator specifying whether the question is phrased negatively (e.g. Which of the following statements are incorrect?)


This converter:
1. Converts the question and choices into a clear prompt format
2. Maps Roman numeral answers to the corresponding choice text
3. Generates structured responses with consistent formatting
4. Includes metadata about the question
5. Preserves the verifiable answer in standard format
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
from datasets import Dataset, DatasetDict, load_dataset

SRC = "LEXam_mcq"

def extract_meta(sample: Dict[str, Any]) -> Dict[str, Any]:
    '''Extract extra information (e.g. language, course etc.) to provide as metadata'''
    extra = {}
    for key in sample:
        if key not in ['question', 'choices', 'gold']:
            extra[key] = sample[key]
    return extra

def comb_prod(choices: list, pos: int) -> str:
    '''Produces a string of options to be added to the user.question field
        it randomly augments for option labels and newline/inline separators'''
    LABEL_LETTERS = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'j']
    LABEL_NUMBERS = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
    LABEL_LET_CAP = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J']

    labels = []
    r = random.random()
    if r < 0.3:
        labels = LABEL_LETTERS
    elif r >= 0.3 and r <= 0.7:
        labels = LABEL_LET_CAP
    else:
        labels = LABEL_NUMBERS

    # getting the punctuation right
    punct = '.' if random.random()> 0.5 else ')'
    
    verifiable = labels[pos]

    combinations = ''
    for i in range(len(choices)):
        choices[i] = f'{labels[i]}{punct} {choices[i]}'
        
    combinations = f'{random.choice([' ', "\n"])}'.join(choices)

    return combinations, verifiable, punct

def gen_answer(answers: list, lang: str) -> str:
    '''Produces a string of {answer_label}{text} with an emphasis
        on multiple answer punctuations'''
    final = 'und ' if lang == 'de' else 'and '
    if len(answers) < 2:
        return answers[0]
    final = final + answers.pop()
    while answers:
        answers[-1] = answers[-1][:-1] + '; ' if answers[-1].endswith(".") else answers[-1]
        final = answers.pop() + final
    return final

def convert_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a single sample to the new format."""
    
    # Start with existing fields
    converted: Dict[str, Any] = {
        "conversation_id": "",
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
    choices = re.findall(r"'(.*?)'",sample['choices'])
    right_choice = choices[sample['gold']]

    combinations, verifiable, punct = comb_prod(choices, sample['gold'])
    question = question + f'\n{combinations}'
    extra_info = extract_meta(sample)

    if (right_choice == 'none of the statements') or (right_choice == 'keine der Aussagen'):
        answer = f'{verifiable}{punct} ' + right_choice
    else:
        prefixes = re.findall(r'\b[ivx]+\b', right_choice)
        pattern = r'^(?:' + '|'.join(prefixes) + r')\..*' 
        answers = re.findall(pattern, question, re.MULTILINE)
        if len(answers)==0: answers = prefixes
        answer = f'{verifiable}{punct} ' + gen_answer(answers, sample['language'])
    
    # Process initial_prompt
    converted["initial_prompt"] = {
        "role": "user",
        "content": question,
        "metadata": extra_info
        }
    
    # No available functions in this dataset
    converted["available_functions"] = []

    parts: list[Dict] = [
        {
            "type": "response",
            "content": answer,
            "metadata": 
                {}
        }
    ]

    if answer is not None:
        parts.append({
            "type": "verifiable-responses",
            "answers": [verifiable],
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
            "conversation_type": "LEXam_mcq",
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
        dataset = load_dataset("LEXam-Benchmark/LEXam",
                                "mcq_4_choices")
    else:
        # Load JSON data
        print(f"Loading data from {args.input}")
        try:
            with open(args.input, 'r') as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading input file: {e}")
            sys.exit(1)
        dataset = DatasetDict({'test': Dataset.from_list(data)})
    
    if not isinstance(dataset, DatasetDict):
        print(f"Unexpected dataset type.")
        sys.exit(1)
        
    data = dataset["test"]
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