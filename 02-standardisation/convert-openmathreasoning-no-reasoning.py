import re
import json, sys, argparse, hashlib
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from datasets import Dataset, DatasetDict, load_dataset

SRC = "nvidia/OpenMathReasoning"
UTC = timezone.utc


def parse_pass_rate(raw: str) -> Optional[float]:
    """Convert pass_rate_72b_tir string to float, or None for 'n/a'."""
    if not raw or raw == "n/a":
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def should_keep(sample: Dict[str, Any], min_pass_rate: Optional[float]) -> bool:
    """
    Filter criteria:
    - Drop pass_rate == 0.0 (solution is always wrong per TIR evaluation).
    - Drop pass_rate < min_pass_rate if a threshold is given.
    - Rows with pass_rate = None ('n/a') are kept — they are unverifiable proof or
      open-ended problems, not confirmed wrong.
    """
    rate = parse_pass_rate(sample.get("pass_rate_72b_tir", "n/a"))
    if rate is not None and rate == 0.0:
        return False
    if min_pass_rate is not None and rate is not None and rate < min_pass_rate:
        return False
    return True


def extract_solution_parts(generated_solution: str, keep_reasoning: bool) -> list[Dict[str, Any]]:
    """Build assistant parts, optionally preserving <think> traces as thoughts."""
    think_blocks = [
        match.group(0).strip()
        for match in re.finditer(r"<think>.*?</think>", generated_solution, flags=re.DOTALL)
    ]
    solution = re.sub(r"<think>.*?</think>", "", generated_solution, flags=re.DOTALL).strip()

    parts: list[Dict[str, Any]] = []
    if keep_reasoning and think_blocks:
        parts.append({
            "type": "thought",
            "content": "\n\n".join(think_blocks),
            "metadata": {},
        })

    parts.append({
        "type": "response",
        "content": solution,
        "metadata": {},
    })
    return parts


def convert_sample(sample: Dict[str, Any], keep_reasoning: bool = False) -> Dict[str, Any]:
    """Convert a single sample to the unified chat format."""
    parts = extract_solution_parts(sample["generated_solution"], keep_reasoning)
    if sample.get("expected_answer"):
        parts.append({
            "type": "verifiable-responses",
            "answers": [sample["expected_answer"]],
        })

    # Stable conversation_id from problem text — lets downstream tools
    # group/deduplicate the ~6 solutions per unique problem.
    problem_hash = hashlib.sha256(sample["problem"].encode()).hexdigest()[:16]

    return {
        "conversation_id": problem_hash,
        "dataset_source": SRC,
        "original_metadata": {
            "problem_source":    sample.get("problem_source", ""),
            "problem_type":      sample.get("problem_type", ""),
            "generation_model":  sample.get("generation_model", ""),
            "pass_rate_72b_tir": parse_pass_rate(sample.get("pass_rate_72b_tir", "n/a")),
            "inference_mode":    sample.get("inference_mode", ""),
            "used_in_kaggle":    sample.get("used_in_kaggle", False),
        },
        "created_timestamp": datetime.now(UTC).isoformat(),
        "system_prompt": {"content": "", "metadata": {}},
        "initial_prompt": {
            "role": "user",
            "content": sample["problem"],
            "metadata": {},
        },
        "available_functions": [],
        "conversation_branches": [
            {
                "messages": [
                    {
                        "role": "assistant",
                        "parts": parts,
                    }
                ]
            }
        ],
    }


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
    output_path.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(output_path))

    metadata = load_existing_metadata(output_path) or {}

    processing_entry = {
        "operation": f"convert_{SRC}",
        "script": f"convert_{SRC}.py",
        "timestamp": datetime.now(UTC).isoformat(),
        "input_path": args.input,
        "output_path": str(output_path),
        "num_processes": args.num_proc,
        "limit": args.limit,
        "min_pass_rate": args.min_pass_rate,
        "keep_zero_pass": args.keep_zero_pass,
        "keep_reasoning": args.keep_reasoning,
        "problem_types": args.problem_types,
        "description": f"Converted {SRC} dataset to unified chat format",
    }

    if "processing_log" not in metadata:
        metadata["processing_log"] = []
    metadata["processing_log"].append(processing_entry)

    if "format" not in metadata:
        metadata["format"] = "chat_format_v1"
    if "source_dataset" not in metadata:
        metadata["source_dataset"] = SRC
    if "conversion_details" not in metadata:
        metadata["conversion_details"] = {
            "conversation_type": "math_reasoning",
            "added_fields": ["system_prompt", "conversation_branches"],
            "edited_fields": [
                "generated_solution (think tags preserved as thought part)"
                if args.keep_reasoning
                else "generated_solution (think tags stripped)"
            ],
            "format": "new_chat_format_with_parts",
        }

    metadata_file = output_path / "dataset_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Dataset saved to {output_path}")
    print(f"Metadata saved to {metadata_file}")

def cli():
    p = argparse.ArgumentParser(description=f"Convert {SRC} dataset to unified chat format")
    p.add_argument("-i", "--input", default=None,
                   help="Input JSON file path. If omitted, loads from HuggingFace Hub.")
    p.add_argument("-o", "--output", required=True, help="Output directory path")
    p.add_argument("--num-proc", type=int, default=8, help="Number of processes for dataset operations")
    p.add_argument("--limit", type=int, default=None, help="Limit number of samples to process")
    p.add_argument("--min-pass-rate", type=float, default=None,
                   help=(
                       "Drop solutions with pass_rate_72b_tir below this threshold "
                       "(e.g. 0.25). Solutions with pass_rate='n/a' (unverifiable proof/"
                       "open-ended problems) are always kept."
                   ))
    p.add_argument("--keep-zero-pass", action="store_true", default=False,
                   help="Keep solutions with pass_rate=0.0 (always wrong). Filtered by default.")
    p.add_argument("--keep-reasoning", action="store_true", default=False,
                   help=(
                       "Preserve <think>...</think> reasoning traces as a thought part. "
                       "By default, reasoning traces are stripped from generated_solution."
                   ))
    p.add_argument("--problem-types", choices=["has_answer", "has_answer_and_proof", "all"],
                   default="all",
                   help=(
                       "Filter by problem_type. "
                       "'has_answer' keeps only 'has_answer_extracted' (~60%% of data). "
                       "'has_answer_and_proof' keeps 'has_answer_extracted' + 'converted_proof' (~76%%). "
                       "'all' keeps all types including 'no_answer_extracted' (default)."
                   ))
    return p.parse_args()

def main():
    args = cli()
    output_path = Path(args.output)

    if output_path.exists():
        response = input(f"{output_path} exists. Overwrite? [y/N]: ")
        if response.lower() != "y":
            sys.exit(0)

    if args.input is None:
        # "cot" is a split name, not a config — load_dataset returns a Dataset directly
        data = load_dataset("nvidia/OpenMathReasoning", split="cot")
    else:
        print(f"Loading data from {args.input}")
        try:
            with open(args.input, 'r') as f:
                raw = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading input file: {e}")
            sys.exit(1)
        data = Dataset.from_list(raw)

    print(f"Loaded {len(data)} samples")

    if args.limit and args.limit > 0:
        data = data.select(range(min(args.limit, len(data))))
        print(f"Limited to {len(data)} samples")

    allowed_problem_types = {
        "has_answer":           {"has_answer_extracted"},
        "has_answer_and_proof": {"has_answer_extracted", "converted_proof"},
        "all":                  None,  # no filter
    }[args.problem_types]

    if allowed_problem_types is not None:
        print(f"Filtering to problem_types: {sorted(allowed_problem_types)}")

    print("Converting and filtering samples...")
    converted_samples = []
    filtered_zero_pass = 0
    filtered_low_pass = 0
    filtered_problem_type = 0
    for i, sample in enumerate(data):
        if i % 1000 == 0:
            print(f"Processing sample {i}/{len(data)}")

        if allowed_problem_types is not None and sample.get("problem_type") not in allowed_problem_types:
            filtered_problem_type += 1
            continue

        rate = parse_pass_rate(sample.get("pass_rate_72b_tir", "n/a"))
        if not args.keep_zero_pass and rate is not None and rate == 0.0:
            filtered_zero_pass += 1
            continue
        if args.min_pass_rate is not None and rate is not None and rate < args.min_pass_rate:
            filtered_low_pass += 1
            continue

        converted_samples.append(convert_sample(sample, keep_reasoning=args.keep_reasoning))

    if allowed_problem_types is not None:
        print(f"Filtered out: {filtered_problem_type} samples with excluded problem_type")
    if not args.keep_zero_pass:
        print(f"Filtered out: {filtered_zero_pass} zero-pass-rate samples (always wrong)")
    else:
        print("Zero-pass-rate filter skipped (--keep-zero-pass)")
    if args.min_pass_rate is not None:
        print(f"Filtered out: {filtered_low_pass} samples below min_pass_rate={args.min_pass_rate}")
    print(f"Kept: {len(converted_samples)} samples")

    print("Creating DatasetDict...")
    dataset_dict = DatasetDict({"train": Dataset.from_list(converted_samples)})

    save_dataset_and_metadata(dataset_dict, output_path, args)
    print("Conversion complete!")

if __name__ == "__main__":
    main()
