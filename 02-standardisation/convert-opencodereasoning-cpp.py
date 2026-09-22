import json, sys, argparse, random
from pathlib import Path
from subprocess import run
from datetime import datetime, UTC
from typing import Dict, Any, Optional

from datasets import Dataset, DatasetDict, load_dataset

SRC = "nvidia/OpenCodeReasoning-2"

hf_datasets = {
    "taco": load_dataset("BAAI/TACO", trust_remote_code=True),
    "apps": load_dataset("codeparrot/apps", trust_remote_code=True),
    "code_contests": load_dataset("deepmind/code_contests"),
    "open-r1/codeforces": load_dataset("open-r1/codeforces")
}

def get_question(ds_name, split, index):
    benchmark = hf_datasets[ds_name][split][int(index)]
    if ds_name == "code_contests":
        if not benchmark["description"]:
            return None
        return benchmark["description"]
    elif ds_name in ["taco", "apps"]:
        return benchmark["question"]
    elif ds_name == "open-r1/codeforces":
        if not benchmark["description"]:
            return None
        question = benchmark["description"]
        
        # Randomly select Input/Output label variations
        input_labels = ["Input", "Input:", "input", "input:"]
        output_labels = ["Output", "Output:", "output", "output:"]
        input_label = random.choice(input_labels)
        output_label = random.choice(output_labels)
        
        if benchmark["input_format"]:
            question += f"\n\n{input_label}\n\n" + benchmark["input_format"]
        if benchmark["output_format"]:
            question += f"\n\n{output_label}\n\n" + benchmark["output_format"]
        if benchmark["examples"]:
            question += "\n\nExamples"
            for example in benchmark["examples"]:
                # Randomly select labels for each example
                example_input_label = random.choice(input_labels)
                example_output_label = random.choice(output_labels)
                if "input" in example:
                    question += f"\n\n{example_input_label}\n\n" + example["input"]
                if "output" in example:
                    question += f"\n\n{example_output_label}\n\n" + example["output"]
        if benchmark["note"]:
            question += "\n\nNote\n\n" + benchmark["note"]
        return question

    return None

def lint_code(code: str) -> Optional[str]:
    result = run(
        ["clang-format"],
        input=code.strip().encode(),
        capture_output=True,
    )
    if result.returncode != 0:
        return None
    return f"```cpp\n{result.stdout.decode()}\n```"

def parse_sample(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    linted_code = lint_code(row["solution"])
    if linted_code is None:
        return None
    
    messages = [
        {
            "role": "assistant",
            "parts": [
                {
                    "type": "thought",
                    "content": row["r1_generation"].split("<think>")[1].split("</think>")[0].strip(),
                    "metadata": {}
                },
                {
                    "type": "response",
                    "content": linted_code,
                    "metadata": {}
                }
            ]
        }
    ]

    raw_question = get_question(row["dataset"], row["split"], int(row["index"]))
    question = raw_question.split("Explanation\n")[0].strip()

    return {
        "system": "",
        "initial": {
            "role": "user",
            "content": question,
            "metadata": {
                "question_id": row["question_id"],
                "has_explanation": raw_question != question,
            }
        },
        "messages": messages,
    }

def convert_row(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    p = parse_sample(row)
    if p is None:
        return None
    return {
        "conversation_id": row["id"],
        "dataset_source":  SRC,
        "system_prompt": {"content": p["system"], "metadata": {}},
        "initial_prompt": p["initial"],
        "available_functions": p["functions"] if "functions" in p else [],
        "conversation_branches": [{"messages": p["messages"]}],
        "created_timestamp": datetime.now(UTC).isoformat()
    }

def process_split(ds: Dataset, num_proc: int) -> Dataset:
    result = ds.filter(lambda x: float(x["pass_rate"]) > 0.98, num_proc=num_proc)
    # Remove the 'sample' column after processing to avoid schema conflicts
    result = result.map(convert_row,
                    num_proc=num_proc,
                    desc="Converting opencodereasoning-cpp")

    result = result.filter(lambda x: x is not None, num_proc=num_proc)
    # Remove the original 'sample' column to clean up the schema
    if 'sample' in result.column_names:
        result = result.remove_columns(['sample'])
    return result

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
        "operation": "convert_opencodereasoning-cpp",
        "script": "convert_opencodereasoning-cpp.py",
        "timestamp": datetime.now(UTC).isoformat(),
        "input_path": str(input_path),
        "output_path": str(output_path),
        "num_processes": args.num_proc,
        "limit": args.limit,
    }
    
    # Add to processing log
    if "processing_log" not in metadata:
        metadata["processing_log"] = []
    metadata["processing_log"].append(processing_entry)
    
    # Add format metadata if not already present
    if "format" not in metadata:
        metadata["format"] = "chat_format_v1"
    if "source_dataset" not in metadata:
        metadata["source_dataset"] = "opencodereasoning-cpp"
    
    # Save metadata
    metadata_file = output_path / "dataset_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Dataset saved to {output_path}")
    print(f"Metadata saved to {metadata_file}")

# ───────────— CLI / main ───────────── #
def cli():
    p = argparse.ArgumentParser()
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--num-proc", type=int, default=64)
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()

def main():
    a = cli()
    inp = Path(SRC)
    out = Path(a.output if not a.output.endswith("/")
               else a.output + inp.name + "-converted")

    if out.exists() and input(f"{out} exists. overwrite? [y/N]: ").lower() != "y":
        sys.exit(0)

    ds = load_dataset(str(inp), split="cpp")
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
