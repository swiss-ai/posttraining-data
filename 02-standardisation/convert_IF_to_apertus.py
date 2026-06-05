#!/usr/bin/env python3
"""
Convert datasets to standard post-training format.

Converts multiple source datasets into the standardized HuggingFace dataset format
with conversation_branches structure matching other datasets in this folder.

Output schema:
- conversation_id: unique identifier
- dataset_source: name of the source dataset
- original_metadata: JSON string of original metadata
- created_timestamp: ISO timestamp
- system_prompt: {"content": str, "metadata": str}
- initial_prompt: {"role": str, "content": str, "metadata": str}
- available_functions: [{"name": str, "description": str, "parameters": str}]
- conversation_branches: [{"messages": [{"role": str, "parts": [{"type": str, "content": str, "metadata": str, "name": str, "args": str, "answers": [str]}]}]}]
"""

import re
import sys
import json
import random
import argparse
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from datasets import Dataset, DatasetDict, load_from_disk, load_dataset
from tqdm import tqdm

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_ANALYSIS_SEP_RE = re.compile(r"^analysis.*?assistantfinal", re.DOTALL | re.IGNORECASE)


def strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> reasoning traces and strip surrounding whitespace."""
    return _THINK_RE.sub("", text).strip()


def clean_gpt_reasoning(text: str) -> Optional[str]:
    """
    Clean gpt-oss-120b reasoning artifacts from assistant response content.

    - If content starts with 'analysis' and contains 'assistantfinal', strip
      everything up to and including 'assistantfinal' and return the remainder.
    - If content starts with 'analysis' but has no 'assistantfinal', the entry
      is pure unfinished reasoning — return None to signal the record should be dropped.
    - Otherwise return the text unchanged.
    """
    if not text.lower().startswith("analysis"):
        return text
    cleaned = _ANALYSIS_SEP_RE.sub("", text).strip()
    if cleaned == text.strip():
        # No separator found — pure reasoning, signal drop
        return None
    return cleaned


# =============================================================================
# Dataset Configuration
# =============================================================================
# Define all data sources with their paths, types, and sample counts
# Set num_samples to None or -1 to use all samples

DATASET_CONFIG = {
    "multiturnIF_LSAIE": {
        "path": "/capstor/store/cscs/swissai/infra01/reasoning/data/sft_1.1/multiturn-if/30_01_merged_dialogues.jsonl",
        "type": "jsonl",
        "num_samples": None,  # Use all (filtered by selected=True)
        "messages_field": "messages",
        "filter_field": "selected",
        "filter_value": True,
    },
    "if-eng_Latn-12k-v1-mix-multiturn-verified": {
        "path": "swiss-ai/if-sft-verified-multiturn",
        "type": "hf_hub",
        "messages_field": "messages",
        "num_samples": None,
        "filter_field": None,
        "filter_value": False,
    },
    "dolci-if-eng_Latn-200k": {
        "path": "/capstor/store/cscs/swissai/infra01/reasoning/data/sft_1.1/if-dolci-aug/dolci-if-eng_Latn-200k-19-01",
        "type": "hf_passthrough",  # Already in standard format, just filter and sample
        "num_samples": 70000,
        "filter_field": "verification_passed",
        "filter_value": True,
    },
    "dolci-if-eng_Latn-70k-balanced": {
        "path": "/capstor/store/cscs/swissai/infra01/reasoning/data/sft_1.1/if-dolci-aug/dolci-if-eng_Latn-200k-04-04",
        "type": "hf_passthrough",  # Already in standard format, just filter and sample
        "num_samples": None,
        "filter_field": "verification_passed",
        "filter_value": True,
    },
}


# =============================================================================
# Schema Helper Functions
# =============================================================================

def create_part(part_type: str = "response", content: str = "", metadata: str = "",
                name: str = "", args: str = "", answers: Optional[List[str]] = None) -> Dict[str, Any]:
    """Create a schema-compliant message part."""
    part = {
        "type": part_type,
        "content": content,
        "metadata": metadata,
        "name": name,
        "args": args,
    }
    if answers is not None:
        part["answers"] = answers
    return part


def create_message(role: str, parts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Create a schema-compliant message."""
    return {
        "role": role,
        "parts": parts
    }


def create_system_prompt(content: str = "", metadata: str = "") -> Dict[str, Any]:
    """Create a schema-compliant system prompt."""
    return {
        "content": content,
        "metadata": metadata
    }


def create_initial_prompt(content: str = "", role: str = "user", metadata: str = "") -> Dict[str, Any]:
    """Create a schema-compliant initial prompt."""
    return {
        "role": role,
        "content": content,
        "metadata": metadata
    }


def normalize_function(func: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a function definition to the standard schema."""
    if not isinstance(func, dict):
        return {"name": "", "description": "", "parameters": ""}

    # Get parameters and serialize to string if needed
    params = func.get("parameters", {})
    if isinstance(params, dict):
        params = json.dumps(params, ensure_ascii=False)
    elif params is None:
        params = ""

    return {
        "name": func.get("name", "") or "",
        "description": func.get("description", "") or "",
        "parameters": params
    }


def generate_conversation_id(dataset_source: str, content: str, sample_id: Optional[str] = None) -> str:
    """Generate a unique conversation ID."""
    dataset_prefix = dataset_source.replace('/', '_').replace('-', '_')

    if sample_id:
        content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()[:8]
        return f"{dataset_prefix}_{sample_id}_{content_hash}"
    else:
        content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]
        return f"{dataset_prefix}_{content_hash}"


def serialize_metadata(value: Any) -> str:
    """Serialize any metadata value to a JSON string."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


# =============================================================================
# Conversion Functions
# =============================================================================

def convert_simple_messages_to_standard(
    messages: List[Dict[str, Any]],
    dataset_source: str,
    original_id: str = "",
    original_metadata: Any = None
) -> Dict[str, Any]:
    """
    Convert a list of simple messages (role + content) to standard format.

    Args:
        messages: List of {"role": str, "content": str} dicts
        dataset_source: Name of the source dataset
        original_id: Original ID from source
        original_metadata: Original metadata to preserve

    Returns:
        Standard format record
    """
    if not messages:
        return None

    system_prompt_content = ""
    initial_prompt_content = ""
    conversation_messages = []

    for i, msg in enumerate(messages):
        role = msg.get("role", "user")
        content = msg.get("content", "") or ""

        if role == "system" and not system_prompt_content:
            system_prompt_content = content
        elif role == "user" and not initial_prompt_content:
            initial_prompt_content = content
        else:
            # Add to conversation messages
            part = create_part(part_type="response", content=content)
            conversation_messages.append(create_message(role, [part]))

    if not initial_prompt_content:
        return None

    # Generate conversation ID
    conversation_id = generate_conversation_id(dataset_source, initial_prompt_content, original_id)

    return {
        "conversation_id": conversation_id,
        "dataset_source": dataset_source,
        "original_metadata": serialize_metadata(original_metadata),
        "created_timestamp": datetime.now(timezone.utc).isoformat(),
        "system_prompt": create_system_prompt(system_prompt_content),
        "initial_prompt": create_initial_prompt(initial_prompt_content),
        "available_functions": [],
        "conversation_branches": [{"messages": conversation_messages}]
    }


def convert_parts_messages_to_standard(
    system_prompt: Optional[Dict[str, Any]],
    initial_prompt: Optional[Dict[str, Any]],
    conversation_branches: List[Dict[str, Any]],
    available_functions: List[Dict[str, Any]],
    dataset_source: str,
    original_id: str = "",
    original_metadata: Any = None
) -> Dict[str, Any]:
    """
    Convert a dataset already in parts format to the standard schema.

    Args:
        system_prompt: {"content": str, ...}
        initial_prompt: {"content": str, "role": str, ...}
        conversation_branches: [{"messages": [...]}]
        available_functions: List of function definitions
        dataset_source: Name of the source dataset
        original_id: Original ID from source
        original_metadata: Original metadata to preserve

    Returns:
        Standard format record
    """
    # Extract system prompt content
    system_content = ""
    if system_prompt and isinstance(system_prompt, dict):
        system_content = system_prompt.get("content", "") or ""

    # Extract initial prompt content
    initial_content = ""
    initial_role = "user"
    if initial_prompt and isinstance(initial_prompt, dict):
        initial_content = initial_prompt.get("content", "") or ""
        initial_role = initial_prompt.get("role", "user") or "user"

    if not initial_content:
        return None

    # Normalize conversation branches
    normalized_branches = []
    for branch in (conversation_branches or []):
        if not isinstance(branch, dict):
            continue

        branch_messages = branch.get("messages", [])
        normalized_messages = []

        for msg in branch_messages:
            if not isinstance(msg, dict):
                continue

            role = msg.get("role", "user")
            parts = msg.get("parts", [])

            normalized_parts = []
            for part in parts:
                if not isinstance(part, dict):
                    continue

                normalized_parts.append(create_part(
                    part_type=part.get("type", "response") or "response",
                    content=part.get("content", "") or "",
                    metadata=serialize_metadata(part.get("metadata")),
                    name=part.get("name", "") or "",
                    args=serialize_metadata(part.get("args")),
                    answers=part.get("answers") if "answers" in part else None
                ))

            if normalized_parts:
                normalized_messages.append(create_message(role, normalized_parts))

        if normalized_messages:
            normalized_branches.append({"messages": normalized_messages})

    # If no branches but we have content, create empty branch
    if not normalized_branches:
        normalized_branches = [{"messages": []}]

    # Normalize available functions
    normalized_functions = [normalize_function(f) for f in (available_functions or [])]

    # Generate conversation ID
    conversation_id = generate_conversation_id(dataset_source, initial_content, original_id)

    return {
        "conversation_id": conversation_id,
        "dataset_source": dataset_source,
        "original_metadata": serialize_metadata(original_metadata),
        "created_timestamp": datetime.now(timezone.utc).isoformat(),
        "system_prompt": create_system_prompt(system_content),
        "initial_prompt": create_initial_prompt(initial_content, initial_role),
        "available_functions": normalized_functions,
        "conversation_branches": normalized_branches
    }


# =============================================================================
# Data Loading Functions
# =============================================================================

def get_nested_value(data: dict, field_path: str) -> Any:
    """Get a value from a nested dict using dot notation (e.g., 'metadata.status')."""
    keys = field_path.split('.')
    value = data
    for key in keys:
        if isinstance(value, dict):
            value = value.get(key)
        else:
            return None
    return value


def load_jsonl_records(
    input_path: str,
    dataset_name: str,
    filter_field: Optional[str] = None,
    filter_value: Any = None,
    num_samples: Optional[int] = None,
    messages_field: str = "messages"
) -> List[Dict[str, Any]]:
    """Load and convert JSONL dataset to standard format records."""
    records = []

    print(f"Loading JSONL from: {input_path}")
    with open(input_path, 'r') as f:
        lines = f.readlines()

    print(f"Processing {len(lines)} entries...")
    for idx, line in enumerate(tqdm(lines, desc=f"Converting {dataset_name}")):
        data = json.loads(line)

        # Apply filter if specified (supports nested fields like 'metadata.status')
        if filter_field:
            actual_value = get_nested_value(data, filter_field)
            if actual_value != filter_value:
                continue

        messages = data.get(messages_field, [])
        if not messages:
            continue

        # Ensure we have a system message first if not present
        if not messages or messages[0].get("role") != "system":
            messages = [{"role": "system", "content": ""}] + messages

        # Get original metadata (exclude messages field)
        original_metadata = {k: v for k, v in data.items() if k != messages_field}

        # Convert to standard format
        record = convert_simple_messages_to_standard(
            messages=messages,
            dataset_source=dataset_name,
            original_id=str(data.get("id", idx)),
            original_metadata=original_metadata
        )

        if record:
            records.append(record)

    # Sample if num_samples specified
    if num_samples and num_samples > 0 and len(records) > num_samples:
        random.shuffle(records)
        records = records[:num_samples]

    print(f"Loaded {len(records)} samples from {dataset_name}")
    return records


def load_hf_records(
    input_path: str,
    dataset_name: str,
    filter_field: Optional[str] = None,
    filter_value: Any = None,
    num_samples: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Load and convert HuggingFace dataset to standard format records."""
    print(f"Loading HF dataset from: {input_path}")
    ds = load_from_disk(input_path)

    # Handle DatasetDict vs Dataset
    if hasattr(ds, 'keys'):
        ds = ds['train']

    records = []

    print(f"Processing {len(ds)} entries...")
    for i in tqdm(range(len(ds)), desc=f"Converting {dataset_name}"):
        sample = ds[i]

        # Apply filter if specified
        if filter_field and sample.get(filter_field) != filter_value:
            continue

        # Get original metadata
        original_metadata = {
            "verification_passed": sample.get("verification_passed"),
        }

        # Convert using parts-based converter
        record = convert_parts_messages_to_standard(
            system_prompt=sample.get("system_prompt"),
            initial_prompt=sample.get("initial_prompt"),
            conversation_branches=sample.get("conversation_branches", []),
            available_functions=sample.get("available_functions", []),
            dataset_source=dataset_name,
            original_id=sample.get("conversation_id", str(i)),
            original_metadata=original_metadata
        )

        if record:
            records.append(record)

    # Sample if num_samples specified
    if num_samples and num_samples > 0 and len(records) > num_samples:
        random.shuffle(records)
        records = records[:num_samples]

    print(f"Loaded {len(records)} samples from {dataset_name}")
    return records


def load_hf_passthrough_records(
    input_path: str,
    dataset_name: str,
    filter_field: Optional[str] = None,
    filter_value: Any = None,
    num_samples: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Load HF dataset already in standard format - just filter and sample, no conversion."""
    print(f"Loading HF dataset (passthrough): {input_path}")
    ds = load_from_disk(input_path)

    # Handle DatasetDict vs Dataset
    if hasattr(ds, 'keys'):
        ds = ds['train']

    records = []

    print(f"Processing {len(ds)} entries (passthrough mode)...")
    for i in tqdm(range(len(ds)), desc=f"Filtering {dataset_name}"):
        sample = ds[i]

        # Apply filter if specified
        if filter_field and sample.get(filter_field) != filter_value:
            continue

        # Pass through as-is (already in standard format)
        # Just ensure the record is a plain dict
        record = dict(sample)

        # Update dataset_source to match the config name
        record["dataset_source"] = dataset_name

        # Clean gpt-oss-120b reasoning artifacts
        if record.get("generation_model") == "openai/gpt-oss-120b":
            drop_record = False
            new_branches = []
            for branch in record.get("conversation_branches", []):
                new_messages = []
                for msg in branch.get("messages", []):
                    if msg.get("role") == "assistant":
                        new_parts = []
                        for part in msg.get("parts", []):
                            content = part.get("content", "")
                            # Strip <think>...</think> blocks
                            if "<think>" in content:
                                content = strip_think_blocks(content)
                            # Strip analysis...assistantfinal prefix; drop if pure reasoning
                            cleaned = clean_gpt_reasoning(content)
                            if cleaned is None:
                                drop_record = True
                                break
                            part = {**part, "content": cleaned}
                            new_parts.append(part)
                        if drop_record:
                            break
                        msg = {**msg, "parts": new_parts}
                    new_messages.append(msg)
                if drop_record:
                    break
                new_branches.append({**branch, "messages": new_messages})

            if drop_record:
                continue

            record["conversation_branches"] = new_branches

        records.append(record)

    # Sample if num_samples specified
    if num_samples and num_samples > 0 and len(records) > num_samples:
        random.shuffle(records)
        records = records[:num_samples]

    print(f"Loaded {len(records)} samples from {dataset_name}")
    return records


def load_hf_hub_records(
    dataset_id: str,
    dataset_name: str,
    filter_field: Optional[str] = None,
    filter_value: Any = None,
    num_samples: Optional[int] = None,
    messages_field: str = "conversations"
) -> List[Dict[str, Any]]:
    """Load and convert HuggingFace Hub dataset.

    Supports two message formats:
    - conversations/ShareGPT style: {"from": "human"/"gpt", "value": "..."}
    - standard messages style:      {"role": "user"/"assistant", "content": "..."}
    """
    print(f"Loading HF Hub dataset: {dataset_id}")
    # Local paths saved with save_to_disk need load_from_disk; Hub IDs use load_dataset
    if Path(dataset_id).exists():
        ds = load_from_disk(dataset_id)
    else:
        ds = load_dataset(dataset_id)

    # Handle DatasetDict vs Dataset
    if hasattr(ds, 'keys'):
        ds = ds['train']

    # Pre-filter using HF's native vectorised filter before the conversion loop
    if filter_field is not None:
        print(f"Filtering by {filter_field}={filter_value}...")
        def _matches(x):
            val = x.get(filter_field)
            # Columns sometimes stored as string 'True'/'False' instead of bool
            if isinstance(filter_value, bool) and isinstance(val, str):
                val = val.lower() == "true"
            return val == filter_value
        ds = ds.filter(_matches)
        print(f"{len(ds)} entries after filtering")

    records = []

    # Role mapping covers both ShareGPT and standard role names
    role_map = {
        "human": "user",
        "user": "user",
        "gpt": "assistant",
        "assistant": "assistant",
        "system": "system",
    }

    print(f"Processing {len(ds)} entries...")
    for i in tqdm(range(len(ds)), desc=f"Converting {dataset_name}"):
        sample = ds[i]

        conversations = sample.get(messages_field, [])
        if not conversations or not isinstance(conversations, list):
            continue

        # Normalise both from/value (ShareGPT) and role/content (standard) formats
        messages = []
        for msg in conversations:
            role_raw = msg.get("from") or msg.get("role", "user")
            role = role_map.get(role_raw, role_raw)
            content = msg.get("value") or msg.get("content", "")

            if content:
                messages.append({"role": role, "content": content})

        if not messages:
            continue

        # Ensure system message first
        if messages[0].get("role") != "system":
            messages = [{"role": "system", "content": ""}] + messages

        # Get original metadata
        original_metadata = {k: v for k, v in sample.items() if k != messages_field}

        # Convert to standard format
        record = convert_simple_messages_to_standard(
            messages=messages,
            dataset_source=dataset_name,
            original_id=str(i),
            original_metadata=original_metadata
        )

        if record:
            records.append(record)

    # Sample if num_samples specified
    if num_samples and num_samples > 0 and len(records) > num_samples:
        random.shuffle(records)
        records = records[:num_samples]

    print(f"Loaded {len(records)} samples from {dataset_name}")
    return records


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Convert datasets to standard post-training format with conversation_branches"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASET_CONFIG.keys()),
        help=f"Datasets to include. Available: {list(DATASET_CONFIG.keys())}"
    )
    parser.add_argument(
        "--output-dir",
        default="/iopsstor/scratch/cscs/hyukhymenko/sft-1.1-mixes/if-mix-17-03",
        help="Output directory for HuggingFace dataset"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling"
    )

    args = parser.parse_args()
    random.seed(args.seed)

    all_records = []

    for dataset_name in args.datasets:
        if dataset_name not in DATASET_CONFIG:
            print(f"Warning: Unknown dataset '{dataset_name}', skipping...")
            continue

        config = DATASET_CONFIG[dataset_name]
        print("\n" + "="*60)
        print(f"Processing: {dataset_name}")
        print("="*60)

        if config["type"] == "jsonl":
            records = load_jsonl_records(
                input_path=config["path"],
                dataset_name=dataset_name,
                filter_field=config.get("filter_field"),
                filter_value=config.get("filter_value"),
                num_samples=config.get("num_samples"),
                messages_field=config.get("messages_field", "messages")
            )
        elif config["type"] == "hf":
            records = load_hf_records(
                input_path=config["path"],
                dataset_name=dataset_name,
                filter_field=config.get("filter_field"),
                filter_value=config.get("filter_value"),
                num_samples=config.get("num_samples")
            )
        elif config["type"] == "hf_passthrough":
            records = load_hf_passthrough_records(
                input_path=config["path"],
                dataset_name=dataset_name,
                filter_field=config.get("filter_field"),
                filter_value=config.get("filter_value"),
                num_samples=config.get("num_samples")
            )
        elif config["type"] == "hf_hub":
            records = load_hf_hub_records(
                dataset_id=config["path"],
                dataset_name=dataset_name,
                filter_field=config.get("filter_field"),
                filter_value=config.get("filter_value"),
                num_samples=config.get("num_samples"),
                messages_field=config.get("messages_field", config.get("conversations_field", "conversations"))
            )
        else:
            print(f"Unknown type '{config['type']}' for {dataset_name}, skipping...")
            continue

        all_records.extend(records)

    print("\n" + "="*60)
    print(f"Creating HuggingFace dataset with {len(all_records)} total samples")
    print("="*60)

    # Shuffle all records
    random.shuffle(all_records)

    # Create HF Dataset
    dataset = Dataset.from_list(all_records)
    dataset_dict = DatasetDict({"train": dataset})

    # Save to disk
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(output_path))

    # Print summary
    print(f"\nSaved to: {output_path}")
    print(f"Total samples: {len(all_records)}")
    print("\nPer-dataset breakdown:")
    source_counts = {}
    for r in all_records:
        src = r["dataset_source"]
        source_counts[src] = source_counts.get(src, 0) + 1
    for src, count in sorted(source_counts.items()):
        print(f"  {src}: {count}")

    print("\nOutput schema:")
    print("  - conversation_id: str")
    print("  - dataset_source: str")
    print("  - original_metadata: str (JSON)")
    print("  - created_timestamp: str (ISO)")
    print("  - system_prompt: {content: str, metadata: str}")
    print("  - initial_prompt: {role: str, content: str, metadata: str}")
    print("  - available_functions: [{name, description, parameters}]")
    print("  - conversation_branches: [{messages: [{role, parts: [{type, content, ...}]}]}]")
    print("="*60)


if __name__ == "__main__":
    main()
