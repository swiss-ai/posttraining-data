#!/usr/bin/env python3
"""
Convert Toucan-1.5M (SFT subset) to the standardized chat format.

Toucan (Agent-Ark/Toucan-1.5M) is a large-scale agentic tool-use dataset synthesised
from real MCP servers. This converter targets the curated **SFT** subset, whose columns
are all JSON-encoded strings:

  uuid          : unique id (str, uuid4)
  subset_name   : one of {irrelevant, multi-turn, single-turn-original,
                  single-turn-diversify}
  question      : the user task (== the first user message)
  target_tools  : seed tools used to generate the question
  tools         : OpenAI function defs -> [{"type":"function","function":{name,
                  description, parameters}}, ...]
  messages      : the trajectory -> [{"role", "content"}, ...] with a custom 4-role
                  taxonomy: user / assistant / tool_call / tool_response

Notable properties of the SFT subset (verified by inspection of all 119,287 rows):
  * No `system` role anywhere -> system_prompt is left EMPTY. The Apertus chat template
    injects its own canonical default system prompt when none is provided, and this
    matches the house convention (carry source system prompt if present, else "").
  * No thinking / <think> markup -> no `thought` parts are produced.
  * `tool_call` content is a Python-repr string (json.loads fails, ast.literal_eval
    works) of {"name", "arguments"}, where `arguments` is itself a JSON string.
  * Each tool call and each tool output is its OWN message. Parallel calls appear as a
    run of [tool_call x k, tool_response x k] in index-matched order.

Parallel-call handling (important):
  `07-dataset-aggregation/linearise-dataset.py` cannot represent
  [call, call, output, output] -- on the second output its tool-call buffer is already
  flushed, so it warns and *breaks*, dropping the rest of the turn. We therefore
  INTERLEAVE parallel calls into [call_1, output_1, call_2, output_2, ...] by pairing
  each tool_call with its tool_response in FIFO order (the data confirms this ordering
  is correct). This is the only representation the downstream linearised format / Apertus
  chat template supports.

Output: the unified parts schema (see 02-standardisation/README.md), ready for
stage-07 linearisation.

Example:
  .venv/bin/python 02-standardisation/convert-toucan.py \\
    data/01-hf-data/Toucan-1.5M -o data/02-standardised/
"""

import sys
import ast
import json
import argparse
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

from tqdm import tqdm
from datasets import load_from_disk, DatasetDict, Dataset

DATASET_SOURCE = "Toucan-1.5M"

# Roles in Toucan's `messages` that belong to a single assistant turn.
ASSISTANT_SIDE_ROLES = {"assistant", "tool_call", "tool_response"}


def generate_conversation_id(dataset_source: str, seed: str) -> str:
    """Generate a unique conversation ID (sha256[:12] of a stable seed)."""
    content_hash = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:12]
    prefix = dataset_source.replace("/", "_").replace("-", "_")
    return f"{prefix}_{content_hash}"


def create_unified_part(
    part_type: str,
    content: str = "",
    name: str = "",
    args: str = "",
    metadata: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Create a unified part with all required fields for Arrow schema compatibility.

    All parts share an identical field set (empty strings for unused fields) to prevent
    Arrow schema conflicts when datasets are concatenated in stage 07. Mirrors the part
    schema used by the sibling tool-use converters (apigen / xlam).
    """
    return {
        "type": part_type,
        "content": content,
        "metadata": metadata or {},
        "name": name,
        "args": args,
        "answers": [],  # present for compatibility with the verifiable-responses schema
    }


def parse_tool_call(content: str) -> Tuple[str, str]:
    """
    Parse a `tool_call` message's content into (name, args_json_string).

    Toucan stores this as a Python-repr string, e.g.
        "{'name': 'srv-find_rhymes', 'arguments': '{\"input_word\": \"smile\"}'}"
    so ast.literal_eval is the primary parser; json.loads is a fallback for any rows
    that happen to be valid JSON. `arguments` is already a JSON string and is passed
    through unchanged (the linearised format / Apertus template expect a JSON string).
    """
    parsed: Any = None
    try:
        parsed = ast.literal_eval(content)
    except (ValueError, SyntaxError):
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            return "", ""

    if not isinstance(parsed, dict):
        return "", ""

    name = (parsed.get("name") or "").strip()
    args = parsed.get("arguments", "")

    if isinstance(args, str):
        pass  # already a JSON string -> keep as-is
    elif args is None:
        args = ""
    else:
        # dict / list / other -> serialise to a JSON string for consistency
        try:
            args = json.dumps(args)
        except (TypeError, ValueError):
            args = ""

    return name, args


def convert_tools(tools_json_str: str) -> List[Dict[str, Any]]:
    """
    Convert Toucan's `tools` column (OpenAI function defs) to `available_functions`.

    Toucan nests each def under a "function" key: {"type":"function","function":{...}}.
    `parameters` (JSON Schema) is serialised to a JSON string for Arrow consistency,
    matching the sibling tool-use converters.
    """
    try:
        tools = json.loads(tools_json_str)
    except (json.JSONDecodeError, TypeError):
        return []

    if not isinstance(tools, list):
        return []

    available_functions = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue

        fn = tool.get("function", tool)  # unwrap the OpenAI "function" envelope
        if not isinstance(fn, dict):
            continue

        name = (fn.get("name") or "").strip()
        if not name:
            continue

        params = fn.get(
            "parameters", {"type": "object", "properties": {}, "required": []}
        )
        available_functions.append(
            {
                "name": name,
                "description": fn.get("description", "") or "",
                "parameters": json.dumps(params),
            }
        )

    return available_functions


def build_assistant_parts(
    assistant_msgs: List[Dict[str, Any]], conv_id: str
) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Convert a run of assistant-side messages (assistant / tool_call / tool_response)
    into an ordered parts list, interleaving parallel calls with their outputs.

    Returns (parts, had_anomaly). `had_anomaly` flags structurally odd turns
    (orphan outputs, calls left unmatched, text amid pending calls) for reporting.
    """
    parts: List[Dict[str, Any]] = []
    pending_calls: List[Tuple[str, str]] = []  # FIFO of (name, args)
    had_anomaly = False

    for msg in assistant_msgs:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "assistant":
            # Assistant free-text (pre-amble before calls, or final answer after outputs).
            if pending_calls:
                # Calls without outputs, then text -> not expected in Toucan SFT. Flush the
                # calls so nothing is lost; linearise may warn on this rare shape.
                had_anomaly = True
                for name, args in pending_calls:
                    parts.append(create_unified_part("function-call", name=name, args=args))
                pending_calls = []
            parts.append(create_unified_part("response", content=content if isinstance(content, str) else json.dumps(content)))

        elif role == "tool_call":
            name, args = parse_tool_call(content)
            if name:
                pending_calls.append((name, args))
            else:
                had_anomaly = True

        elif role == "tool_response":
            output = content if isinstance(content, str) else json.dumps(content)
            if pending_calls:
                # Pair this output with the oldest unmatched call -> [call, output].
                name, args = pending_calls.pop(0)
                parts.append(create_unified_part("function-call", name=name, args=args))
                parts.append(create_unified_part("function-output", content=output))
            else:
                # Output with no pending call -> drop (cannot be linearised); flag it.
                had_anomaly = True

    # Trailing calls that never received an output: emit them (linearise flushes a final
    # tool_calls block without breaking). Flag as anomaly for reporting.
    if pending_calls:
        had_anomaly = True
        for name, args in pending_calls:
            parts.append(create_unified_part("function-call", name=name, args=args))

    return parts, had_anomaly


def group_assistant_turns(
    msgs: List[Dict[str, Any]],
) -> List[Tuple[str, List[Dict[str, Any]]]]:
    """
    Split the post-initial-prompt messages into ordered turn groups: each group is either
    a single user message or a contiguous run of assistant-side messages.
    """
    groups: List[Tuple[str, List[Dict[str, Any]]]] = []
    current: List[Dict[str, Any]] = []

    for msg in msgs:
        role = msg.get("role", "")
        if role == "user":
            if current:
                groups.append(("assistant", current))
                current = []
            groups.append(("user", [msg]))
        elif role in ASSISTANT_SIDE_ROLES:
            current.append(msg)
        # silently ignore any other/unknown roles

    if current:
        groups.append(("assistant", current))

    return groups


def convert_sample(row: Dict[str, Any], idx: int) -> Tuple[Optional[Dict[str, Any]], bool]:
    """
    Convert a single Toucan SFT row to the unified chat format.

    Returns (chat_sample_or_None, had_anomaly).
    """
    try:
        messages = json.loads(row["messages"])
    except (json.JSONDecodeError, TypeError, KeyError):
        return None, False

    if not isinstance(messages, list) or not messages:
        return None, False

    # First user message -> initial_prompt; everything after -> conversation branch.
    first_user_idx = next(
        (i for i, m in enumerate(messages) if isinstance(m, dict) and m.get("role") == "user"),
        None,
    )
    if first_user_idx is None:
        return None, False

    initial_prompt_content = messages[first_user_idx].get("content", "") or ""
    rest = messages[first_user_idx + 1:]

    conversation_messages: List[Dict[str, Any]] = []
    had_anomaly = False
    conv_id = generate_conversation_id(DATASET_SOURCE, row.get("uuid", "") or initial_prompt_content)

    for kind, group in group_assistant_turns(rest):
        if kind == "user":
            content = group[0].get("content", "") or ""
            conversation_messages.append(
                {
                    "role": "user",
                    "parts": [create_unified_part("response", content=content)],
                    "metadata": {},
                }
            )
        else:  # assistant-side run
            parts, anomaly = build_assistant_parts(group, conv_id)
            had_anomaly = had_anomaly or anomaly
            if parts:
                conversation_messages.append(
                    {"role": "assistant", "parts": parts, "metadata": {}}
                )

    if not conversation_messages:
        return None, had_anomaly

    chat_sample = {
        "conversation_id": conv_id,
        "dataset_source": DATASET_SOURCE,
        "original_metadata": {
            "uuid": row.get("uuid", "") or "",
            "subset_name": row.get("subset_name", "") or "",
            "target_tools": row.get("target_tools", "") or "",
            "row_index": idx,
        },
        "system_prompt": {"content": "", "metadata": {}},
        "initial_prompt": {
            "role": "user",
            "content": initial_prompt_content,
            "metadata": {},
        },
        "available_functions": convert_tools(row.get("tools", "[]")),
        "conversation_branches": [{"messages": conversation_messages}],
        "created_timestamp": datetime.now().isoformat(),
    }

    return chat_sample, had_anomaly


def process_dataset(dataset: Dataset, chunk_size: int = 2000) -> Dataset:
    """Convert a split to the unified chat format."""
    total = len(dataset)
    print(f"Converting {total:,} samples...")

    converted: List[Dict[str, Any]] = []
    failed = 0
    anomalies = 0

    with tqdm(total=total, desc="Converting Toucan", unit="samples") as pbar:
        for start in range(0, total, chunk_size):
            end = min(start + chunk_size, total)
            for idx in range(start, end):
                sample = dataset[idx]
                chat_sample, anomaly = convert_sample(sample, idx)
                if chat_sample is not None:
                    converted.append(chat_sample)
                    if anomaly:
                        anomalies += 1
                else:
                    failed += 1
                pbar.update(1)
                pbar.set_postfix(failed=failed, anomalies=anomalies)

    print(
        f"Conversion complete: {len(converted):,} converted, {failed:,} skipped, "
        f"{anomalies:,} with structural anomalies"
    )
    if not converted:
        return Dataset.from_list([])
    return Dataset.from_list(converted)


def save_dataset_and_metadata(
    dataset_dict: DatasetDict, output_path: Path, input_path: Path
):
    """Save converted dataset with processing metadata."""
    output_path.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(output_path))

    metadata = {
        "processing_log": [
            {
                "operation": "convert_toucan",
                "script": "convert-toucan.py",
                "timestamp": datetime.now().isoformat(),
                "input_path": str(input_path),
                "output_path": str(output_path),
                "description": (
                    "Converted Toucan-1.5M SFT subset to the unified chat format. "
                    "Empty system_prompt (Apertus template supplies the default); tools "
                    "from the `tools` column; parallel tool calls interleaved with their "
                    "outputs for linearise compatibility."
                ),
            }
        ],
        "format": "chat_format_v1_new",
        "source_dataset": DATASET_SOURCE,
        "conversion_details": {
            "function_format": "openai_compatible",
            "conversation_type": "multi_turn_agentic_tool_use",
            "parts_structure": "unified_schema",
            "special_features": ["multi_turn", "function_calling", "parallel_call_interleaving"],
        },
    }
    with open(output_path / "dataset_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Dataset saved to {output_path}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Toucan-1.5M (SFT subset) to the standardized chat format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  .venv/bin/python 02-standardisation/convert-toucan.py \\\n"
            "    data/01-hf-data/Toucan-1.5M -o data/02-standardised/\n"
        ),
    )
    parser.add_argument(
        "input_path", type=str, help="Path to the Toucan SFT dataset (load_from_disk format)"
    )
    parser.add_argument(
        "-o", "--output", type=str, required=True, help="Output directory for converted dataset"
    )
    parser.add_argument(
        "--batch-size", type=int, default=2000, help="Chunk size for processing (default: 2000)"
    )
    parser.add_argument(
        "--sample", type=int, default=None, help="Only convert the first N samples (for testing)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    input_path = Path(args.input_path)
    if not input_path.exists():
        print(f"Error: input path does not exist: {input_path}")
        sys.exit(1)

    # If output is a directory (trailing slash or existing dir), append the input name.
    output_path = Path(args.output)
    if args.output.endswith("/") or (output_path.exists() and output_path.is_dir()):
        output_path = output_path / input_path.name

    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")

    try:
        dataset = load_from_disk(str(input_path))
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    if not isinstance(dataset, DatasetDict):
        dataset = DatasetDict({"train": dataset})

    processed_splits = {}
    for split_name, split_dataset in dataset.items():
        if args.sample:
            split_dataset = split_dataset.select(range(min(args.sample, len(split_dataset))))
            print(f"Sampling first {len(split_dataset):,} examples of '{split_name}'")
        print(f"Processing '{split_name}': {len(split_dataset):,} samples")
        processed_splits[split_name] = process_dataset(split_dataset, chunk_size=args.batch_size)

    processed_dataset = DatasetDict(processed_splits)
    save_dataset_and_metadata(processed_dataset, output_path, input_path)

    total = sum(len(s) for s in processed_dataset.values())
    print(f"\nConversion completed successfully! Total samples: {total:,}")
    if "train" in processed_dataset:
        print(f"Features: {list(processed_dataset['train'].features.keys())}")


if __name__ == "__main__":
    main()
