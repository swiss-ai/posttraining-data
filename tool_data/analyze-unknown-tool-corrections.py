#!/usr/bin/env python3
"""
Analyze correction behavior after unknown tool calls in linearised datasets.

For each sample flagged with call_unknown_tool, determines what happens after
the assistant calls a tool not in the declared list: does it retry with the
correct name, try a different unknown name, or give up with a text response?

Usage:
    python tool_data/analyze-unknown-tool-corrections.py /path/to/linearised-dataset \
        --flagged-ids flagged-ids.json [--verbose]
"""

import json
import argparse
import sys
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Any, Optional, Set, Tuple

from datasets import load_from_disk, DatasetDict


def normalize_name(name: str) -> str:
    return name.lower().replace("-", "_").replace(" ", "_")


def classify_mismatch(unknown: str, declared_names: Set[str]) -> str:
    norm_unknown = normalize_name(unknown)
    if any(normalize_name(d) == norm_unknown for d in declared_names):
        return "separator_mismatch"
    if any(normalize_name(d).endswith(norm_unknown) for d in declared_names):
        return "prefix_stripped"
    return "other"


def args_similarity(args1: str, args2: str) -> float:
    """Similarity in [0, 1]: JSON keys and values if both parse as objects, else characters."""
    if not args1 and not args2:
        return 1.0
    if not args1 or not args2:
        return 0.0

    try:
        j1 = json.loads(args1)
        j2 = json.loads(args2)
        if isinstance(j1, dict) and isinstance(j2, dict):
            keys1 = set(j1.keys())
            keys2 = set(j2.keys())
            if not keys1 and not keys2:
                return 1.0
            union = keys1 | keys2
            intersection = keys1 & keys2
            key_sim = len(intersection) / len(union)
            val_matches = sum(1 for k in intersection if j1[k] == j2[k])
            val_sim = val_matches / len(intersection) if intersection else 1.0
            return 0.5 * key_sim + 0.5 * val_sim
    except (json.JSONDecodeError, TypeError):
        pass

    if args1 == args2:
        return 1.0
    common = sum(1 for a, b in zip(args1, args2) if a == b)
    return common / max(len(args1), len(args2))


def extract_declared_tools(messages: List[Dict]) -> Set[str]:
    names = set()
    for msg in messages:
        if msg.get("role") != "developer":
            continue
        content = msg.get("content", {})
        if not isinstance(content, dict):
            continue
        tools_str = content.get("tools", "")
        if not tools_str:
            continue
        try:
            tools = json.loads(tools_str) if isinstance(tools_str, str) else tools_str
            for t in tools:
                if isinstance(t, dict) and t.get("name"):
                    names.add(t["name"])
        except (json.JSONDecodeError, TypeError):
            pass
    return names


def extract_all_blocks(messages: List[Dict]) -> List[Tuple[int, int, Dict]]:
    """All assistant blocks as (msg_idx, block_idx, block)."""
    results = []
    for mi, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", {})
        if not isinstance(content, dict):
            continue
        blocks = content.get("blocks", [])
        if not isinstance(blocks, list):
            continue
        for bi, block in enumerate(blocks):
            if isinstance(block, dict):
                results.append((mi, bi, block))
    return results


CAT_CORRECTED_NAME_SAME_ARGS = "A1_corrected_name_same_args"
CAT_CORRECTED_NAME_DIFF_ARGS = "A2_corrected_name_diff_args"
CAT_RETRIED_DIFFERENT_UNKNOWN = "B_retried_different_unknown"
CAT_TEXT_RESPONSE = "C_text_response"
CAT_ERROR_THEN_RESPONSE = "D_error_then_response"
CAT_NO_FURTHER_ACTION = "E_no_further_action"

CATEGORY_LABELS = {
    CAT_CORRECTED_NAME_SAME_ARGS: "Corrected name, same/similar args",
    CAT_CORRECTED_NAME_DIFF_ARGS: "Corrected name, different args",
    CAT_RETRIED_DIFFERENT_UNKNOWN: "Retried with different unknown tool",
    CAT_TEXT_RESPONSE: "Text response only (no retry)",
    CAT_ERROR_THEN_RESPONSE: "Error in output, then text response",
    CAT_NO_FURTHER_ACTION: "No further action / conversation ended",
}


def classify_tool_output(output_block: Optional[Dict]) -> str:
    if output_block is None:
        return "missing"
    outputs = output_block.get("outputs", [])
    if not isinstance(outputs, list) or not outputs:
        return "empty"
    for o in outputs:
        if not isinstance(o, dict):
            continue
        text = o.get("output", "")
        if not text:
            return "empty"
        text_lower = text.lower()
        if any(kw in text_lower for kw in
               ("does not exist", "not found", "error", "unknown tool",
                "no such tool", "invalid tool")):
            return "error"
        return "data"
    return "empty"


def analyze_unknown_call(
    unknown_call_name: str,
    unknown_call_args: str,
    all_blocks: List[Tuple[int, int, Dict]],
    block_position: int,
    declared_names: Set[str],
) -> Tuple[str, Dict[str, Any]]:
    """Classify what follows the unknown call at all_blocks[block_position]."""
    detail: Dict[str, Any] = {
        "unknown_name": unknown_call_name,
        "unknown_args_preview": unknown_call_args[:120] if unknown_call_args else "",
    }

    output_block = None
    output_pos = block_position + 1
    if output_pos < len(all_blocks):
        _, _, candidate = all_blocks[output_pos]
        if candidate.get("type") == "tool_outputs":
            output_block = candidate

    output_class = classify_tool_output(output_block)
    detail["output_class"] = output_class

    search_start = output_pos + 1 if output_block else block_position + 1

    next_call = None
    next_response = None

    for idx in range(search_start, len(all_blocks)):
        _, _, blk = all_blocks[idx]
        btype = blk.get("type")
        if btype == "tool_calls":
            real_calls = [c for c in blk.get("calls", [])
                          if isinstance(c, dict) and c.get("name")]
            if real_calls:
                next_call = real_calls[0]
                break
        elif btype == "response" and next_response is None:
            text = blk.get("text", "")
            if text and text.strip():
                next_response = text

    if next_call is not None:
        nc_name = next_call.get("name", "")
        nc_args = next_call.get("arguments", "")
        detail["next_call_name"] = nc_name
        detail["next_call_args_preview"] = nc_args[:120] if nc_args else ""

        a_sim = args_similarity(unknown_call_args, nc_args)
        detail["args_similarity"] = round(a_sim, 3)

        if nc_name not in declared_names:
            return CAT_RETRIED_DIFFERENT_UNKNOWN, detail
        if a_sim >= 0.5:
            return CAT_CORRECTED_NAME_SAME_ARGS, detail
        return CAT_CORRECTED_NAME_DIFF_ARGS, detail

    if next_response is not None:
        if output_class == "error":
            return CAT_ERROR_THEN_RESPONSE, detail
        return CAT_TEXT_RESPONSE, detail

    return CAT_NO_FURTHER_ACTION, detail


def analyze_sample(
    sample: Dict[str, Any],
    declared_names: Set[str],
) -> List[Tuple[str, Dict[str, Any]]]:
    """(category, detail) for each unknown call in the sample."""
    all_blocks = extract_all_blocks(sample.get("messages", []))
    results = []

    for pos, (mi, bi, block) in enumerate(all_blocks):
        if block.get("type") != "tool_calls":
            continue
        for call in block.get("calls", []):
            if not isinstance(call, dict):
                continue
            name = call.get("name", "")
            if not name:
                continue
            if declared_names and name not in declared_names:
                args = call.get("arguments", "")
                category, detail = analyze_unknown_call(
                    name, args, all_blocks, pos, declared_names
                )
                detail["msg_idx"] = mi
                detail["block_idx"] = bi
                results.append((category, detail))

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Analyze correction behavior after unknown tool calls",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input_path", type=str,
                        help="Path to linearised dataset (load_from_disk)")
    parser.add_argument("--flagged-ids", type=str, required=True,
                        help="Path to flagged IDs JSON (from validate-linearised.py)")
    parser.add_argument("--split", type=str, default=None,
                        help="Dataset split to analyze (default: first available)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max samples to analyze (default: all)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-sample details")
    args = parser.parse_args()

    flagged_path = Path(args.flagged_ids)
    if not flagged_path.exists():
        print(f"Error: flagged IDs file not found: {flagged_path}", file=sys.stderr)
        sys.exit(1)

    with open(flagged_path) as f:
        flagged_data = json.load(f)

    input_path = Path(args.input_path)
    print(f"Loading dataset from: {input_path}")
    dataset = load_from_disk(str(input_path))
    if not isinstance(dataset, DatasetDict):
        dataset = DatasetDict({"train": dataset})

    split = args.split or list(dataset.keys())[0]
    if split not in dataset:
        print(f"Error: split '{split}' not found. Available: {list(dataset.keys())}")
        sys.exit(1)

    ds = dataset[split]

    if split not in flagged_data:
        print(f"Error: split '{split}' not found in flagged IDs file")
        sys.exit(1)

    unknown_ids_list = flagged_data[split].get("call_unknown_tool", [])
    unknown_ids = list(dict.fromkeys(unknown_ids_list))
    print(f"Unique samples with call_unknown_tool: {len(unknown_ids)}")
    print(f"Total call_unknown_tool instances in flagged file: {len(unknown_ids_list)}")

    if args.limit:
        unknown_ids = unknown_ids[:args.limit]
        print(f"Limiting to first {args.limit} samples")

    target_set = set(unknown_ids)
    id_to_idx: Dict[str, int] = {
        cid: i for i, cid in enumerate(ds["conversation_id"]) if cid in target_set
    }

    missing = target_set - set(id_to_idx.keys())
    if missing:
        print(f"Warning: {len(missing)} flagged IDs not found in dataset")

    print(f"\nAnalyzing {len(id_to_idx)} samples...\n")

    category_counts: Counter = Counter()
    output_counts: Counter = Counter()
    mismatch_counts: Counter = Counter()
    total_unknown_calls = 0
    category_examples: Dict[str, List[Tuple[str, Dict]]] = defaultdict(list)

    for idx, cid in enumerate(unknown_ids):
        if cid not in id_to_idx:
            continue

        sample = ds[id_to_idx[cid]]
        declared_names = extract_declared_tools(sample["messages"])
        results = analyze_sample(sample, declared_names)

        for category, detail in results:
            total_unknown_calls += 1
            category_counts[category] += 1
            output_counts[detail.get("output_class", "unknown")] += 1

            mismatch_type = classify_mismatch(detail["unknown_name"], declared_names)
            mismatch_counts[mismatch_type] += 1
            detail["mismatch_type"] = mismatch_type

            category_examples[category].append((cid, detail))

        if (idx + 1) % 200 == 0:
            print(f"  processed {idx + 1}/{len(unknown_ids)}...")

    print()
    print("=" * 65)
    print("  Unknown Tool Call Correction Analysis")
    print("=" * 65)
    print()
    print(f"  Total unique samples with call_unknown_tool: {len(id_to_idx)}")
    print(f"  Total unknown call instances analyzed:       {total_unknown_calls}")
    print()

    print("  Outcome breakdown:")
    print("  " + "-" * 61)
    for cat, label in CATEGORY_LABELS.items():
        count = category_counts.get(cat, 0)
        pct = 100.0 * count / total_unknown_calls if total_unknown_calls else 0
        print(f"    {label:<45s} {count:>5d}  ({pct:5.1f}%)")
    print()

    print("  Tool output contents:")
    print("  " + "-" * 61)
    for cls in ("error", "data", "empty", "missing"):
        count = output_counts.get(cls, 0)
        pct = 100.0 * count / total_unknown_calls if total_unknown_calls else 0
        print(f"    {cls:<45s} {count:>5d}  ({pct:5.1f}%)")
    print()

    print("  Name mismatch patterns:")
    print("  " + "-" * 61)
    for mtype in ("prefix_stripped", "separator_mismatch", "other"):
        count = mismatch_counts.get(mtype, 0)
        pct = 100.0 * count / total_unknown_calls if total_unknown_calls else 0
        print(f"    {mtype:<45s} {count:>5d}  ({pct:5.1f}%)")
    print()

    if args.verbose:
        for cat, label in CATEGORY_LABELS.items():
            examples = category_examples.get(cat, [])
            if not examples:
                continue
            print("=" * 65)
            print(f"  {label} ({len(examples)} instances)")
            print("=" * 65)
            for cid, detail in examples[:10]:
                print(f"\n    Sample: {cid}")
                print(f"      Unknown call: {detail['unknown_name']}")
                if detail.get("unknown_args_preview"):
                    print(f"      Args:         {detail['unknown_args_preview']}")
                print(f"      Output:       {detail['output_class']}")
                print(f"      Mismatch:     {detail['mismatch_type']}")
                if "next_call_name" in detail:
                    print(f"      Next call:    {detail['next_call_name']}")
                    if "args_similarity" in detail:
                        print(f"      Args sim:     {detail['args_similarity']}")
            if len(examples) > 10:
                print(f"\n    ... and {len(examples) - 10} more")
            print()

    print("=" * 65)


if __name__ == "__main__":
    main()
