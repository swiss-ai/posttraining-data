#!/usr/bin/env python3
"""
Analyze samples flagged with tool_use_without_declarations.

For each flagged sample, identifies which tools are called (without being
declared in the developer message) and collects statistics on tool names,
call counts, and tool output content.

Usage:
    python tool_data/analyze-undeclared-tool-usage.py /path/to/linearised-dataset \
        --flagged-ids flagged-ids.json --output report.json [--limit 1000]
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_from_disk, DatasetDict


def extract_developer_tools(messages):
    for msg in messages:
        if msg.get("role") != "developer":
            continue
        content = msg.get("content", {})
        if not isinstance(content, dict):
            continue
        tools_str = content.get("tools", "")
        if not tools_str or (isinstance(tools_str, str) and not tools_str.strip()):
            return set()
        try:
            tools = json.loads(tools_str) if isinstance(tools_str, str) else tools_str
            return {t["name"] for t in tools if isinstance(t, dict) and t.get("name")}
        except (json.JSONDecodeError, TypeError):
            return set()
    return set()


def analyze_sample(messages, declared_tools):
    """Undeclared calls with their output, matched by position in the following tool_outputs block."""
    calls = []

    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", {})
        if not isinstance(content, dict):
            continue
        blocks = content.get("blocks", [])
        if not isinstance(blocks, list):
            continue

        for i, blk in enumerate(blocks):
            if not isinstance(blk, dict) or blk.get("type") != "tool_calls":
                continue

            call_list = blk.get("calls", [])
            output_list = []
            if i + 1 < len(blocks):
                next_blk = blocks[i + 1]
                if isinstance(next_blk, dict) and next_blk.get("type") == "tool_outputs":
                    output_list = next_blk.get("outputs", [])

            for ci, call in enumerate(call_list):
                if not isinstance(call, dict) or not call.get("name"):
                    continue
                name = call["name"]
                if name in declared_tools:
                    continue

                args = call.get("arguments", "") or ""
                output_text = ""
                if ci < len(output_list):
                    o = output_list[ci]
                    if isinstance(o, dict):
                        output_text = o.get("output", "") or ""

                calls.append({
                    "name": name,
                    "args": args,
                    "output": output_text,
                })

    return calls


def classify_output(text):
    if not text or not text.strip():
        return "empty"
    t = text.strip()
    if t.lower() in ("true", "false"):
        return "boolean"
    if len(t) < 50:
        return "short_text"
    return "long_text"


def main():
    parser = argparse.ArgumentParser(
        description="Analyze undeclared tool usage in linearised datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input_path", help="Path to linearised dataset")
    parser.add_argument("--flagged-ids", required=True,
                        help="Path to flagged IDs JSON from validate-linearised.py")
    parser.add_argument("--output", required=True,
                        help="Path for output report JSON")
    parser.add_argument("--split", default=None,
                        help="Dataset split (default: first available)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max unique samples to analyze")
    args = parser.parse_args()

    flagged_path = Path(args.flagged_ids)
    if not flagged_path.exists():
        print(f"Error: flagged IDs file not found: {flagged_path}", file=sys.stderr)
        sys.exit(1)

    with open(flagged_path) as f:
        flagged_data = json.load(f)

    print(f"Loading dataset from: {args.input_path}")
    dataset = load_from_disk(args.input_path)
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

    raw_ids = flagged_data[split].get("tool_use_without_declarations", [])
    unique_ids = list(dict.fromkeys(raw_ids))
    print(f"Total flagged instances: {len(raw_ids)}")
    print(f"Unique samples: {len(unique_ids)}")

    if args.limit:
        unique_ids = unique_ids[:args.limit]
        print(f"Limiting to first {args.limit} unique samples")

    target_set = set(unique_ids)
    print("Building ID -> index mapping...")
    all_cids = ds["conversation_id"]
    id_to_idx = {cid: i for i, cid in enumerate(all_cids) if cid in target_set}

    missing = target_set - set(id_to_idx.keys())
    if missing:
        print(f"Warning: {len(missing)} flagged IDs not found in dataset")

    print(f"Analyzing {len(id_to_idx)} samples...")

    tool_call_counts = Counter()
    tool_sample_counts = Counter()
    tool_output_categories = defaultdict(Counter)
    tool_output_examples = defaultdict(list)
    tool_args_examples = defaultdict(list)
    dataset_source_counts = defaultdict(Counter)
    samples_per_source = Counter()

    for progress, cid in enumerate(unique_ids):
        if cid not in id_to_idx:
            continue

        sample = ds[id_to_idx[cid]]
        messages = sample["messages"]
        source = sample.get("dataset_source", "unknown")
        declared_tools = extract_developer_tools(messages)
        calls = analyze_sample(messages, declared_tools)

        seen_tools = set()
        samples_per_source[source] += 1

        for call in calls:
            name = call["name"]
            tool_call_counts[name] += 1
            dataset_source_counts[source][name] += 1

            if name not in seen_tools:
                tool_sample_counts[name] += 1
                seen_tools.add(name)

            out_cat = classify_output(call["output"])
            tool_output_categories[name][out_cat] += 1

            if len(tool_output_examples[name]) < 5:
                tool_output_examples[name].append(call["output"][:500])

            if len(tool_args_examples[name]) < 5:
                tool_args_examples[name].append(call["args"][:500])

        if (progress + 1) % 5000 == 0:
            print(f"  processed {progress + 1}/{len(unique_ids)}...")

    tools_report = []
    for name in sorted(tool_call_counts, key=tool_call_counts.get, reverse=True):
        tools_report.append({
            "tool_name": name,
            "total_calls": tool_call_counts[name],
            "unique_samples": tool_sample_counts[name],
            "output_categories": dict(tool_output_categories[name]),
            "args_examples": tool_args_examples[name],
            "output_examples": tool_output_examples[name],
        })

    sources_report = []
    for source in sorted(samples_per_source, key=samples_per_source.get, reverse=True):
        sources_report.append({
            "dataset_source": source,
            "unique_samples": samples_per_source[source],
            "tools_used": dict(dataset_source_counts[source]),
        })

    report = {
        "total_flagged_instances": len(raw_ids),
        "unique_samples_analyzed": len(id_to_idx),
        "total_undeclared_calls": sum(tool_call_counts.values()),
        "unique_tool_names": len(tool_call_counts),
        "tools": tools_report,
        "by_dataset_source": sources_report,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print()
    print("=" * 65)
    print("  Undeclared Tool Usage Analysis")
    print("=" * 65)
    print()
    print(f"  Unique samples analyzed:  {len(id_to_idx):,}")
    print(f"  Total undeclared calls:   {sum(tool_call_counts.values()):,}")
    print(f"  Unique tool names:        {len(tool_call_counts)}")
    print()
    print("  Tool breakdown:")
    print("  " + "-" * 61)
    for entry in tools_report:
        name = entry["tool_name"]
        calls = entry["total_calls"]
        samples = entry["unique_samples"]
        cats = entry["output_categories"]
        cats_str = ", ".join(f"{k}={v}" for k, v in sorted(cats.items()))
        print(f"    {name:<30s}  calls={calls:>8,d}  samples={samples:>8,d}")
        print(f"      outputs: {cats_str}")
    print()
    print("  By dataset source:")
    print("  " + "-" * 61)
    for entry in sources_report:
        src = entry["dataset_source"]
        n = entry["unique_samples"]
        tools = entry["tools_used"]
        tools_str = ", ".join(f"{k}={v}" for k, v in sorted(tools.items()))
        print(f"    {src:<40s}  samples={n:>8,d}")
        print(f"      tools: {tools_str}")
    print()
    print(f"  Report written to: {output_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
