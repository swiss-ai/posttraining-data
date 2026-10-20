#!/usr/bin/env python3
"""
Count display_answers tool calls per dataset source, split by whether
display_answers is declared in the developer message's tools.

Usage:
    python tool_data/find-display-answers-datasets.py /path/to/linearised-dataset --output report.json
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import datasets
from datasets import load_from_disk, DatasetDict

datasets.disable_caching()


def check_display_answers(sample):
    has_call = False
    for msg in sample["messages"]:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", {})
        if not isinstance(content, dict):
            continue
        for blk in content.get("blocks", []):
            if not isinstance(blk, dict) or blk.get("type") != "tool_calls":
                continue
            for call in blk.get("calls", []):
                if isinstance(call, dict) and call.get("name") == "display_answers":
                    has_call = True
                    break
            if has_call:
                break
        if has_call:
            break

    if not has_call:
        return {"has_display_answers": False, "display_answers_declared": False}

    declared = False
    for msg in sample["messages"]:
        if msg.get("role") != "developer":
            continue
        content = msg.get("content", {})
        if not isinstance(content, dict):
            continue
        tools_str = content.get("tools", "")
        if not tools_str or (isinstance(tools_str, str) and not tools_str.strip()):
            break
        try:
            tools = json.loads(tools_str) if isinstance(tools_str, str) else tools_str
            for t in tools:
                if isinstance(t, dict) and t.get("name") == "display_answers":
                    declared = True
                    break
        except (json.JSONDecodeError, TypeError):
            pass
        break

    return {"has_display_answers": True, "display_answers_declared": declared}


def main():
    parser = argparse.ArgumentParser(
        description="Find all dataset sources containing display_answers tool calls"
    )
    parser.add_argument("input_path", help="Path to linearised dataset")
    parser.add_argument("--output", default=None,
                        help="Path for output JSON report")
    parser.add_argument("--num-proc", type=int, default=16,
                        help="Number of parallel workers for map (default: 16)")
    args = parser.parse_args()

    print(f"Loading dataset from: {args.input_path}")
    dataset = load_from_disk(args.input_path)
    if not isinstance(dataset, DatasetDict):
        dataset = DatasetDict({"train": dataset})

    split = list(dataset.keys())[0]
    ds = dataset[split]
    print(f"Total samples: {len(ds):,}")

    print(f"\nScanning for display_answers calls (num_proc={args.num_proc})...")
    ds_checked = ds.map(check_display_answers, num_proc=args.num_proc,
                        desc="Scanning for display_answers")

    all_sources = ds_checked["dataset_source"]
    all_has_da = ds_checked["has_display_answers"]
    all_declared = ds_checked["display_answers_declared"]

    undeclared_by_source = Counter()
    declared_by_source = Counter()

    for src, has_da, declared in zip(all_sources, all_has_da, all_declared):
        if not has_da:
            continue
        if declared:
            declared_by_source[src] += 1
        else:
            undeclared_by_source[src] += 1

    total_da = sum(undeclared_by_source.values()) + sum(declared_by_source.values())
    print(f"\nTotal samples with display_answers calls: {total_da:,}")

    undeclared = dict(undeclared_by_source.most_common())
    declared = dict(declared_by_source.most_common())
    for title, counts in (("UNDECLARED (display_answers not in developer tools)", undeclared),
                          ("DECLARED (display_answers in developer tools)", declared)):
        print(f"\n{title}:")
        print(f"  {'Dataset':<50s} {'Samples':>10s}")
        print("  " + "-" * 60)
        if counts:
            for src, count in counts.items():
                print(f"  {src:<50s} {count:>10,d}")
            print(f"  {'TOTAL':<50s} {sum(counts.values()):>10,d}")
        else:
            print("  (none found)")

    if args.output:
        report = {
            "total_with_display_answers": total_da,
            "undeclared": undeclared,
            "undeclared_total": sum(undeclared.values()),
            "declared": declared,
            "declared_total": sum(declared.values()),
        }
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport written to: {output_path}")


if __name__ == "__main__":
    main()
