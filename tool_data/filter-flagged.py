#!/usr/bin/env python3
"""
Filter a linearised dataset by removing samples flagged with specific error/warning codes.

Reads a flagged-IDs JSON (produced by validate-linearised.py --flagged-ids) and removes
all samples whose conversation_id matches any of the specified codes.

Usage:
    python tool_data/filter-flagged.py /path/to/linearised-dataset --flagged-ids flagged.json \
        --remove-codes call_unknown_tool call_arguments_json --output /path/to/output [--dry-run]
"""

import json
import argparse
import sys
from pathlib import Path
from collections import Counter

from datasets import load_from_disk, DatasetDict


def load_remove_ids(flagged_path: Path, remove_codes: list, split_filter: str = None):
    """Load flagged JSON and collect conversation_ids to remove per split.

    Returns:
        per_split_ids: dict mapping split -> set of conversation_ids to remove
        per_code_counts: dict mapping split -> Counter of code -> unique id count
        available_codes: set of all codes found in the JSON
    """
    with open(flagged_path) as f:
        flagged = json.load(f)

    available_codes = set()
    for split_data in flagged.values():
        available_codes.update(split_data.keys())

    per_split_ids = {}
    per_code_counts = {}

    splits = [split_filter] if split_filter else list(flagged.keys())
    for split_name in splits:
        if split_name not in flagged:
            continue
        split_data = flagged[split_name]
        remove_set = set()
        code_counts = Counter()
        for code in remove_codes:
            if code not in split_data:
                continue
            ids = set(split_data[code])
            code_counts[code] = len(ids)
            remove_set.update(ids)
        per_split_ids[split_name] = remove_set
        per_code_counts[split_name] = code_counts

    return per_split_ids, per_code_counts, available_codes


def main():
    parser = argparse.ArgumentParser(
        description="Filter a linearised dataset by removing samples flagged with specific codes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input_path", type=str,
                        help="Path to linearised dataset (load_from_disk format)")
    parser.add_argument("--flagged-ids", required=True, type=str,
                        help="Path to flagged JSON from validate-linearised.py")
    parser.add_argument("--remove-codes", required=True, nargs="+", type=str,
                        help="Error/warning codes to filter on")
    parser.add_argument("--output", required=True, type=str,
                        help="Output path for filtered dataset")
    parser.add_argument("--split", type=str, default=None,
                        help="Filter a specific split (default: all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be removed without writing")
    args = parser.parse_args()

    input_path = Path(args.input_path)
    flagged_path = Path(args.flagged_ids)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: input path does not exist: {input_path}")
        sys.exit(1)
    if not flagged_path.exists():
        print(f"Error: flagged-ids path does not exist: {flagged_path}")
        sys.exit(1)

    per_split_ids, per_code_counts, available_codes = load_remove_ids(
        flagged_path, args.remove_codes, args.split
    )

    missing_codes = set(args.remove_codes) - available_codes
    if missing_codes:
        print(f"Warning: codes not found in flagged JSON: {', '.join(sorted(missing_codes))}")
        print(f"  Available codes: {', '.join(sorted(available_codes))}")

    print(f"Loading dataset from: {input_path}")
    dataset = load_from_disk(str(input_path))
    if not isinstance(dataset, DatasetDict):
        dataset = DatasetDict({"train": dataset})

    splits_to_process = [args.split] if args.split else list(dataset.keys())

    filtered_dict = {}
    for split_name in splits_to_process:
        if split_name not in dataset:
            print(f"Warning: split '{split_name}' not in dataset. "
                  f"Available: {list(dataset.keys())}")
            continue

        ds = dataset[split_name]
        remove_set = per_split_ids.get(split_name, set())
        total = len(ds)

        if not remove_set:
            print(f"\nSplit '{split_name}': {total} samples, 0 to remove (no matching IDs)")
            filtered_dict[split_name] = ds
            continue

        conversation_ids = ds["conversation_id"]
        keep_indices = [i for i, cid in enumerate(conversation_ids) if cid not in remove_set]
        removed = total - len(keep_indices)

        code_counts = per_code_counts.get(split_name, Counter())
        code_summary = ", ".join(
            f"{code}={count}" for code, count in code_counts.most_common()
        )

        print(f"\nSplit '{split_name}':")
        print(f"  Total:   {total}")
        print(f"  Remove:  {removed} unique IDs ({code_summary})")
        print(f"  Keep:    {len(keep_indices)}")

        if not args.dry_run:
            filtered_dict[split_name] = ds.select(keep_indices)
        else:
            filtered_dict[split_name] = ds

    if args.split:
        for split_name in dataset:
            if split_name not in filtered_dict:
                filtered_dict[split_name] = dataset[split_name]

    if args.dry_run:
        print("\n[DRY RUN] No output written.")
        return

    result = DatasetDict(filtered_dict)
    print(f"\nSaving filtered dataset to: {output_path}")
    result.save_to_disk(str(output_path))

    print("\nDone.")
    for split_name in splits_to_process:
        if split_name in dataset and split_name in result:
            orig = len(dataset[split_name])
            kept = len(result[split_name])
            print(f"  {split_name}: {orig} -> {kept} ({orig - kept} removed)")


if __name__ == "__main__":
    main()
