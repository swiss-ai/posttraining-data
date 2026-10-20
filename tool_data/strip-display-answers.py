#!/usr/bin/env python3
"""
Strip undeclared display_answers tool calls from linearised dataset samples.

For samples flagged with tool_use_without_declarations (which are verifiable-
responses samples where the lineariser created display_answers tool_calls/
tool_outputs blocks without declaring the tool in the developer message),
this script removes those blocks from the assistant messages.

Non-matching samples are passed through unchanged.

Uses flagged IDs from validate-linearised.py to identify affected samples,
then verifies each one actually contains undeclared display_answers calls
before modifying.

Usage:
    python tool_data/strip-display-answers.py /path/to/linearised-dataset \
        --flagged-ids flagged-ids.json --output /path/to/output [--dry-run]
"""

import json
import argparse
import sys
from pathlib import Path
import datasets
from datasets import load_from_disk, DatasetDict

datasets.disable_caching()


def has_undeclared_display_answers(messages):
    """Check if sample has display_answers calls without declaration."""
    declared = False
    for msg in messages:
        if msg.get("role") != "developer":
            continue
        content = msg.get("content", {})
        if not isinstance(content, dict):
            continue
        tools_str = content.get("tools", "")
        if not tools_str or (isinstance(tools_str, str) and not tools_str.strip()):
            break
        try:
            tool_list = json.loads(tools_str) if isinstance(tools_str, str) else tools_str
            for t in tool_list:
                if isinstance(t, dict) and t.get("name") == "display_answers":
                    declared = True
                    break
        except (json.JSONDecodeError, TypeError):
            pass
        break

    if declared:
        return False

    for msg in messages:
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
                    return True
    return False


def strip_display_answers_blocks(messages):
    """Remove display_answers tool_calls blocks and their paired tool_outputs blocks."""
    new_messages = []
    for msg in messages:
        if msg.get("role") != "assistant":
            new_messages.append(msg)
            continue

        content = msg.get("content", {})
        if not isinstance(content, dict):
            new_messages.append(msg)
            continue

        blocks = content.get("blocks", [])
        if not isinstance(blocks, list):
            new_messages.append(msg)
            continue

        new_blocks = []
        skip_next_outputs = False

        for blk in blocks:
            if not isinstance(blk, dict):
                new_blocks.append(blk)
                skip_next_outputs = False
                continue

            btype = blk.get("type")

            if skip_next_outputs and btype == "tool_outputs":
                outputs = [o for o in blk.get("outputs", []) if isinstance(o, dict)]
                is_da_output = outputs and all(
                    o.get("name") == "display_answers" for o in outputs
                )
                if is_da_output:
                    skip_next_outputs = False
                    continue
                else:
                    skip_next_outputs = False
                    new_blocks.append(blk)
                    continue

            skip_next_outputs = False

            if btype == "tool_calls":
                calls = blk.get("calls", [])
                real_calls = [c for c in calls if isinstance(c, dict) and c.get("name")]
                if real_calls and all(c["name"] == "display_answers" for c in real_calls):
                    skip_next_outputs = True
                    continue
                else:
                    new_blocks.append(blk)
            else:
                new_blocks.append(blk)

        new_content = dict(content)
        new_content["blocks"] = new_blocks
        new_msg = dict(msg)
        new_msg["content"] = new_content
        new_messages.append(new_msg)

    return new_messages


def main():
    parser = argparse.ArgumentParser(
        description="Strip undeclared display_answers tool calls from linearised samples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input_path", help="Path to linearised dataset")
    parser.add_argument("--flagged-ids", required=True,
                        help="Path to flagged IDs JSON from validate-linearised.py")
    parser.add_argument("--output", required=True,
                        help="Output path for cleaned dataset")
    parser.add_argument("--split", default=None,
                        help="Process a specific split (default: all)")
    parser.add_argument("--num-proc", type=int, default=16,
                        help="Number of parallel workers (default: 16)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would be modified without writing")
    args = parser.parse_args()

    input_path = Path(args.input_path)
    flagged_path = Path(args.flagged_ids)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: input path does not exist: {input_path}", file=sys.stderr)
        sys.exit(1)
    if not flagged_path.exists():
        print(f"Error: flagged-ids path does not exist: {flagged_path}", file=sys.stderr)
        sys.exit(1)

    with open(flagged_path) as f:
        flagged = json.load(f)

    print(f"Loading dataset from: {input_path}")
    dataset = load_from_disk(str(input_path))
    if not isinstance(dataset, DatasetDict):
        dataset = DatasetDict({"train": dataset})

    splits_to_process = [args.split] if args.split else list(dataset.keys())

    result_dict = {}
    total_modified = 0
    total_skipped = 0

    for split_name in splits_to_process:
        if split_name not in dataset:
            print(f"Error: split '{split_name}' not found. Available: {list(dataset.keys())}",
                  file=sys.stderr)
            sys.exit(1)

        ds = dataset[split_name]

        raw_ids = flagged.get(split_name, {}).get("tool_use_without_declarations", [])
        target_ids = set(raw_ids)
        print(f"\nSplit '{split_name}': {len(ds):,} samples, "
              f"{len(target_ids):,} flagged for tool_use_without_declarations")

        if not target_ids:
            print("  No flagged IDs — passing through unchanged")
            result_dict[split_name] = ds
            continue

        def process_sample(sample):
            cid = sample["conversation_id"]
            if cid not in target_ids:
                return {**sample, "_modified": False}

            if not has_undeclared_display_answers(sample["messages"]):
                return {**sample, "_modified": False}

            new_messages = strip_display_answers_blocks(sample["messages"])
            return {**sample, "messages": new_messages, "_modified": True}

        if args.dry_run:
            print("  [DRY RUN] Scanning to count affected samples...")
        else:
            print(f"  Processing samples (num_proc={args.num_proc})...")

        ds_result = ds.map(
            process_sample, num_proc=args.num_proc,
            desc=f"{'Dry-run scan' if args.dry_run else 'Stripping display_answers'} ({split_name})",
        )

        modified = sum(ds_result["_modified"])
        flagged_in_ds = sum(1 for cid in ds["conversation_id"] if cid in target_ids)
        skipped = flagged_in_ds - modified

        if args.dry_run:
            print(f"  Flagged IDs in dataset:                  {flagged_in_ds:,}")
            print(f"  Would modify (verified display_answers): {modified:,}")
            print(f"  Would skip (no undeclared display_answers): {skipped:,}")
            result_dict[split_name] = ds
        else:
            result_dict[split_name] = ds_result.remove_columns("_modified")
            print(f"  Modified: {modified:,}")
            print(f"  Skipped (no undeclared display_answers): {skipped:,}")

        total_modified += modified
        total_skipped += skipped

    if args.split:
        for split_name in dataset:
            if split_name not in result_dict:
                result_dict[split_name] = dataset[split_name]

    if args.dry_run:
        print(f"\n[DRY RUN] Would modify {total_modified:,} samples, "
              f"skip {total_skipped:,}. No output written.")
        return

    result = DatasetDict(result_dict)
    print(f"\nSaving cleaned dataset to: {output_path}")
    result.save_to_disk(str(output_path))

    print("\nDone.")
    print(f"  Total modified: {total_modified:,}")
    print(f"  Total skipped:  {total_skipped:,}")
    for split_name in splits_to_process:
        if split_name in dataset:
            print(f"  {split_name}: {len(dataset[split_name]):,} samples "
                  f"-> {len(result_dict[split_name]):,} samples (same count, blocks stripped)")


if __name__ == "__main__":
    main()
