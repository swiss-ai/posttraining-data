#!/usr/bin/env python3
"""Cross-check augmented samples between original and injected datasets.

Picks N random augmented indices from the dry-run/indices JSON, loads those
samples from both the original and augmented datasets, prints a side-by-side
comparison, and saves the full comparison to a JSON file.

Usage:
    python cross-check-injection.py injection_dry_run.json \
        --original /path/to/original \
        --augmented /path/to/augmented \
        --num-samples 5 \
        --output comparison.json \
        --seed 123
"""

import argparse
import json
from datasets import load_from_disk, DatasetDict


def load_dataset_train(path: str):
    """Load dataset and return the train split."""
    ds = load_from_disk(path)
    if isinstance(ds, DatasetDict):
        ds = ds["train"]
    return ds


def extract_developer(messages):
    """Extract the developer message from a messages list."""
    return next((m for m in messages if m["role"] == "developer"), None)


def has_tool_calls(messages) -> bool:
    """Check if any assistant message has non-empty tool calls."""
    for m in messages:
        if m["role"] != "assistant":
            continue
        for block in m["content"].get("blocks", []):
            if block["type"] == "tool_calls":
                for call in block.get("calls", []):
                    if call.get("name", ""):
                        return True
    return False


def compare_sample(idx, orig_sample, aug_sample):
    """Compare a single sample and return a comparison dict + print summary."""
    orig_dev = extract_developer(orig_sample["messages"])
    aug_dev = extract_developer(aug_sample["messages"])

    orig_tools_empty = not orig_dev["content"]["tools"] or not orig_dev["content"]["tools"].strip()
    aug_tools_empty = not aug_dev["content"]["tools"] or not aug_dev["content"]["tools"].strip()
    thinking_preserved = orig_dev["content"]["has_thinking"] == aug_dev["content"]["has_thinking"]

    orig_other = [m for m in orig_sample["messages"] if m["role"] != "developer"]
    aug_other = [m for m in aug_sample["messages"] if m["role"] != "developer"]
    msgs_identical = orig_other == aug_other

    aug_has_tool_calls = has_tool_calls(aug_sample["messages"])

    # Parse injected tools
    injected_tools = []
    if not aug_tools_empty:
        try:
            injected_tools = json.loads(aug_dev["content"]["tools"])
        except json.JSONDecodeError:
            injected_tools = []

    tool_names = [t.get("name", "?") for t in injected_tools]

    # First user message snippet
    first_user = next((m for m in orig_sample["messages"] if m["role"] == "user"), None)
    user_snippet = ""
    if first_user:
        parts = first_user["content"].get("parts", [])
        if parts:
            user_snippet = parts[0].get("text", "")[:200]

    # Print
    print(f"\n{'='*70}")
    print(f"Sample index: {idx}")
    print(f"{'='*70}")
    print(f"  Original tools empty:     {orig_tools_empty}")
    print(f"  Augmented tools empty:    {aug_tools_empty}")
    print(f"  has_thinking (orig):      {orig_dev['content']['has_thinking']}")
    print(f"  has_thinking (aug):       {aug_dev['content']['has_thinking']}")
    print(f"  has_thinking preserved:   {thinking_preserved}")
    print(f"  Injected tool count:      {len(injected_tools)}")
    if tool_names:
        print(f"  Tool names:               {tool_names[:5]}")
        if len(tool_names) > 5:
            print(f"                            ... and {len(tool_names) - 5} more")
    fmt = aug_dev["content"]["formatted_tools"]
    print(f"  formatted_tools present:  {bool(fmt and fmt.strip())}")
    print(f"  Non-dev messages count:   {len(orig_other)} (orig) / {len(aug_other)} (aug)")
    print(f"  Non-dev msgs identical:   {msgs_identical}")
    print(f"  Assistant has tool_calls: {aug_has_tool_calls}")
    if user_snippet:
        print(f"  User message snippet:     {user_snippet[:120]}...")

    # Build comparison record
    return {
        "index": idx,
        "original_tools_empty": orig_tools_empty,
        "augmented_tools_empty": aug_tools_empty,
        "has_thinking_original": orig_dev["content"]["has_thinking"],
        "has_thinking_augmented": aug_dev["content"]["has_thinking"],
        "has_thinking_preserved": thinking_preserved,
        "injected_tool_count": len(injected_tools),
        "injected_tool_names": tool_names,
        "formatted_tools_present": bool(fmt and fmt.strip()),
        "non_dev_messages_identical": msgs_identical,
        "assistant_has_tool_calls": aug_has_tool_calls,
        "user_message_snippet": user_snippet,
        "original_developer": orig_dev["content"],
        "augmented_developer": aug_dev["content"],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Cross-check augmented samples between original and injected datasets"
    )
    parser.add_argument(
        "indices_json",
        help="Path to the dry-run or save-indices JSON file",
    )
    parser.add_argument(
        "--original",
        required=True,
        help="Path to the original (pre-injection) dataset",
    )
    parser.add_argument(
        "--augmented",
        required=True,
        help="Path to the augmented (post-injection) dataset",
    )
    parser.add_argument(
        "--num-samples", "-n",
        type=int,
        default=3,
        help="Number of random augmented samples to compare (default: 3)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="injection_comparison.json",
        help="Output path for the comparison JSON (default: injection_comparison.json)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for sample selection (default: 123)",
    )
    args = parser.parse_args()

    # Load indices
    with open(args.indices_json) as f:
        indices_data = json.load(f)
    augmented_indices = indices_data["augmented_indices"]

    import random
    rng = random.Random(args.seed)
    n = min(args.num_samples, len(augmented_indices))
    selected = rng.sample(augmented_indices, n)
    selected.sort()

    print(f"Selected {n} augmented indices: {selected}")

    # Load datasets
    print(f"\nLoading original dataset: {args.original}")
    orig_ds = load_dataset_train(args.original)
    print(f"Loading augmented dataset: {args.augmented}")
    aug_ds = load_dataset_train(args.augmented)

    # Compare
    comparisons = []
    all_ok = True
    for idx in selected:
        record = compare_sample(idx, orig_ds[idx], aug_ds[idx])
        comparisons.append(record)
        if not record["original_tools_empty"]:
            print(f"  [WARN] Original already had tools at index {idx}")
            all_ok = False
        if record["augmented_tools_empty"]:
            print(f"  [WARN] Augmented has no tools at index {idx}")
            all_ok = False
        if not record["has_thinking_preserved"]:
            print(f"  [FAIL] has_thinking changed at index {idx}")
            all_ok = False
        if not record["non_dev_messages_identical"]:
            print(f"  [FAIL] Non-developer messages differ at index {idx}")
            all_ok = False
        if record["assistant_has_tool_calls"]:
            print(f"  [WARN] Assistant has tool calls at index {idx}")
            all_ok = False

    # Summary
    print(f"\n{'='*70}")
    if all_ok:
        print(f"All {n} samples passed cross-check.")
    else:
        print(f"Some samples had warnings or failures — see above.")

    # Save
    output = {
        "indices_json": args.indices_json,
        "original_path": args.original,
        "augmented_path": args.augmented,
        "seed": args.seed,
        "num_samples": n,
        "comparisons": comparisons,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Comparison saved to: {args.output}")


if __name__ == "__main__":
    main()
