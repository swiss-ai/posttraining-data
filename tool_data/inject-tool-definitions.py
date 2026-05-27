#!/usr/bin/env python3
"""
Inject tool definitions into non-tool-calling SFT samples.

Creates "tools-aware but not tool-calling" training samples by taking tool
definitions (tools + formatted_tools) from tool-calling source datasets and
injecting them into the developer message of target samples that have no tools
and no actual tool usage in their assistant messages.

This teaches the model when NOT to call tools.

Algorithm:
  1. Extract tool definitions from source datasets (e.g. Toucan, EnvScaler,
     OpenSeeker) -- collects (tools, formatted_tools) pairs from developer
     messages into a pool.
  2. Sample num_samples tool sets from the pool without replacement.
  3. Find eligible targets in the main SFT mix -- samples where the developer
     message has empty tools AND assistant messages contain no tool blocks
     (excludes display_answers / verifiable-responses samples).
  4. Randomly select num_samples eligible indices, pair each with a sampled
     tool set.
  5. Augment via dataset.map(): for selected samples, inject tools and
     formatted_tools into the developer message, preserving has_thinking and
     all other messages untouched.
  6. Save the full dataset (augmented + unchanged) as an HF DatasetDict with
     updated dataset_metadata.json. Optionally (--save-indices) persist
     augmented indices, eligible candidates, and index-to-injection-order
     mapping to injection_indices.json in the output directory.
  7. Validate by reloading from disk and verifying: total count unchanged,
     augmented samples have correct tool definitions, non-augmented samples
     are identical to input.

Use --dry-run to execute only the preparation steps (extract tool sets,
sample, find eligible candidates, select indices) and export the indices
JSON without augmenting or saving any dataset. Useful for inspecting
which samples would be changed before committing to a full run.
"""

import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set

import datasets
from datasets import Dataset, DatasetDict, load_from_disk

# Disable caching so map() doesn't try to write cache files next to
# potentially read-only source datasets.
datasets.disable_caching()


def load_existing_metadata(dataset_path: Path) -> Dict[str, Any]:
    """Load existing metadata from dataset directory."""
    metadata_file = dataset_path / "dataset_metadata.json"
    if metadata_file.exists():
        with open(metadata_file, "r") as f:
            return json.load(f)
    return {}


def extract_tool_sets(source_path: str) -> List[Dict[str, str]]:
    """
    Extract (tools, formatted_tools) pairs from a linearised tool-calling dataset.

    Returns a list of dicts with 'tools' and 'formatted_tools' keys,
    one per sample that has non-empty tools.
    """
    print(f"Loading tool source: {source_path}")
    ds = load_from_disk(source_path, keep_in_memory=True)
    if isinstance(ds, DatasetDict):
        ds = ds["train"]

    tool_sets = []
    for sample in ds:
        for msg in sample["messages"]:
            if msg["role"] == "developer":
                tools = msg["content"]["tools"]
                fmt = msg["content"]["formatted_tools"]
                if tools and tools.strip():
                    tool_sets.append({"tools": tools, "formatted_tools": fmt})
                break
    print(f"  Extracted {len(tool_sets)} tool sets from {Path(source_path).name}")
    return tool_sets


def _has_tool_usage(messages: List[Dict[str, Any]]) -> bool:
    """
    Check if any assistant message contains actual tool_calls or tool_outputs
    with non-empty content (name or output).

    This catches display_answers and other tool blocks that should not be
    mixed with injected tool definitions.
    """
    for msg in messages:
        if msg["role"] != "assistant":
            continue
        blocks = msg["content"].get("blocks", [])
        for block in blocks:
            if block["type"] == "tool_calls":
                for call in block.get("calls", []):
                    if call.get("name", ""):
                        return True
            if block["type"] == "tool_outputs":
                for out in block.get("outputs", []):
                    if out.get("output", ""):
                        return True
    return False


def _has_empty_tools(messages: List[Dict[str, Any]]) -> bool:
    """Check if the developer message has empty tools."""
    for msg in messages:
        if msg["role"] == "developer":
            tools = msg["content"].get("tools", "")
            return not tools or not tools.strip()
    return True


def find_eligible_indices(dataset: Dataset) -> List[int]:
    """
    Find indices of samples eligible for tool injection using parallel map.

    A sample is eligible if:
    1. Its developer message has empty tools
    2. Its assistant messages contain no actual tool blocks
    """

    def check_eligibility(sample, idx):
        messages = sample["messages"]
        eligible = _has_empty_tools(messages) and not _has_tool_usage(messages)
        return {"_eligible": eligible}

    tagged = dataset.map(
        check_eligibility,
        with_indices=True,
        num_proc=16,
        desc="Finding eligible samples",
    )

    eligible = [i for i, e in enumerate(tagged["_eligible"]) if e]
    return eligible


def validate_output(
    input_ds: Dataset,
    output_ds: Dataset,
    augmented_indices: Set[int],
    index_to_toolset: Dict[int, Dict[str, str]],
) -> None:
    """
    Validate the augmented dataset against the original input.

    Uses parallel map to verify:
    - Augmented samples have correct tools/formatted_tools and unchanged has_thinking
    - Non-augmented samples are identical to input
    - Total sample count is preserved

    Prints statistics and raises RuntimeError on validation failures.
    """
    assert len(output_ds) == len(input_ds), (
        f"Total sample count changed: {len(input_ds)} -> {len(output_ds)}"
    )
    print(f"  [OK] Total sample count preserved: {len(output_ds)}")

    # Serialise expected values for augmented samples into a lookup keyed by index.
    # Each entry: (expected_tools, expected_formatted_tools)
    expected_lookup = {
        idx: (ts["tools"], ts["formatted_tools"])
        for idx, ts in index_to_toolset.items()
    }

    def validate_sample(sample, idx):
        messages = sample["messages"]
        dev = next(m for m in messages if m["role"] == "developer")
        tools = dev["content"].get("tools", "")
        has_tools = bool(tools and tools.strip())

        result = {
            "_has_tools": has_tools,
            "_is_augmented": idx in augmented_indices,
            "_error": "",
        }

        if idx in expected_lookup:
            exp_tools, exp_fmt = expected_lookup[idx]
            if dev["content"]["tools"] != exp_tools:
                result["_error"] = f"idx {idx}: tools mismatch"
            elif dev["content"]["formatted_tools"] != exp_fmt:
                result["_error"] = f"idx {idx}: formatted_tools mismatch"
        return result

    out_tagged = output_ds.map(
        validate_sample,
        with_indices=True,
        num_proc=16,
        desc="Validating output samples",
    )

    # Also check has_thinking and unchanged samples against input
    def validate_against_input(sample, idx):
        in_msgs = sample["messages"]
        in_dev = next(m for m in in_msgs if m["role"] == "developer")
        return {
            "_in_has_tools": bool(
                in_dev["content"].get("tools", "")
                and in_dev["content"]["tools"].strip()
            ),
            "_in_has_thinking": in_dev["content"]["has_thinking"],
            "_in_tools": in_dev["content"].get("tools", ""),
            "_in_formatted_tools": in_dev["content"].get("formatted_tools", ""),
        }

    in_tagged = input_ds.map(
        validate_against_input,
        with_indices=True,
        num_proc=16,
        desc="Reading input for comparison",
    )

    # Collect statistics and errors
    errors = []
    original_with_tools = sum(in_tagged["_in_has_tools"])
    output_with_tools = sum(out_tagged["_has_tools"])
    augmented_verified = sum(out_tagged["_is_augmented"])
    unchanged_verified = len(output_ds) - augmented_verified

    # Collect map-phase errors
    for err in out_tagged["_error"]:
        if err:
            errors.append(err)

    # Cross-check has_thinking and unchanged developer fields.
    # Read columns once as lists to avoid per-sample random access.
    out_messages_col = output_ds["messages"]
    in_has_thinking = in_tagged["_in_has_thinking"]
    in_tools_col = in_tagged["_in_tools"]
    in_fmt_col = in_tagged["_in_formatted_tools"]

    for idx in augmented_indices:
        out_dev = next(
            m for m in out_messages_col[idx] if m["role"] == "developer"
        )
        if out_dev["content"]["has_thinking"] != in_has_thinking[idx]:
            errors.append(f"idx {idx}: has_thinking changed")

    # Spot-check a sample of non-augmented indices for unchanged fields
    non_augmented_count = len(output_ds) - len(augmented_indices)
    check_limit = 0
    if non_augmented_count > 0:
        check_limit = min(non_augmented_count, 50_000)
        non_aug_indices = [i for i in range(len(output_ds)) if i not in augmented_indices]
        rng = random.Random(0)
        check_indices = rng.sample(non_aug_indices, check_limit) if len(non_aug_indices) > check_limit else non_aug_indices

        for idx in check_indices:
            out_dev = next(
                m for m in out_messages_col[idx] if m["role"] == "developer"
            )
            if out_dev["content"].get("tools", "") != in_tools_col[idx]:
                errors.append(f"idx {idx}: non-augmented sample tools changed")
            if out_dev["content"].get("formatted_tools", "") != in_fmt_col[idx]:
                errors.append(f"idx {idx}: non-augmented sample formatted_tools changed")
            if out_dev["content"]["has_thinking"] != in_has_thinking[idx]:
                errors.append(f"idx {idx}: non-augmented sample has_thinking changed")

    newly_augmented = output_with_tools - original_with_tools

    if errors:
        print(f"\n  [FAIL] {len(errors)} validation errors:")
        for err in errors[:20]:
            print(f"    - {err}")
        if len(errors) > 20:
            print(f"    ... and {len(errors) - 20} more")
    else:
        print(f"  [OK] All samples validated successfully")

    print(f"\n  --- Statistics ---")
    print(f"  Input total samples:            {len(input_ds)}")
    print(f"  Output total samples:           {len(output_ds)}")
    print(f"  Samples with tools (input):     {original_with_tools}")
    print(f"  Samples with tools (output):    {output_with_tools}")
    print(f"  Newly augmented:                {newly_augmented}")
    print(f"  Augmented samples verified:     {augmented_verified}")
    print(f"  Unchanged samples checked:      {check_limit if non_augmented_count > 0 else 0}")
    print(f"  Unchanged samples total:        {unchanged_verified}")

    if errors:
        raise RuntimeError(f"Validation failed with {len(errors)} errors")


def build_indices_data(
    seed: int,
    num_samples: int,
    eligible_indices: List[int],
    total_samples: int,
    pool_size: int,
    tool_sources: List[str],
    selected_indices_list: List[int],
) -> Dict[str, Any]:
    """Build the indices payload for --save-indices and --dry-run export.

    Maps each augmented target index to its injection order: position i
    in selected_indices_list was paired with sampled_tool_sets[i].
    To reconstruct which tool set was injected, replay the sampling
    with the same seed: rng.sample(all_tool_sets, num_samples)[pos].
    JSON requires string keys, so indices are stringified.
    """
    index_to_order = {
        str(idx): pos
        for pos, idx in enumerate(selected_indices_list)
    }
    return {
        "seed": seed,
        "samples_augmented": num_samples,
        "eligible_candidates": len(eligible_indices),
        "total_samples": total_samples,
        "tool_pool_size": pool_size,
        "tool_sources": tool_sources,
        "augmented_indices": sorted(selected_indices_list),
        "eligible_indices": sorted(eligible_indices),
        "index_to_injection_order": index_to_order,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inject tool definitions into non-tool-calling SFT samples"
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to the target linearised dataset",
    )
    parser.add_argument(
        "--tool-sources",
        nargs="+",
        required=True,
        help="Paths to linearised tool-calling datasets to extract tool definitions from",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of non-tool samples to augment (default: total source samples)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for the augmented dataset (required unless --dry-run)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--save-indices",
        action="store_true",
        help="Save augmented indices, eligible candidates, and index-to-toolset "
        "mapping to a JSON file in the output directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run preparation steps (load sources, find eligible candidates, "
        "select indices) and export the indices JSON without augmenting or "
        "saving the dataset. Use --dry-run-output to set the output path.",
    )
    parser.add_argument(
        "--dry-run-output",
        type=str,
        default=None,
        help="Path for the dry-run indices JSON file "
        "(default: ./injection_dry_run.json)",
    )

    args = parser.parse_args()

    if not args.dry_run and args.output is None:
        parser.error("--output is required unless --dry-run is specified")
    if args.dry_run_output and not args.dry_run:
        parser.error("--dry-run-output requires --dry-run")
    rng = random.Random(args.seed)

    # Step 1: Load source tool sets
    print("=" * 60)
    print("Step 1: Loading tool sets from source datasets")
    print("=" * 60)
    all_tool_sets = []
    for source_path in args.tool_sources:
        tool_sets = extract_tool_sets(source_path)
        all_tool_sets.extend(tool_sets)

    pool_size = len(all_tool_sets)
    print(f"\nTotal tool set pool: {pool_size}")

    # Step 2: Determine num_samples and sample tool sets
    num_samples = args.num_samples if args.num_samples is not None else pool_size
    print(f"Requested augmentation count: {num_samples}")

    if num_samples > pool_size:
        raise ValueError(
            f"--num-samples ({num_samples}) exceeds tool set pool size ({pool_size}). "
            f"Sampling without replacement is not possible."
        )

    print("\n" + "=" * 60)
    print("Step 2: Sampling tool sets")
    print("=" * 60)
    sampled_tool_sets = rng.sample(all_tool_sets, num_samples)
    print(f"Sampled {len(sampled_tool_sets)} tool sets without replacement")

    # Step 3: Load target dataset
    print("\n" + "=" * 60)
    print("Step 3: Loading target dataset")
    print("=" * 60)
    print(f"Loading from: {args.input_path}")
    target_ds = load_from_disk(args.input_path)
    if isinstance(target_ds, DatasetDict):
        dataset = target_ds["train"]
    else:
        dataset = target_ds
    print(f"Target dataset size: {len(dataset)}")

    # Step 4: Find eligible candidate samples
    print("\n" + "=" * 60)
    print("Step 4: Finding eligible samples")
    print("=" * 60)
    eligible_indices = find_eligible_indices(dataset)
    print(f"Eligible candidates: {len(eligible_indices)} / {len(dataset)}")

    if num_samples > len(eligible_indices):
        raise ValueError(
            f"--num-samples ({num_samples}) exceeds eligible candidates "
            f"({len(eligible_indices)}). Not enough samples to augment."
        )

    # Step 5: Select target indices
    print("\n" + "=" * 60)
    print("Step 5: Selecting target indices")
    print("=" * 60)
    selected_indices_list = rng.sample(eligible_indices, num_samples)
    selected_indices = set(selected_indices_list)
    print(f"Selected {len(selected_indices)} indices for augmentation")

    # Build lookup: index -> tool set (use list to preserve deterministic pairing)
    index_to_toolset = dict(zip(selected_indices_list, sampled_tool_sets))

    # Dry run: export indices and exit without augmenting
    if args.dry_run:
        dry_run_path = Path(args.dry_run_output or "injection_dry_run.json")
        indices_data = build_indices_data(
            seed=args.seed,
            num_samples=num_samples,
            eligible_indices=eligible_indices,
            total_samples=len(dataset),
            pool_size=pool_size,
            tool_sources=args.tool_sources,
            selected_indices_list=selected_indices_list,
        )
        dry_run_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dry_run_path, "w") as f:
            json.dump(indices_data, f, indent=2)

        print(f"\n{'=' * 60}")
        print("Dry run complete")
        print(f"{'=' * 60}")
        print(f"  Eligible candidates:            {len(eligible_indices)} / {len(dataset)}")
        print(f"  Selected for augment:           {num_samples}")
        print(f"  Tool set pool size:             {pool_size}")
        print(f"  Tool sources:                   {len(args.tool_sources)}")
        for src in args.tool_sources:
            print(f"    - {src}")
        print(f"  Indices exported to:            {dry_run_path}")
        print(f"\nNo dataset was modified or saved.")
        return

    # Step 6: Apply augmentation via map
    print("\n" + "=" * 60)
    print("Step 6: Applying augmentation")
    print("=" * 60)

    def augment_sample(sample, idx):
        if idx not in index_to_toolset:
            return sample

        tool_set = index_to_toolset[idx]
        new_messages = []
        for msg in sample["messages"]:
            if msg["role"] == "developer":
                new_content = {
                    "tools": tool_set["tools"],
                    "has_thinking": msg["content"]["has_thinking"],
                    "formatted_tools": tool_set["formatted_tools"],
                }
                new_messages.append({"role": "developer", "content": new_content})
            else:
                new_messages.append(msg)

        return {**sample, "messages": new_messages}

    augmented_dataset = dataset.map(
        augment_sample,
        with_indices=True,
        desc="Augmenting samples with tool definitions",
        num_proc=16,
    )

    print(f"Augmentation complete: {len(augmented_dataset)} samples")

    # Step 7: Save
    print("\n" + "=" * 60)
    print("Step 7: Saving augmented dataset")
    print("=" * 60)
    output_path = Path(args.output)
    output_ds = DatasetDict({"train": augmented_dataset})

    # Load existing metadata from input
    original_metadata = load_existing_metadata(Path(args.input_path))

    processing_entry = {
        "operation": "inject_tool_definitions",
        "script": "inject-tool-definitions.py",
        "timestamp": datetime.now().isoformat(),
        "input_path": args.input_path,
        "output_path": str(output_path),
        "samples_processed": len(augmented_dataset),
        "samples_augmented": num_samples,
        "tool_sources": args.tool_sources,
        "tool_pool_size": pool_size,
        "eligible_candidates": len(eligible_indices),
        "seed": args.seed,
    }

    metadata = {
        **original_metadata,
        "processing_log": original_metadata.get("processing_log", [])
        + [processing_entry],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_ds.save_to_disk(str(output_path))

    with open(output_path / "dataset_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Dataset saved to: {output_path}")
    print(f"Metadata updated with processing log entry")

    # Save indices if requested
    if args.save_indices:
        indices_path = output_path / "injection_indices.json"
        indices_data = build_indices_data(
            seed=args.seed,
            num_samples=num_samples,
            eligible_indices=eligible_indices,
            total_samples=len(dataset),
            pool_size=pool_size,
            tool_sources=args.tool_sources,
            selected_indices_list=selected_indices_list,
        )
        with open(indices_path, "w") as f:
            json.dump(indices_data, f, indent=2)
        print(f"Indices saved to: {indices_path}")

    # Step 8: Validation (round-trip: reload from disk to verify serialisation)
    print("\n" + "=" * 60)
    print("Step 8: Validation")
    print("=" * 60)

    print("Loading output dataset from disk for round-trip validation...")
    reloaded_ds = load_from_disk(str(output_path))
    if isinstance(reloaded_ds, DatasetDict):
        reloaded_ds = reloaded_ds["train"]

    validate_output(
        input_ds=dataset,
        output_ds=reloaded_ds,
        augmented_indices=selected_indices,
        index_to_toolset=index_to_toolset,
    )

    # Final summary
    print(f"\n  Eligible candidates:            {len(eligible_indices)}")
    print(f"  Tool set pool size:             {pool_size}")
    print(f"  Tool sources:                   {len(args.tool_sources)}")
    for src in args.tool_sources:
        print(f"    - {src}")
    print(f"  Output path:                    {output_path}")


if __name__ == "__main__":
    main()
