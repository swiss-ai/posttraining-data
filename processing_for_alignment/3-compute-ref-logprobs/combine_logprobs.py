"""
Combine partitioned/checkpointed logprob results into a single dataset.

Adapted from swiss_alignment.data_alignment.merge_partitions_swissaiformat.

Handles the directory structure produced by compute_logprobs.py:
  partitions/
    <start>-<end>/
      subpart_0/
        checkpoint-2048/
        checkpoint-4096/
        ...
      subpart_1/
        checkpoint-2048/
        ...
    <start>-<end>/
      ...

Usage:
    python combine_logprobs.py \
        --partitions-dir /path/to/output/partitions \
        --output-dir /path/to/merged
"""

import argparse
import re
from pathlib import Path

from datasets import DatasetDict, concatenate_datasets, load_from_disk


def find_partition_dirs(partitions_dir):
    """Find and sort partition directories (named <start>-<end>) by start index."""
    pattern = re.compile(r"^(\d+)-(\d+)$")
    dirs = []
    for entry in Path(partitions_dir).iterdir():
        if entry.is_dir() and pattern.match(entry.name):
            start, end = map(int, entry.name.split("-"))
            dirs.append((start, end, entry))
    dirs.sort(key=lambda x: x[0])
    return dirs


def find_subpartition_dirs(partition_dir):
    """Find and sort subpartition directories (named subpart_N)."""
    pattern = re.compile(r"^subpart_(\d+)$")
    dirs = []
    for entry in Path(partition_dir).iterdir():
        if entry.is_dir() and pattern.match(entry.name):
            idx = int(entry.name.split("_")[1])
            dirs.append((idx, entry))
    dirs.sort(key=lambda x: x[0])
    return dirs


def find_checkpoint_dirs(subpart_dir):
    """Find and sort checkpoint directories (named checkpoint-N) by index."""
    pattern = re.compile(r"^checkpoint-(\d+)$")
    dirs = []
    for entry in Path(subpart_dir).iterdir():
        if entry.is_dir() and pattern.match(entry.name):
            idx = int(entry.name.split("-")[1])
            dirs.append((idx, entry))
    dirs.sort(key=lambda x: x[0])
    return dirs


def main(args):
    partitions_dir = Path(args.partitions_dir)

    partition_dirs = find_partition_dirs(partitions_dir)
    if not partition_dirs:
        print(f"No partition directories found in {partitions_dir}")
        return

    print(f"Found {len(partition_dirs)} partitions")

    all_datasets = []
    total_rows = 0

    for start, end, partition_path in partition_dirs:
        print(f"\nPartition [{start}, {end}):")
        subpart_dirs = find_subpartition_dirs(partition_path)

        if not subpart_dirs:
            print(f"  WARNING: No subpartition dirs found in {partition_path}")
            continue

        for subpart_idx, subpart_path in subpart_dirs:
            checkpoints = find_checkpoint_dirs(subpart_path)
            if not checkpoints:
                print(f"  WARNING: No checkpoints in {subpart_path}")
                continue

            for ckpt_idx, ckpt_path in checkpoints:
                ds = load_from_disk(str(ckpt_path))
                all_datasets.append(ds)
                total_rows += len(ds)
                print(
                    f"  subpart_{subpart_idx}/checkpoint-{ckpt_idx}: {len(ds)} rows"
                )

    if not all_datasets:
        print("No data found!")
        return

    print(f"\nConcatenating {len(all_datasets)} chunks ({total_rows} total rows)...")
    combined = concatenate_datasets(all_datasets)
    print(f"Combined dataset: {len(combined)} rows")
    print(f"Columns: {combined.column_names}")

    if "ref_chosen_logprob" in combined.column_names:
        skipped = sum(1 for x in combined["ref_chosen_logprob"] if x is None)
        if skipped > 0:
            print(f"Rows with None logprobs (exceeded max_seq_len): {skipped}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving to {output_dir}")
    DatasetDict({"train_split": combined}).save_to_disk(str(output_dir))
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combine partitioned logprob datasets"
    )
    parser.add_argument(
        "--partitions-dir",
        type=str,
        required=True,
        help="Directory containing <start>-<end>/subpart_N/checkpoint-M/ structure",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Path to save combined dataset",
    )
    args = parser.parse_args()
    main(args)
