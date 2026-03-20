"""
Split a dataset with reference_completions into N separate datasets,
one per completion index. Each gets a 'response' column so the original
response_annotation/annotate.py can process it directly.

Usage:
    python split_by_completion.py \
        --dataset-path /path/to/dataset_with_ref_completions \
        --output-dir /path/to/output/completions_split
"""

import argparse
import os
from datasets import load_from_disk
from tqdm import tqdm


def main(args):
    dataset = load_from_disk(args.dataset_path)
    if hasattr(dataset, "keys"):
        split = "train_split" if "train_split" in dataset else list(dataset.keys())[0]
        print(f"DatasetDict detected, using '{split}' split")
        dataset = dataset[split]

    n_completions = args.n_completions
    n_rows = len(dataset)
    print(f"Dataset size: {n_rows}, completions per row: {n_completions}")

    num_proc = args.num_cpus

    for i in tqdm(range(n_completions), desc="Splitting completions", unit="split"):
        split_dataset = dataset.map(
            lambda row: {"response": row["reference_completions"][i]},
            num_proc=num_proc,
            desc=f"Extracting completion {i}",
        )
        output_path = os.path.join(args.output_dir, f"completion_{i}")
        os.makedirs(output_path, exist_ok=True)
        split_dataset.save_to_disk(output_path, num_proc=num_proc)

    print(f"Done. {n_completions} datasets written to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--n-completions", type=int, default=30)
    parser.add_argument("--num-cpus", type=int, default=288)
    args = parser.parse_args()
    main(args)
