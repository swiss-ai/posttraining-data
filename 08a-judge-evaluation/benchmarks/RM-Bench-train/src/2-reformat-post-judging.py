"""
Reformats the output of the judge into paired rows for RM-Bench evaluation metrics. 

[TODO]

Usage:

python -m benchmarks.RM-Bench-train.src.2-reformat-post-judging \
    --input-path benchmarks/RM-Bench-train/2-judged/train/?? \
    --output-path benchmarks/RM-Bench-train/3-rereformatted/train \
    --judge-args-path judges/01.py
"""

import os
from argparse import ArgumentParser

from datasets import Dataset, load_from_disk

from src.utils import load_module

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--judge-args-path", type=str, required=True)
    args = parser.parse_args()

    # Load judge args
    judge_args = load_module(args.judge_args_path)

    print(f"Loading dataset from {args.input_path}")
    input_ds = load_from_disk(args.input_path)

    print("Reformatting dataset...")
    output_ds = []
    ...
    output_ds = Dataset.from_list(output_ds)
    input_ds.cleanup_cache_files()

    print(f"Saving dataset to {args.output_path}")
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    output_ds.save_to_disk(args.output_path)
