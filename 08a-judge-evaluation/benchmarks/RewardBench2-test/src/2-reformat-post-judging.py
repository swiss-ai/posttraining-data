"""
Reformats judge output into per-prompt rows for RewardBench-2 evaluation metrics.

Input dataset (2-judged/{judge_name}):
  - One row per response, columns: prompt, response, prompt_id, subset, rb2_id,
    num_correct, score_distribution
  - prompt_id format: "{subset}_{rb2_id}_{chosen|rejected}_{i}"

Output dataset (3-rereformatted/{judge_name}):
  - One row per original RB2 prompt
  - Columns: rb2_id, subset, num_correct, scores (ordered list), result
    - scores: [chosen_0, ..., chosen_k, rejected_0, ..., rejected_m]
              (None where the judge failed)
    - result: for non-Ties: 1/k if chosen_0 is among k tied max-scorers, else 0.0
              for Ties: None (computed later by process_single_model in step 3)

Usage (from the 08a-judge-evaluation repo root):

python -m benchmarks.RewardBench2-test.src.2-reformat-post-judging \
    --judge-args-path judges/01.py
"""

import os
from argparse import ArgumentParser
from collections import defaultdict

import datasets
from datasets import Dataset, load_from_disk

datasets.disable_progress_bar()
from tqdm import tqdm

from src.utils import load_module


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--judge-args-path", type=str, required=True)
    args = parser.parse_args()

    judge_args = load_module(args.judge_args_path)
    judge_name = os.path.basename(args.judge_args_path).rstrip(".py")
    input_dir = f"benchmarks/RewardBench2-test/2-judged/{judge_name}"
    output_dir = f"benchmarks/RewardBench2-test/3-rereformatted/{judge_name}"

    print(f"Loading dataset from {input_dir}")
    input_ds = load_from_disk(input_dir)

    output_ds = []
    row_ids = set("_".join(x.split("_")[:2]) for x in input_ds["prompt_id"])
    for row_id in tqdm(row_ids):
        subset_ds = input_ds.filter(lambda x: x["prompt_id"].startswith(row_id))

        chosen = []
        rejected = []
        scores_chosen = []
        scores_rejected = []
        for x in subset_ds:
            score = judge_args.get_score_from_distribution(x["score_distribution"])
            if "chosen" in x["prompt_id"]:
                chosen.append(x["response"])
                scores_chosen.append(score)
            else:
                rejected.append(x["response"])
                scores_rejected.append(score)

        output_ds.append({
            "id": x["rb2_id"],
            "prompt": x["prompt"],
            "chosen": chosen,
            "rejected": rejected,
            "num_correct": len(chosen),
            "num_incorrect": len(rejected),
            "total_completions": len(chosen) + len(rejected),
            "subset": x["subset"],
            "scores_chosen": scores_chosen,
            "scores_rejected": scores_rejected,
            "scores": scores_chosen + scores_rejected,
        })

    output_ds = Dataset.from_list(output_ds)
    input_ds.cleanup_cache_files()

    print(f"Saving {len(output_ds)} rows to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    output_ds.save_to_disk(output_dir)
