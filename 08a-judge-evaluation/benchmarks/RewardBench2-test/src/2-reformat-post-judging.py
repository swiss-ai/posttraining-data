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

    python -m benchmarks.RewardBench2-test.src.2-reformat-post-judging
    python -m benchmarks.RewardBench2-test.src.2-reformat-post-judging --judge-name 01
"""

import os
from argparse import ArgumentParser
from collections import defaultdict

import datasets
from datasets import Dataset, load_from_disk
import random

datasets.disable_progress_bar()
from tqdm import tqdm

from src.utils import load_module


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--judge-name", type=str, default=None)
    args = parser.parse_args()
    
    benchmark_root = "benchmarks/RewardBench2-test/"
    judged_root = os.path.join(benchmark_root, "2-judged")

    input_paths = sorted([
        os.path.join(judged_root, x) 
        for x in os.listdir(judged_root) if x.isdigit()
    ])

    if args.judge_name:
        input_paths = [p for p in input_paths if os.path.basename(p) == args.judge_name]

    for input_path in input_paths:
        judge_name = os.path.basename(input_path)
        judge_args_path = f"judges/{judge_name}.py"
        if not os.path.exists(judge_args_path):
            print(f"Skipping {judge_args_path}: file not found")
            continue

        print(f"\n--- Processing {judge_name} ---")
        judge_args = load_module(judge_args_path)

        print(f"Loading dataset from {input_path}")
        input_ds = load_from_disk(input_path)

        output_ds = []
        row_ids = set("_".join(x.split("_")[:2]) for x in input_ds["prompt_id"])
        for row_id in tqdm(row_ids):
            subset_ds = input_ds.filter(lambda x: x["prompt_id"].startswith(row_id))

            chosen = []
            rejected = []
            scores_chosen = []
            scores_rejected = []
            noNone_scores_chosen = []
            noNone_scores_rejected = []
            for x in subset_ds:
                score = judge_args.get_score_from_distribution(x["score_distribution"])
                if score is not None:
                    comparator_score = score
                else:
                    comparator_score = float(random.sample(judge_args.scoring_range, 1)[0])
                if "chosen" in x["prompt_id"]:
                    chosen.append(x["response"])
                    scores_chosen.append(score)
                    noNone_scores_chosen.append(comparator_score)
                else:
                    rejected.append(x["response"])
                    scores_rejected.append(score)
                    noNone_scores_rejected.append(comparator_score)

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
                "noNone_scores_chosen": noNone_scores_chosen,
                "noNone_scores_rejected": noNone_scores_rejected,
                "scores": scores_chosen + scores_rejected,
                "noNone_scores": noNone_scores_chosen + noNone_scores_rejected,
            })

        output_ds = Dataset.from_list(output_ds)
        input_ds.cleanup_cache_files()

        output_path = f"benchmarks/RewardBench2-test/3-rereformatted/{judge_name}"
        print(f"Saving {len(output_ds)} rows to {output_path}")
        os.makedirs(output_path, exist_ok=True)
        output_ds.save_to_disk(output_path)
