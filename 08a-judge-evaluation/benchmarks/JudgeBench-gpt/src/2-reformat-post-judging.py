"""
Reformats the output of the judge into paired rows for JudgeBench evaluation metrics.

The input dataset:
- has these columns: 'original_id', 'source', 'response_model', 'label', 'response', 'prompt_id', 'prompt', 'score_distribution'
- has one row per response, not per pair

The output dataset:
- has these columns: 'pair_id', 'original_id', 'source', 'question', 'response_model', 'response_A', 'response_B', 'score_A', 'score_B', 'label', 'judgments'
- has two responses per row

Usage:

    python -m benchmarks.JudgeBench-gpt.src.2-reformat-post-judging
    python -m benchmarks.JudgeBench-gpt.src.2-reformat-post-judging --judge-args-path judges/01.py
"""

import os
from argparse import ArgumentParser
import glob
import random
from typing import List, Optional

from datasets import Dataset, load_from_disk

from src.utils import load_module

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--judge-name", type=str, default=None)
    args = parser.parse_args()

    random.seed(1)

    judged_root = "benchmarks/JudgeBench-gpt/2-judged"
    if args.judge_name:
        judge_names = [args.judge_name]
    else:
        judge_names = sorted([x for x in os.listdir(judged_root) if x.isdigit()])
    for judge_name in judge_names:
        judge_args_path = f"judges/{judge_name}.py"
        if not os.path.exists(judge_args_path):
            print(f"Skipping {judge_args_path}: file not found")
            continue

        print(f"\n--- Processing {judge_args_path} ---")
        judge_args = load_module(judge_args_path)

        input_dir = f"benchmarks/JudgeBench-gpt/2-judged/{judge_name}"
        output_dir = f"benchmarks/JudgeBench-gpt/3-rereformatted/{judge_name}"

        print(f"Loading dataset from {input_dir}")
        input_ds = load_from_disk(input_dir)

        print("Reformatting dataset...")
        input_ds_A = input_ds.filter(lambda x: x["prompt_id"][-1] == "A")
        input_ds_B = input_ds.filter(lambda x: x["prompt_id"][-1] == "B")

        output_ds = []
        for x_A in input_ds_A:
            prompt_id_B = f"{x_A['prompt_id'][:-1]}B"
            temp = input_ds_B.filter(
                lambda x: x["source"] == x_A["source"] and x["prompt_id"] == prompt_id_B
            )
            if len(temp) != 1:
                print(f"Something went wrong with {x_A['prompt_id']}")
                print(temp)
            x_B = temp[0]

            # Compute judgments
            # compute_final_metrics expects two judgments per pair, one for each possible ordering of the response pair
            # here we are just artificially creating the second judgement because all our judges are independent of ordering.
            # this is so we can compare against the reported scores
            score_A = judge_args.get_score_from_distribution(x_A["score_distribution"])
            score_B = judge_args.get_score_from_distribution(x_B["score_distribution"])

            # determine scores to use for comparison (replace Nones by random score)
            if score_A is not None:
                comparator_score_A = score_A
            else:
                comparator_score_A = float(random.sample(judge_args.scoring_range, 1)[0])
            if score_B is not None:
                comparator_score_B = score_B
            else:
                comparator_score_B = float(random.sample(judge_args.scoring_range, 1)[0])

            # based on comparator scores, potentially flip judgements. resolve ties randomly
            judgments = [{"decision": "A>B"}, {"decision": "B>A"}]
            if comparator_score_A > comparator_score_B:
                pass
            elif comparator_score_A < comparator_score_B or random.random() < 0.5:
                judgments = judgments[::-1]

            output_ds.append({
                "pair_id": x_A["prompt_id"][:-2],
                "original_id": x_A["original_id"],
                "source": x_A["source"],
                "question": x_A["prompt"],
                "response_model": x_A["response_model"],
                "response_A": x_A["response"],
                "response_B": x_B["response"],
                "score_A": score_A,
                "score_B": score_B,
                "label": x_A["label"],
                "judgments": judgments,
            })
        output_ds = Dataset.from_list(output_ds)
        input_ds.cleanup_cache_files()

        print(f"Saving dataset to {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        output_ds.save_to_disk(output_dir)
