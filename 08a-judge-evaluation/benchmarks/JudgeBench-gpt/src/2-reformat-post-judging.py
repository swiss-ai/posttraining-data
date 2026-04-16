"""
Reformats the output of the judge into paired rows for JudgeBench evaluation metrics. 

The input dataset:
- has these columns: 'original_id', 'source', 'response_model', 'label', 'response', 'prompt_id', 'prompt', 'score_distribution'
- has one row per response, not per pair

The output dataset:
- has these columns: 'pair_id', 'original_id', 'source', 'question', 'response_model', 'response_A', 'response_B', 'score_A', 'score_B', 'label', 'judgments'
- has two responses per row

Usage:

python -m benchmarks.JudgeBench-gpt.src.2-reformat-post-judging \
    --judge-args-path judges/01.py
"""

import os
from argparse import ArgumentParser
import random

from datasets import Dataset, load_from_disk

from src.utils import load_module

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--judge-args-path", type=str, required=True)
    args = parser.parse_args()

    # Load judge args
    random.seed(1)
    judge_args = load_module(args.judge_args_path)

    # Determine input/output dirs based on judge name
    judge_name = os.path.basename(args.judge_args_path).rstrip(".py")
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
        judgments = [{"decision": "A>B"}, {"decision": "B>A"}]

        # compare responses directly based on scores, resolve ties randomly
        if score_A is not None and score_B is not None:
            if score_A > score_B:
                pass
            elif score_A < score_B:
                judgments = judgments[::-1]
            else:
                if random.random() < 0.5:
                    judgments = judgments[::-1]
        
        # declare the response that got a score at all as the better one
        elif score_A is not None and score_B is None:
            pass
        elif score_A is None and score_B is not None:
            judgments = judgments[::-1]

        # both responses got no scores
        else:
            if random.random() < 0.5:
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
