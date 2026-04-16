"""
Reformats the output of the judge into paired rows for RM-Bench evaluation metrics. 

Usage:

python -m benchmarks.RM-Bench-train.src.2-reformat-post-judging \
    --judge-args-path judges/01.py
"""

import os
from argparse import ArgumentParser

import datasets
from datasets import Dataset, load_from_disk
datasets.disable_progress_bar()

from tqdm import tqdm

from src.utils import load_module

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--judge-args-path", type=str, required=True)
    args = parser.parse_args()

    # Load judge args
    judge_args = load_module(args.judge_args_path)

    # Determine input/output dirs based on judge name
    judge_name = os.path.basename(args.judge_args_path).rstrip(".py")
    input_dir = f"benchmarks/RM-Bench-train/2-judged/{judge_name}"
    output_dir = f"benchmarks/RM-Bench-train/3-rereformatted/{judge_name}"

    print(f"Loading dataset from {input_dir}")
    input_ds = load_from_disk(input_dir)

    print("Reformatting dataset...")
    output_ds = []
    
    original_prompt_ids = set(
        x.split("_")[0] for x in input_ds["prompt_id"]
    )
    for original_prompt_id in tqdm(original_prompt_ids):
        subset = input_ds.filter(lambda x: x["prompt_id"].startswith(original_prompt_id))
        temp = {
            "id": original_prompt_id,
            "prompt": subset[0]["prompt"],
            "domain": subset[0]["domain"],
            "chosen": [],
            "rejected": [],
            "score_chosen": [],
            "score_rejected": [],
        }

        for key in ["concise", "plain", "markdown"]:
            x = subset.filter(lambda x: x["prompt_id"] == f"{original_prompt_id}_chosen_{key}")[0]
            score = judge_args.get_score_from_distribution(x["score_distribution"])
            temp["chosen"].append(x["response"])
            temp["score_chosen"].append(score)

            x = subset.filter(lambda x: x["prompt_id"] == f"{original_prompt_id}_rejected_{key}")[0]
            score = judge_args.get_score_from_distribution(x["score_distribution"])
            temp["rejected"].append(x["response"])
            temp["score_rejected"].append(score)

        output_ds.append(temp)
    
    output_ds = Dataset.from_list(output_ds)
    input_ds.cleanup_cache_files()

    print(f"Saving dataset to {output_dir}")
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    output_ds.save_to_disk(output_dir)
