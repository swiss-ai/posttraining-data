"""
Reformats the output of the judge into paired rows for RM-Bench evaluation metrics.

Usage (from the 08a-judge-evaluation repo root):

    python -m benchmarks.RM-Bench-train.src.2-reformat-post-judging
    python -m benchmarks.RM-Bench-train.src.2-reformat-post-judging --judge-name 01
"""

import concurrent.futures
import os
import random
from argparse import ArgumentParser

import datasets
from datasets import Dataset, load_from_disk
datasets.disable_progress_bar()

from tqdm import tqdm

from src.utils import load_module


def process_judge(input_path: str, benchmark_root: str) -> None:
    judge_name = os.path.basename(input_path)
    judge_args_path = f"judges/{judge_name}.py"
    if not os.path.exists(judge_args_path):
        print(f"Skipping {judge_args_path}: file not found")
        return

    judge_args = load_module(judge_args_path)

    output_dir = os.path.join(benchmark_root, f"3-rereformatted/{judge_name}")

    input_ds = load_from_disk(input_path)
    output_ds = []

    original_prompt_ids = set(
        x.split("_")[0] for x in input_ds["prompt_id"]
    )
    for original_prompt_id in tqdm(original_prompt_ids, desc=judge_name):
        subset = input_ds.filter(lambda x: x["prompt_id"].startswith(original_prompt_id))
        temp = {
            "id": original_prompt_id,
            "prompt": subset[0]["prompt"],
            "domain": subset[0]["domain"],
            "chosen": [],
            "rejected": [],
            "score_chosen": [],
            "score_rejected": [],
            "noNone_score_chosen": [],
            "noNone_score_rejected": [],
        }

        for key in ["concise", "plain", "markdown"]:
            x = subset.filter(lambda x: x["prompt_id"] == f"{original_prompt_id}_chosen_{key}")[0]
            score = judge_args.get_score_from_distribution(x["score_distribution"])
            temp["chosen"].append(x["response"])
            temp["score_chosen"].append(score)
            if score is not None:
                temp["noNone_score_chosen"].append(score)
            else:
                temp["noNone_score_chosen"].append(float(random.sample(judge_args.scoring_range, 1)[0]))

            x = subset.filter(lambda x: x["prompt_id"] == f"{original_prompt_id}_rejected_{key}")[0]
            score = judge_args.get_score_from_distribution(x["score_distribution"])
            temp["rejected"].append(x["response"])
            temp["score_rejected"].append(score)
            if score is not None:
                temp["noNone_score_rejected"].append(score)
            else:
                temp["noNone_score_rejected"].append(float(random.sample(judge_args.scoring_range, 1)[0]))

        output_ds.append(temp)

    output_ds = Dataset.from_list(output_ds)
    input_ds.cleanup_cache_files()

    print(f"Saving dataset to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    output_ds.save_to_disk(output_dir)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--judge-name", type=str, default=None)
    args = parser.parse_args()

    benchmark_root = "benchmarks/RM-Bench-train/"
    judged_root = os.path.join(benchmark_root, "2-judged")

    input_paths = sorted([
        os.path.join(judged_root, x)
        for x in os.listdir(judged_root) if x.isdigit()
    ])

    if args.judge_name:
        input_paths = [p for p in input_paths if os.path.basename(p) == args.judge_name]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_judge, p, benchmark_root): p for p in input_paths}
        for future in concurrent.futures.as_completed(futures):
            future.result()
