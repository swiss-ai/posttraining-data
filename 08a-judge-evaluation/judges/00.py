"""
This judge is unlike the others of this repo. It takes the final scores of judges 01-04 and returns the mean score.
This mean computation is done per benchmark.

Usage:

python -m judges.00 --benchmark <benchmark>
"""

import argparse
from collections import defaultdict
import copy
from datasets import load_from_disk
from src.utils import load_module

judge_name = "00"
component_judge_names = ["01", "02", "03", "04"]

def get_score_from_distribution(score_distribution: dict[str, float]) -> float:
    # there is only one value in the score distribution dict!
    return list(score_distribution.values())[0]


if __name__ == "__main__":
    # only run this script directly when creating the judge responses file for 2-reformat-post-judging
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, required=True)
    args = parser.parse_args()

    prompt_id2component_scores = defaultdict(list)
    for component_judge_name in component_judge_names:
        component_judge_cfg = load_module(f"judges/{component_judge_name}.py")

        filepath = f"benchmarks/{args.benchmark}/2-judged/{component_judge_name}"
        component_ds = load_from_disk(filepath)
        if prompt_id2component_scores:
            assert set(prompt_id2component_scores.keys()) == set(component_ds["prompt_id"])

        for x in component_ds:
            prompt_id2component_scores[x["prompt_id"]].append(
                component_judge_cfg.get_score_from_distribution(x["score_distribution"])
            )
        
    score_distributions = []
    for x in component_ds:
        component_scores = prompt_id2component_scores[x["prompt_id"]]
        mean_score = sum(component_scores) / len(component_scores)
        score_distributions.append({"mean-of-01-to-04": mean_score})

    output_ds = copy.deepcopy(component_ds)
    output_ds = output_ds.remove_columns("score_distribution")
    output_ds = output_ds.add_column("score_distribution", score_distributions)
    output_ds.save_to_disk(f"benchmarks/{args.benchmark}/2-judged/{judge_name}")