"""
Run RM-Bench accuracy metrics (hard / normal / easy and domain aggregates) on
rereformatted judge outputs, using :func:`scripts.utils.compute_accuracy` from
the THU-KEG/RM-Bench submodule.

Usage (from the ``08a-judge-evaluation`` repo root)::

    python -m benchmarks.RM-Bench-train.src.3-run-metrics
    python -m benchmarks.RM-Bench-train.src.3-run-metrics --judge-name 01-ActiveUF-Helpfulness
"""

from __future__ import annotations

import json
import os
import sys
from argparse import ArgumentParser
from typing import Any, Dict, List

import numpy as np
from datasets import load_from_disk

################################################################################
# Copy-paste from THU-KEG/RM-Bench (slightly modified to report number of skipped comparisons)
def split_dataset_by_domain(dataset: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    domains = ["chat","math","code","safety"]
    domain_dataset_dict = {}
    for domain in domains:
        domain_dataset_dict[domain] = [example for example in dataset if example['domain'].startswith(domain)]
    
    # pop the domain keys
    for domain in domain_dataset_dict:
        for example in domain_dataset_dict[domain]:
            example.pop('domain')
    
    return domain_dataset_dict
    
def compute_accuracy(results: List[Dict[str, Any]]) -> Dict[str, float]:
    if 'domain' in results[0]:
        # this indicates this is total_dataset.json
        print('We are handling total_dataset.json')
        print('Splitting the dataset by domain...')
        # thus we need to split the results into different domains
        split_results = split_dataset_by_domain(results)
        domain_results = {}
        for domain in split_results:
            domain_results[domain] = compute_accuracy(split_results[domain])
            print(domain, domain_results[domain])
        domain_avg_results = {}
        for domain in domain_results:
            domain_avg_results[domain] = {
                "none_rate": domain_results[domain]["none_rate"],
                "acc": np.mean(list(
                    val for key, val in domain_results[domain].items() if key not in ["none_rate", "acc"]
        ))}
            print(domain, domain_avg_results[domain])
        domain_hard_normal_easy_acc = {
            "hard_acc": np.mean([domain_results[domain]["hard_acc"] for domain in domain_results]),
            "normal_acc": np.mean([domain_results[domain]["normal_acc"] for domain in domain_results]),
            "easy_acc": np.mean([domain_results[domain]["easy_acc"] for domain in domain_results]),
        }
        total_avg_acc = np.mean([domain_avg_results[domain]["acc"] for domain in domain_avg_results])
        # merge the results into one falten dictionary
        final_results = {}
        # merge domain_avg_results into final_results
        final_results.update(domain_avg_results)
        # merge domain_hard_normal_easy_acc into final_results
        final_results.update(domain_hard_normal_easy_acc)
        # merge total_avg_acc into final_results
        final_results.update({"total_avg_acc": total_avg_acc})
        return final_results
            
    
    # results is a list of dictionaries, each dictionary contains the following keys:
    # score_chosen: [float, float, float], the scores of the chosen responses
    # score_rejected: [float, float, float], the scores of the rejected responses
    # the scores are in the order of [concise, detailed_plain, detailed_markdown]
    # we will compare the scores of chosen responses and rejected responses iteratively
    # formatted as a 3x3 matrix, where the rows represent the scores of chosen responses
    # and the columns represent the scores of rejected responses
    MATRIX_SIZE = 3 # the column and row size of the matrix
    acc_matrix = np.zeros((MATRIX_SIZE, MATRIX_SIZE))
    count_matrix = np.zeros((MATRIX_SIZE, MATRIX_SIZE))
    total_n_comparisons = 0
    n_random_comparisons = 0
    for result in results:
        for i in range(len(result["score_chosen"])):
            for j in range(len(result["score_rejected"])):
                total_n_comparisons += 1
                if result["score_chosen"][i] is None or result["score_rejected"][j] is None:
                    n_random_comparisons += 1

                count_matrix[i][j] += 1
                if result["noNone_score_chosen"][i] > result["noNone_score_rejected"][j]:
                    acc_matrix[i][j] += 1

    # divide each cell by the number of non-None comparisons for that cell (exclude rather than penalise)
    with np.errstate(invalid="ignore"):
        acc_matrix = np.where(count_matrix > 0, acc_matrix / count_matrix, 0.0)
    # compute the hard,normal,easy accuracy
    # hard accuracy: the average of the upper-right triangle of the matrix
    # namely chosen responses with less fancy style compared to rejected responses with more fancy style
    upper_right_count = MATRIX_SIZE * (MATRIX_SIZE - 1) / 2
    hard_acc = np.sum(np.triu(acc_matrix, 1)) / upper_right_count
    # normal accuracy: the average of the diagonal of the matrix
    # namely chosen responses with the same style compared to rejected responses with the same style
    normal_acc = np.mean(np.diag(acc_matrix))
    # easy accuracy: the average of the lower-left triangle of the matrix
    # namely chosen responses with more fancy style compared to rejected responses with less fancy style
    lower_left_count = MATRIX_SIZE * (MATRIX_SIZE - 1) / 2
    easy_acc = np.sum(np.tril(acc_matrix, -1)) / lower_left_count

    return {
        "hard_acc": hard_acc,
        "normal_acc": normal_acc,
        "easy_acc": easy_acc,
        "none_rate": 100 * n_random_comparisons / total_n_comparisons,
    }
################################################################################


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--judge-name", type=str, default=None)
    args = parser.parse_args()

    benchmark_root = "benchmarks/RM-Bench-train/"
    rereformatted_root = os.path.join(benchmark_root, "3-rereformatted")

    input_paths = sorted([
        os.path.join(rereformatted_root, x)
        for x in os.listdir(rereformatted_root) if x[0].isdigit()
    ])

    if args.judge_name:
        input_paths = [p for p in input_paths if os.path.basename(p) == args.judge_name]

    for input_path in input_paths:
        judge_name = os.path.basename(input_path)
        print(f"Loading {input_path} ...")
        input_ds = load_from_disk(input_path)
        results = input_ds.to_list()
        if not results:
            print(f"  skip (empty): {judge_name}")
            input_ds.cleanup_cache_files()
            continue

        print(f"  Computing accuracy ({len(results)} prompts) ...")
        metrics: Dict[str, Any] = compute_accuracy(results)

        output_path = os.path.join(benchmark_root, f"4-results/{judge_name}.json")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        print(f"  Writing {output_path}")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        input_ds.cleanup_cache_files()
