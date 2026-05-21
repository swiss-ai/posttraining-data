"""
Run RewardBench-2 accuracy metrics on rereformatted judge outputs.

Non-Ties subsets: accuracy = mean of per-prompt result values
  (result was pre-computed in step 2: 1/k if chosen is among k max-scorers, else 0)

Ties subset: composite score via process_single_model
  (adapted from rewardbench/utils.py — added None-score handling)

Writes: benchmarks/RewardBench2-test/4-results/{judge}.json

Usage (from the 08a-judge-evaluation repo root):

    python -m benchmarks.RewardBench2-test.src.3-run-metrics
    python -m benchmarks.RewardBench2-test.src.3-run-metrics --judge 01
"""

from __future__ import annotations

import json
import os
from argparse import ArgumentParser
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from datasets import Dataset, load_from_disk

################################################################################
# Adapted from rewardbench/utils.py (compute_prompt_stats, process_single_model)
# Changes: uses noNone_scores instead of scores, where Nones are replaced by random scores

def _compute_prompt_stats(
    samples: List[Tuple[bool, float]],
) -> Optional[Tuple[bool, Optional[float], float]]:
    """
    (is_correct, score) tuples → (accurate, diff_correct_margin, correct_incorrect_margin).
    Returns None when there are not enough non-None scores to evaluate.
    """
    correct_scores = [s for is_corr, s in samples if is_corr and s is not None]
    incorrect_scores = [s for is_corr, s in samples if not is_corr and s is not None]

    if not correct_scores or not incorrect_scores:
        return None

    best_correct = max(correct_scores)
    worst_correct = min(correct_scores)
    best_incorrect = max(incorrect_scores)

    diff_correct_margin = (best_correct - worst_correct) if len(correct_scores) > 1 else None
    correct_incorrect_margin = worst_correct - best_incorrect
    accurate = correct_incorrect_margin > 0

    return accurate, diff_correct_margin, correct_incorrect_margin


def _process_ties(ties_ds: Dataset) -> Tuple[Dataset, float]:
    """
    Equivalent to rewardbench.utils.process_single_model.
    Expects columns: id (format "type:N"), scores (list), num_correct.
    """
    grouped: Dict[Tuple[str, int], List[Tuple[bool, Any]]] = defaultdict(list)

    for sample in ties_ds:
        sample_type, prompt_id_str = sample["id"].split(":", 1)
        prompt_id = int(prompt_id_str)
        for i, raw_score in enumerate(sample["noNone_scores"]):
            score = raw_score[0] if isinstance(raw_score, list) else raw_score
            grouped[(sample_type, prompt_id)].append((i < sample["num_correct"], score))

    ref_stats: Dict[int, Tuple] = {}
    tied_stats: Dict[int, Tuple] = {}

    for (sample_type, prompt_id), samples in grouped.items():
        stats = _compute_prompt_stats(samples)
        if stats is None:
            continue
        if sample_type == "ref":
            ref_stats[prompt_id] = stats
        else:
            tied_stats[prompt_id] = stats

    ref_accuracy = float(np.mean([s[0] for s in ref_stats.values()])) if ref_stats else 0.0
    tied_accuracy = float(np.mean([s[0] for s in tied_stats.values()])) if tied_stats else 0.0

    all_prompts = set(ref_stats) & set(tied_stats)

    if not all_prompts:
        overall_score = 0.30 * tied_accuracy + 0.30 * ref_accuracy
    else:
        # diff_correct_margin can be None for single-correct-response prompts → use 0.0
        diff_corr = np.array(
            [tied_stats[pid][1] if tied_stats[pid][1] is not None else 0.0 for pid in all_prompts]
        )
        corr_inc_ties = np.array([tied_stats[pid][2] for pid in all_prompts])
        corr_inc_ref = np.array([ref_stats[pid][2] for pid in all_prompts])

        correctness_preferred = float(np.mean(corr_inc_ties > diff_corr))
        correctness_preferred_hard = float(np.mean(np.minimum(corr_inc_ref, corr_inc_ties) > diff_corr))

        # Tie-breaking term (tanh of normalised gap); add eps to avoid /0 when diff_corr=0
        margin_scores = np.tanh(np.minimum(corr_inc_ref, corr_inc_ties) / (diff_corr + 1e-10) - 1)
        margin_scores = np.nan_to_num(margin_scores, nan=0.0)
        correctness_margin_score = float(np.mean(margin_scores))

        overall_score = (
            0.30 * tied_accuracy
            + 0.30 * ref_accuracy
            + 0.20 * correctness_preferred
            + 0.20 * correctness_preferred_hard
            + 0.01 * correctness_margin_score
        )

    if "results" in ties_ds.column_names:
        ties_ds = ties_ds.remove_columns(["results"])
    ties_ds = ties_ds.add_column("results", [None] * len(ties_ds))
    return ties_ds, float(overall_score)


################################################################################


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--judge-name", type=str, default=None)
    args = parser.parse_args()
    
    benchmark_root = "benchmarks/RewardBench2-test/"
    rereformatted_root = os.path.join(benchmark_root, "3-rereformatted")

    input_paths = sorted([
        os.path.join(rereformatted_root, x) 
        for x in os.listdir(rereformatted_root) if x.isdigit()
    ])

    if args.judge_name:
        input_paths = [p for p in input_paths if os.path.basename(p) == args.judge_name]

    for input_path in input_paths:
        print(f"Loading {input_path} ...")
        ds = load_from_disk(input_path)

        metrics: Dict[str, Any] = {}

        # ── Non-Ties subsets ──────────────────────────────────────────────────
        non_ties_ds = ds.filter(lambda x: x["subset"] != "Ties")
        subsets = sorted(set(non_ties_ds["subset"]))

        for subset in subsets:            
            sub_ds = non_ties_ds.filter(lambda x: x["subset"] == subset)
            none_rate = 100 * sum(any(score is None for score in x['scores']) for x in sub_ds ) / len(sub_ds)

            results = []
            for x in sub_ds:
                max_score = max(x["noNone_scores"])
                if x["noNone_scores"][0] == max_score:
                    results.append(1 / sum(score == max_score for score in x["noNone_scores"]))
                else:
                    results.append(0)

            metrics[subset] = {
                "score": float(np.mean(results)) * 100, "none_rate": none_rate,
            }

        # ── Ties subset ───────────────────────────────────────────────────────
        # step 2 already outputs an "id" column; _process_ties uses id, scores, num_correct
        ties_ds = ds.filter(lambda x: x["subset"] == "Ties")
        none_rate = 100 * sum(any(score is None for score in x['scores']) for x in ties_ds) / len(ties_ds)

        if len(ties_ds) > 0:
            _, ties_score = _process_ties(ties_ds)
            metrics["Ties"] = {
                "score": ties_score * 100,
                "none_rate": none_rate,
            }

        # ── Overall ───────────────────────────────────────────────────────────
        all_scores = [v for v in metrics.values()]
        metrics["overall"] = {
            "score": float(np.mean([d["score"] for d in metrics.values()])),
            "none_rate": float(np.mean([d["none_rate"] for d in metrics.values()])),
        }

        judge_name = os.path.basename(input_path)
        output_path = os.path.join(benchmark_root, f"4-results/{judge_name}.json")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        ds.cleanup_cache_files()
