"""
Run metrics on the JudgeBench-gpt dataset.

Usage (from the 08a-judge-evaluation repo root):

    python -m benchmarks.JudgeBench-gpt.src.3-run-metrics
    python -m benchmarks.JudgeBench-gpt.src.3-run-metrics --judge 01
"""

from argparse import ArgumentParser
from datasets import load_from_disk
import os
from typing import List, Dict, Any, Optional
import json

################################################################################
# copy-paste of JudgeBench.utils.metrics

from typing import List, Dict, Any

# compute final metrics
# basically just compute how accuracy the judge is

def flip_judgment(decision: str) -> str:
    if decision == "A>B":
        decision = "B>A"
    elif decision == "B>A":
        decision = "A>B"
    return decision


def compute_final_metrics(pairs: List[Dict[str, Any]], reverse_order: bool) -> None:
    n_pairs = len(pairs)

    if not reverse_order:
        n_correct = sum(
            pair["judgments"][0]["decision"] == pair["label"]
            for pair in pairs
        )
        n_incorrect = n_pairs - n_correct
        return 100*n_correct/n_pairs
        
    else:
        
        n_all_correct = 0
        n_all_incorrect = 0
        n_some_correct = 0
        
        n_correct = 0
        n_incorrect = 0
        n_tie = 0
        
        n_nulls = 0
        n_inconsistent = 0
        
        for pair in pairs:
            
            label = pair["label"]
            judgment1, judgment2 = pair["judgments"]
        
            decision1 = judgment1["decision"] if judgment1 is not None else None
            decision2 = flip_judgment(judgment2["decision"] if judgment2 is not None else None)
            
            if decision1 is None or decision2 is None:
                n_nulls += 1
            
            # consistency metrics
            if decision1 == label and decision2 == label:
                n_all_correct += 1
            elif decision1 != label and decision2 != label:
                n_all_incorrect += 1
            else:
                n_some_correct += 1
                
            if decision1 != decision2:
                n_inconsistent += 1
                
            # new metrics
            counter = 0
            for decision in [decision1, decision2]:
                if decision == label:
                    counter += 1
                elif decision == flip_judgment(label):
                    counter -= 1
                
            if counter > 0:
                n_correct += 1
            elif counter < 0:
                n_incorrect += 1
            else:
                n_tie += 1
        
        return 100*n_correct/n_pairs

################################################################################


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--judge-name", type=str, default=None)
    args = parser.parse_args()
    
    benchmark_root = "benchmarks/JudgeBench-gpt/"
    rereformatted_root = os.path.join(benchmark_root, "3-rereformatted")
    input_paths = sorted([
        os.path.join(rereformatted_root, x) 
        for x in os.listdir(rereformatted_root) if x.isdigit()
    ])

    if args.judge_name:
        input_paths = [p for p in input_paths if os.path.basename(p) == args.judge_name]

    for input_path in input_paths:
        print(f"Loading {input_path} ...")
        input_ds = load_from_disk(input_path)

        print("Running metrics...")
        metrics_dict = {}
        for source in ["mmlu-pro", "livebench-reasoning", "livebench-math", "livecodebench", ""]:
            dataset_name = "overall" if source == "" else source
            sub_ds = input_ds.filter(lambda x, s=source: x["source"].startswith(s))
            none_count = len(sub_ds.filter(lambda x: x["score_A"] is None or x["score_B"] is None))
            none_rate = none_count / len(sub_ds) if len(sub_ds) > 0 else 0.0
            score = compute_final_metrics(sub_ds, reverse_order=True)
            metrics_dict[dataset_name] = {"score": score, "none_rate": none_rate}

        judge_name = os.path.basename(input_path)
        output_path = os.path.join(benchmark_root, f"4-results/{judge_name}.json")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        print(f"  Writing {output_path}")
        with open(output_path, "w") as f:
            json.dump(metrics_dict, f, indent=4)

        input_ds.cleanup_cache_files()