"""

"Run metrics on the JudgeBench-gpt dataset.

Usage:
python -m benchmarks.JudgeBench-gpt.src.3-run-metrics \
    --input-dir benchmarks/JudgeBench-gpt/3-rereformatted/01
"""

from argparse import ArgumentParser
from datasets import load_from_disk
import os
from typing import List, Dict, Any
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


def compute_final_metrics(pairs: List[Dict[str, Any]], reverse_order: bool, include_fn=lambda x: x) -> None:
    
    pairs = [pair for pair in pairs if include_fn(pair)]

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
    parser.add_argument("--input-dir", type=str)
    args = parser.parse_args()

    print(f"Loading dataset from {args.input_dir}")
    input_ds = load_from_disk(args.input_dir)

    print("Running metrics...")
    metrics_dict = {}
    for source in ["mmlu-pro", "livebench-reasoning", "livebench-math", "livecodebench", ""]:
        if source == "":
            dataset_name = "overall"
        else:
            dataset_name = source
        metrics_dict[dataset_name] = compute_final_metrics(
            input_ds, 
            reverse_order=True, 
            include_fn=lambda x: x["source"].startswith(source),
        )

    # additionally compute no-score and tie rates
    metrics_dict["two-None rate"] = len(input_ds.filter(lambda x: x["score_A"] is None and x["score_B"] is None)) / len(input_ds) * 100
    metrics_dict["one-None rate"] = len(input_ds.filter(lambda x: x["score_A"] is None or x["score_B"] is None)) / len(input_ds) * 100
    metrics_dict["tie rate"] = len(input_ds.filter(lambda x: x["score_A"] == x["score_B"])) / len(input_ds) * 100
    metrics_dict["tie rate (among zero-None)"] = len(input_ds.filter(lambda x: x["score_A"] is not None and x["score_B"] is not None and x["score_A"] == x["score_B"])) / len(input_ds) * 100

    # write scores to disk
    filename = os.path.basename(args.input_dir)
    output_path = f"benchmarks/JudgeBench-gpt/4-results/{filename}.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Writing scores to {output_path}")
    with open(output_path, "w") as f:
        json.dump(metrics_dict, f, indent=4)

    input_ds.cleanup_cache_files()