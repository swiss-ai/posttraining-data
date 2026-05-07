"""
Reformats allenai/reward-bench-2 (test split) for use with src/judge.py.

Each row is expanded into one row per response:
  - For non-Ties subsets: 1 chosen + 3 rejected = 4 rows (positions 0-3)
  - For the Ties subset:  len(chosen) + len(rejected) rows (variable)

Encoding: prompt_id = "{rb2_id}_resp_{i}"
  - i = 0 .. num_correct-1:  the "correct" (chosen) responses
  - i = num_correct .. total-1: the "incorrect" (rejected) responses

Columns in the output dataset:
  prompt, response, prompt_id, subset, rb2_id, num_correct

Example run command (from the 08a-judge-evaluation repo root):

python -m benchmarks.RewardBench2-test.src.0-reformat-benchmark \
  --input-path allenai/reward-bench-2 \
  --output-path benchmarks/RewardBench2-test/1-reformatted
"""

import argparse
import os

from datasets import Dataset, load_dataset
from tqdm import tqdm

from src.utils import stringify_prompt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, default="allenai/reward-bench-2")
    parser.add_argument("--output-path", type=str, required=True)
    args = parser.parse_args()

    print(f"Loading dataset from {args.input_path} split test")
    input_ds = load_dataset(args.input_path, split="test")

    print("Reformatting dataset...")
    output_ds = []
    for input_x in tqdm(input_ds):
        rb2_id = input_x["id"]
        subset = input_x["subset"]
        num_correct = input_x["num_correct"]
        prompt = stringify_prompt(input_x["prompt"])

        # Ordered list: chosen responses first (indices 0..num_correct-1), then rejected
        for key in ["chosen", "rejected"]:
          for i, response in enumerate(input_x[key]):
              output_ds.append({
                  "prompt": prompt,
                  "response": response,
                  "prompt_id": f"{subset}_{rb2_id}_{key}_{i}",
                  "subset": subset,
                  "rb2_id": rb2_id,
                  "num_correct": num_correct,
              })

    output_ds = Dataset.from_list(output_ds)
    print(f"Expanded {len(input_ds)} prompts → {len(output_ds)} response rows")

    print(f"Saving dataset to {args.output_path}")
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    output_ds.save_to_disk(args.output_path)
