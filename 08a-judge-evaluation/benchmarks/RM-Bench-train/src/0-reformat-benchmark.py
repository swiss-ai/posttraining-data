"""
    Reformat the THU-KEG/RM-Bench dataset for our judge. Changes include:

    - Splitting response sextuplets into separate samples
    - Creating new columns based on existing ones:
        - response := one of the six responses associated with the current prompt
        - prompt_id := f"{id}_{label}_resp_{i}", where label = chosen or rejected, and i = 1,2,3
    - Original columns that will be kept: domain, prompt

Example run command:

python -m benchmarks.RM-Bench-train.src.0-reformat-benchmark \
    --input-path THU-KEG/RM-Bench \
    --output-path benchmarks/RM-Bench-train/1-reformatted \
    --split train
"""

import argparse
import os

from datasets import Dataset, load_dataset
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, default="THU-KEG/RM-Bench")
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--split", type=str, required=True, default="train")
    args = parser.parse_args()

    # load the dataset
    print(f"Loading dataset from {args.input_path} split {args.split}")
    input_ds = load_dataset(args.input_path, split=args.split)

    print("Reformatting dataset...")
    keys = ["concise", "plain", "markdown"]
    output_ds = []
    for input_x in tqdm(input_ds):
        template_output_x = {
            "prompt": input_x['prompt'],
            "domain": input_x["domain"],
        }
        for label in ["chosen", "rejected"]:
            for key, response in zip(keys, input_x[label]):
                output_x = template_output_x.copy()
                output_x.update({
                    "response": response,
                    "prompt_id": f"{input_x['id']}_{label}_{key}",
                })
                output_ds.append(output_x)
    output_ds = Dataset.from_list(output_ds)
        
    print(f"Saving dataset to {args.output_path}")
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    output_ds.save_to_disk(args.output_path)