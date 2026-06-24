"""
Reformats the ScalerLab/JudgeBench dataset to work with ../src/judge.py. Changes include:

- Splitting response pairs into separate samples
- Creating new columns based on existing ones:
    - response := response_{i}, where i is either A or B depending on which response is being kept
    - prompt_id := f"{pair_id}_{i}", where i follows above
    - prompt := question
- Original columns that will be kept: original_id, source, response_model, label

Example run command:

python -m benchmarks.JudgeBench-gpt.src.0-reformat-benchmark \
    --input-path ScalerLab/JudgeBench \
    --output-path benchmarks/JudgeBench-gpt/1-reformatted \
    --split gpt
"""

import argparse
import os

from datasets import Dataset, load_dataset
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, default="ScalerLab/JudgeBench")
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--split", type=str, required=True, default="gpt")
    args = parser.parse_args()

    # load the dataset
    print(f"Loading dataset from {args.input_path} split {args.split}")
    input_ds = load_dataset(args.input_path, split=args.split)

    print("Reformatting dataset...")
    output_ds = []
    for input_x in tqdm(input_ds):
        template_output_x = {
            key: input_x[key]
            for key in ["original_id", "source", "response_model", "label"]
        }

        for index in ["A", "B"]:
            output_x = template_output_x.copy()
            output_x.update({
                "response": input_x[f"response_{index}"],
                "prompt_id": f"{input_x['pair_id']}_{index}",
                "prompt": input_x["question"],
            })
            output_ds.append(output_x)
    output_ds = Dataset.from_list(output_ds)
        
    print(f"Saving dataset to {args.output_path}")
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    output_ds.save_to_disk(args.output_path)