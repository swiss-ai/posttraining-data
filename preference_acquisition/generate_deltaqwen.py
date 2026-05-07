"""
Generate delta-Qwen preference pairs from the combined annotated dataset.

For each row, picks Qwen3-32B as chosen and Qwen3-0.6B as rejected,
using their responses from the model_evaluations column.
"""

import os

from datasets import DatasetDict, Features, Value, load_from_disk

DATASET_PATH = "/iopsstor/scratch/cscs/dmelikidze/posttraining-data/response_annotation/datasets/combined_annotated"
OUTPUT_PATH = "/iopsstor/scratch/cscs/dmelikidze/posttraining-data/preference_acquisition/datasets/Qwen3-32B_vs_0.6B"

CHOSEN_MODEL = "Qwen3-32B"
REJECTED_MODEL = "Qwen3-0.6B"
NUM_PROC = min(os.cpu_count() or 4, 288)

print(f"Using {NUM_PROC} processes.")

dataset = load_from_disk(DATASET_PATH)
if "train" in dataset:
    dataset = dataset["train"]

print(f"Loaded dataset: {len(dataset)} rows, columns: {dataset.column_names}")


def extract_deltaqwen(batch):
    out = {
        "prompt_id": [],
        "chosen": [],
        "rejected": [],
        "chosen_model": [],
        "rejected_model": [],
        "chosen_score": [],
        "rejected_score": [],
    }

    for prompt_msgs_raw, prompt_id, evaluations in zip(
        batch["prompt"], batch["prompt_id"], batch["model_evaluations"]
    ):
        eval_by_model = {a["model"]: a for a in evaluations}

        if CHOSEN_MODEL not in eval_by_model or REJECTED_MODEL not in eval_by_model:
            continue

        chosen = eval_by_model[CHOSEN_MODEL]
        rejected = eval_by_model[REJECTED_MODEL]

        prompt_msgs = [{"role": m["role"], "content": m["content"]} for m in prompt_msgs_raw]

        out["prompt_id"].append(prompt_id)
        out["chosen"].append(
            prompt_msgs + [{"role": "assistant", "content": chosen["response"]}]
        )
        out["rejected"].append(
            prompt_msgs + [{"role": "assistant", "content": rejected["response"]}]
        )
        out["chosen_model"].append(CHOSEN_MODEL)
        out["rejected_model"].append(REJECTED_MODEL)
        out["chosen_score"].append(float(chosen["final_score"]))
        out["rejected_score"].append(float(rejected["final_score"]))

    return out


output_features = Features({
    "prompt_id": Value("string"),
    "chosen": [{"role": Value("string"), "content": Value("string")}],
    "rejected": [{"role": Value("string"), "content": Value("string")}],
    "chosen_model": Value("string"),
    "rejected_model": Value("string"),
    "chosen_score": Value("float64"),
    "rejected_score": Value("float64"),
})

print("Extracting delta-Qwen preference pairs ...", flush=True)
processed = dataset.map(
    extract_deltaqwen,
    batched=True,
    batch_size=2000,
    num_proc=NUM_PROC,
    remove_columns=dataset.column_names,
    features=output_features,
    desc="Extracting delta-Qwen pairs",
)

processed = processed.filter(lambda x: x["prompt_id"] is not None)

final_dataset_dict = DatasetDict({"train_split": processed})

print(f"Created preference dataset with {len(processed)} examples.")
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
final_dataset_dict.save_to_disk(OUTPUT_PATH)
print(f"Saved to {OUTPUT_PATH}")
